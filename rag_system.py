"""
基于 ChromaDB 的本地持久化向量知识库（RAG 检索层）。

使用 ChromaDB 内置默认嵌入函数（当前实现多为 ONNX ``all-MiniLM-L6-v2``），
首次运行会下载模型（约数十 MB）到项目内 ``.cache/chroma/onnx_models/``（见下方重定向逻辑），需磁盘与网络权限。

本模块不构成医疗建议；检索结果仅供 Agent 辅助决策。
"""

from __future__ import annotations

import logging
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, List, Optional

# -----------------------------------------------------------------------------
# ChromaDB：延迟导入失败时给出明确提示（便于在未安装依赖时快速定位问题）
# -----------------------------------------------------------------------------
try:
    import chromadb
except ImportError as exc:  # pragma: no cover - 仅在缺依赖时触发
    raise ImportError(
        "未安装 chromadb。请先执行：pip install chromadb"
    ) from exc

logger = logging.getLogger(__name__)


def _print_ingest_progress(
    filename: str,
    chunk_one_based: int,
    total_chunks: int,
    *,
    collection_label: str,
) -> None:
    """终端可见的入库进度（立即 flush，避免长时间无输出）。"""
    print(
        f"  [RAG 入库] 集合 {collection_label!r} | 文件 {filename!r} | "
        f"当前进度：第 {chunk_one_based}/{total_chunks} 块",
        file=sys.stdout,
        flush=True,
    )


def _should_emit_chunk_progress(idx: int, n: int) -> bool:
    """块数很多时降频，避免刷屏（仍保证首、末与约 10 个中间点）。"""
    if n <= 0:
        return False
    if n <= 40:
        return True
    if idx == 0 or idx == n - 1:
        return True
    step: int = max(1, n // 10)
    return (idx + 1) % step == 0


def _redirect_onnx_embedding_cache_to_project() -> None:
    """
    将 Chroma 默认 ONNX 嵌入模型的下载目录从 ``~/.cache/chroma/...`` 改到本文件旁
    ``.cache/chroma/onnx_models/<model>``。

    这样在 CI、沙箱或无家目录写权限环境下仍可缓存模型；与默认嵌入权重一致。
    """
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )

        _pkg_root = Path(__file__).resolve().parent
        ONNXMiniLM_L6_V2.DOWNLOAD_PATH = (
            _pkg_root / ".cache" / "chroma" / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME
        )
    except Exception as exc:
        logger.warning(
            "无法重定向 ONNX 模型缓存路径，将回退 Chroma 默认（通常需写 ~/.cache）：%s",
            exc,
        )


# 在创建任何 Collection、触发嵌入之前完成重定向
_redirect_onnx_embedding_cache_to_project()

# 持久化目录（相对于进程工作目录；也可用绝对路径）
DEFAULT_CHROMA_DIR: Final[str] = "./chroma_db"

# 知识库集合名称（协议 / 症状矩阵）
COLLECTION_NAME: Final[str] = "psmf_knowledge"

# 食谱食材集合（独立 Collection）
COLLECTION_FOOD: Final[str] = "food_db"

# 默认要入库的 Markdown 文件名（与 rag_system.py 同目录，或见 base_dir）
DEFAULT_SOURCE_FILES: Final[tuple[str, ...]] = (
    "psmf_core_protocol.md",
    "symptom_diagnostic_matrix.md",
    "psmf_training_guide.md",
    "psmf_micronutrient_electrolyte_guide.md",
)

# 食谱知识库单文件
FOOD_SOURCE_FILES: Final[tuple[str, ...]] = ("psmf_food_database.md",)

# 文本切块参数：按 Markdown 标题层级进行语义切块，表格保持完整，超长块再回退到滑窗
RAG_CHUNKING_VERSION: Final[str] = "markdown_heading_v2"
DEFAULT_CHUNK_SIZE: Final[int] = 900
DEFAULT_CHUNK_OVERLAP: Final[int] = 120
DEFAULT_INGEST_BATCH_SIZE: Final[int] = 16
_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class MarkdownChunk:
    """A retrieval chunk with heading metadata for explainable RAG."""

    text: str
    section_path: str
    section_title: str
    heading_level: int


@dataclass
class _MarkdownSection:
    heading_path: tuple[tuple[int, str], ...]
    blocks: List[str]


def _is_markdown_heading(line: str) -> bool:
    return _parse_markdown_heading(line) is not None


def _clean_heading_title(title: str) -> str:
    cleaned: str = title.strip().strip("#").strip()
    cleaned = re.sub(r"^[*_`]+|[*_`]+$", "", cleaned).strip()
    cleaned = cleaned.replace("\\_", "_").replace("\\)", ")")
    return cleaned or "Untitled Section"


def _parse_markdown_heading(line: str) -> Optional[tuple[int, str]]:
    match = _HEADING_RE.match(line.strip())
    if not match:
        return None
    return len(match.group(1)), _clean_heading_title(match.group(2))


def _is_markdown_table_line(line: str) -> bool:
    s: str = line.strip()
    return len(s) >= 2 and s.startswith("|") and s.endswith("|")


def _split_markdown_sections(text: str) -> List[_MarkdownSection]:
    """Split Markdown into heading-scoped sections while keeping tables together."""
    sections: List[_MarkdownSection] = []
    heading_stack: List[tuple[int, str]] = []
    block_buf: List[str] = []
    table_buf: List[str] = []
    section_blocks: List[str] = []

    def flush_block() -> None:
        nonlocal block_buf
        block = "\n".join(block_buf).strip()
        if block:
            section_blocks.append(block)
        block_buf = []

    def flush_table() -> None:
        nonlocal table_buf
        block = "\n".join(table_buf).strip()
        if block:
            section_blocks.append(block)
        table_buf = []

    def flush_section() -> None:
        nonlocal section_blocks
        if section_blocks:
            sections.append(
                _MarkdownSection(
                    heading_path=tuple(heading_stack),
                    blocks=section_blocks,
                )
            )
            section_blocks = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        if _is_markdown_table_line(line):
            flush_block()
            table_buf.append(line)
            continue

        if table_buf:
            flush_table()

        heading = _parse_markdown_heading(line)
        if heading is not None:
            flush_block()
            flush_section()
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if not line.strip():
            flush_block()
            continue

        block_buf.append(line)

    flush_block()
    if table_buf:
        flush_table()
    flush_section()
    return sections


def _fixed_window_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    chunks: List[str] = []
    start: int = 0
    n: int = len(text)
    while start < n:
        end: int = min(start + chunk_size, n)
        piece: str = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _is_table_block(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines) and all(_is_markdown_table_line(line) for line in lines)


def _split_table_block(block: str, max_chars: int) -> List[str]:
    """Split a large Markdown table by rows, repeating the table header."""
    rows = [row.rstrip() for row in block.splitlines() if row.strip()]
    if len(rows) <= 2:
        return _fixed_window_chunks(block, chunk_size=max_chars, overlap=0)

    header_len = 2 if re.match(r"^\|\s*:?-{3,}", rows[1].strip()) else 1
    header = rows[:header_len]
    body_rows = rows[header_len:]
    parts: List[str] = []
    current: List[str] = list(header)

    for row in body_rows:
        candidate = "\n".join(current + [row])
        if len(candidate) <= max_chars or len(current) == header_len:
            current.append(row)
            continue

        part = "\n".join(current).strip()
        if len(part) > max_chars:
            parts.extend(_fixed_window_chunks(part, chunk_size=max_chars, overlap=0))
        elif part:
            parts.append(part)
        current = list(header) + [row]

    part = "\n".join(current).strip()
    if len(part) > max_chars:
        parts.extend(_fixed_window_chunks(part, chunk_size=max_chars, overlap=0))
    elif part:
        parts.append(part)
    return parts


def _split_large_block(block: str, max_chars: int, overlap: int) -> List[str]:
    """Split only when a single semantic block is too large for one chunk."""
    safe_max = max(200, max_chars)
    if _is_table_block(block):
        return _split_table_block(block, safe_max)
    return _fixed_window_chunks(block, chunk_size=safe_max, overlap=overlap)


def _semantic_tail(text: str, overlap: int) -> str:
    """Return a paragraph-aligned tail for continuity between adjacent chunks."""
    if overlap <= 0:
        return ""
    paras: List[str] = [p.strip() for p in text.split("\n\n") if p.strip()]
    tail: List[str] = []
    total = 0
    for para in reversed(paras):
        plen = len(para) + (2 if tail else 0)
        if tail and total + plen > overlap:
            break
        if plen > overlap * 2:
            break
        tail.insert(0, para)
        total += plen
    return "\n\n".join(tail).strip()


def _format_section_prefix(heading_path: tuple[tuple[int, str], ...]) -> str:
    return "\n".join(
        f"{'#' * min(level, 6)} {title}" for level, title in heading_path
    ).strip()


def _section_path_text(heading_path: tuple[tuple[int, str], ...]) -> str:
    return " > ".join(title for _, title in heading_path) or "Document"


def _build_chunk_text(prefix: str, body: str) -> str:
    body = body.strip()
    if not prefix:
        return body
    if not body:
        return prefix
    return f"{prefix}\n\n{body}".strip()


def _chunk_section(
    section: _MarkdownSection,
    *,
    chunk_size: int,
    overlap: int,
) -> List[MarkdownChunk]:
    """Create chunks inside a single heading section."""
    prefix = _format_section_prefix(section.heading_path)
    section_path = _section_path_text(section.heading_path)
    section_title = section.heading_path[-1][1] if section.heading_path else "Document"
    heading_level = section.heading_path[-1][0] if section.heading_path else 0
    max_body_chars = max(200, chunk_size - len(prefix) - 2)
    chunks: List[MarkdownChunk] = []
    current_body = ""

    def append_body(body: str) -> None:
        text = _build_chunk_text(prefix, body)
        if text:
            chunks.append(
                MarkdownChunk(
                    text=text,
                    section_path=section_path,
                    section_title=section_title,
                    heading_level=heading_level,
                )
            )

    def flush_current() -> None:
        nonlocal current_body
        if current_body.strip():
            append_body(current_body)
        current_body = ""

    for block in section.blocks:
        block = block.strip()
        if not block:
            continue

        if len(_build_chunk_text(prefix, block)) > chunk_size:
            flush_current()
            for piece in _split_large_block(block, max_body_chars, overlap):
                append_body(piece)
            continue

        candidate_body = block if not current_body else current_body + "\n\n" + block
        if len(_build_chunk_text(prefix, candidate_body)) <= chunk_size:
            current_body = candidate_body
            continue

        previous_body = current_body
        flush_current()
        tail = _semantic_tail(previous_body, min(overlap, max_body_chars // 2))
        candidate_body = tail + "\n\n" + block if tail else block
        current_body = (
            candidate_body
            if len(_build_chunk_text(prefix, candidate_body)) <= chunk_size
            else block
        )

    flush_current()
    return chunks


def _chunk_markdown(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[MarkdownChunk]:
    """
    将 Markdown 文档切成按标题层级组织的语义块。

    每个 chunk 都携带当前 ``# / ## / ###`` 标题路径，并在正文前重复标题上下文；
    表格按行保留，只有超长段落或超长表格才回退到滑窗切分。
    """
    cleaned: str = text.strip()
    if not cleaned:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    sections = _split_markdown_sections(cleaned)
    chunks: List[MarkdownChunk] = []
    for section in sections:
        chunks.extend(
            _chunk_section(section, chunk_size=chunk_size, overlap=overlap)
        )
    return chunks


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    将 Markdown 文档切成适合检索的语义块，并仅返回文本。

    保留该函数是为了兼容旧调用；入库时使用 ``_chunk_markdown`` 获取标题元数据。
    """
    return [
        chunk.text
        for chunk in _chunk_markdown(text, chunk_size=chunk_size, overlap=overlap)
    ]


def _rerank_food_chunks(query: str, chunks: List[str]) -> List[str]:
    """Light lexical rerank so food queries prefer actionable food/table chunks."""
    if not chunks:
        return chunks
    food_terms = (
        "鸡胸", "蛋清", "鱼", "虾", "牛肉", "瘦肉", "乳清", "豆腐",
        "蛋白", "低脂", "每100g", "100g", "食材", "表 1", "核心高优",
        "高蛋白", "净碳水", "脂肪",
    )
    query_hits = [t for t in food_terms if t.lower() in query.lower() or t in query]

    def score(doc: str) -> tuple[int, int]:
        s = 0
        for term in query_hits:
            if term.lower() in doc.lower() or term in doc:
                s += 6
        for marker in ("|", "每100g", "100 g", "100g", "蛋白质", "脂肪", "净碳水"):
            if marker in doc:
                s += 2
        for action_marker in ("表 1", "核心高优", "食材", "鸡胸", "蛋清"):
            if action_marker in doc:
                s += 5
        for abstract_marker in ("算法", "Principle", "Agent 不得", "技术规范"):
            if abstract_marker in doc:
                s -= 4
        return s, -len(doc)

    return sorted(chunks, key=score, reverse=True)


def _read_utf8_file(path: Path) -> str:
    """以 UTF-8 读取整个文件。"""
    return path.read_text(encoding="utf-8")


def _add_documents_in_batches(
    collection: Any,
    *,
    ids: List[str],
    documents: List[str],
    metadatas: List[dict],
    collection_label: str,
    batch_size: int = DEFAULT_INGEST_BATCH_SIZE,
) -> None:
    """分批写入 Chroma，避免一次性嵌入大批中文长文本导致内存峰值过高。"""
    if batch_size <= 0:
        batch_size = DEFAULT_INGEST_BATCH_SIZE
    total: int = len(ids)
    for start in range(0, total, batch_size):
        end: int = min(start + batch_size, total)
        print(
            f"[RAG 入库] 集合 {collection_label!r} | 正在写入第 {start + 1}-{end}/{total} 条…",
            file=sys.stdout,
            flush=True,
        )
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


class PSMFRAGKnowledgeBase:
    """
    PSMF 本地向量知识库封装：入库（ingest）与检索（search）。

    使用 ChromaDB 默认嵌入函数，无需在代码中显式指定外部 Embedding 模型类名，
    由 ``PersistentClient`` 创建的 Collection 在 ``add`` / ``query`` 时自动编码文本。
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_CHROMA_DIR,
        base_dir: Optional[Path] = None,
    ) -> None:
        """
        :param persist_directory: ChromaDB 持久化根目录（将创建 ``./chroma_db`` 等）
        :param base_dir: Markdown 源文件所在目录；默认为本文件所在目录（便于从任意 cwd 运行）
        """
        self._persist_directory: str = persist_directory
        self._base_dir: Path = (
            base_dir if base_dir is not None else Path(__file__).resolve().parent
        )

        # 延迟创建客户端，避免仅 import 本模块就写磁盘
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Any] = None

    def _get_client(self) -> chromadb.PersistentClient:
        """懒加载持久化客户端；失败时记录堆栈并向上抛出。"""
        if self._client is None:
            try:
                # 本地落盘路径；ChromaDB 会在该目录下保存 sqlite 与向量数据
                self._client = chromadb.PersistentClient(path=self._persist_directory)
            except Exception:
                logger.exception("创建 ChromaDB PersistentClient 失败")
                raise
        return self._client

    def _get_collection(self, *, recreate: bool = False) -> Any:
        """
        获取或创建集合。若 ``recreate`` 为 True，则先删除同名集合再重建（用于全量重灌）。
        """
        client: chromadb.PersistentClient = self._get_client()
        if recreate:
            try:
                client.delete_collection(name=COLLECTION_NAME)
            except Exception as exc:
                # 集合不存在时删除可能报错，忽略以保证幂等
                logger.debug("删除集合（可能不存在）时出现提示：%s", exc)

        try:
            collection: Any = client.get_or_create_collection(
                name=COLLECTION_NAME,
                # 不传入 embedding_function 时使用 Chroma 默认嵌入
            )
        except Exception:
            logger.exception("创建或获取 ChromaDB 集合失败")
            raise

        self._collection = collection
        return collection

    def ingest_documents(
        self,
        source_filenames: Optional[tuple[str, ...]] = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        recreate_collection: bool = True,
    ) -> int:
        """
        读取指定 Markdown 文件，切块后写入向量库。

        :param source_filenames: 文件名元组；默认使用 ``DEFAULT_SOURCE_FILES``
        :param chunk_size: 切块大小（字符）
        :param chunk_overlap: 切块重叠（字符）
        :param recreate_collection: 为 True 时先删除旧集合，避免重复入库产生重复条目
        :return: 成功写入的文本块数量
        :raises FileNotFoundError: 任一源文件不存在
        :raises ValueError: 读取后无有效文本块
        """
        files: tuple[str, ...] = source_filenames or DEFAULT_SOURCE_FILES
        collection: Any = self._get_collection(recreate=recreate_collection)

        all_ids: List[str] = []
        all_documents: List[str] = []
        all_metadatas: List[dict] = []

        for filename in files:
            file_path: Path = self._base_dir / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"找不到知识库文件：{file_path}")

            try:
                raw_text: str = _read_utf8_file(file_path)
            except OSError as exc:
                logger.error("读取文件失败：%s", file_path)
                raise OSError(f"无法读取文件：{file_path}") from exc

            chunks: List[MarkdownChunk] = _chunk_markdown(
                raw_text, chunk_size=chunk_size, overlap=chunk_overlap
            )
            if not chunks:
                logger.warning("文件切块后为空，已跳过：%s", file_path)
                continue

            n_file: int = len(chunks)
            print(
                f"[RAG 入库] 集合 {COLLECTION_NAME!r} | 正在读取并切块：{filename!r}（共 {n_file} 块）",
                file=sys.stdout,
                flush=True,
            )
            for idx, chunk in enumerate(chunks):
                if _should_emit_chunk_progress(idx, n_file):
                    _print_ingest_progress(
                        filename,
                        idx + 1,
                        n_file,
                        collection_label=COLLECTION_NAME,
                    )
                chunk_id: str = f"{filename}#{idx}"
                all_ids.append(chunk_id)
                all_documents.append(chunk.text)
                all_metadatas.append(
                    {
                        "source": filename,
                        "chunk_index": idx,
                        "chunk_chars": len(chunk.text),
                        "section_path": chunk.section_path,
                        "section_title": chunk.section_title,
                        "heading_level": chunk.heading_level,
                        "chunking": RAG_CHUNKING_VERSION,
                    }
                )

        if not all_documents:
            raise ValueError(
                "没有可入库的文本块：请检查源文件是否为空或切块参数是否过严。"
            )

        try:
            print(
                f"[RAG 入库] 集合 {COLLECTION_NAME!r} | 正在嵌入并写入向量索引（共 {len(all_ids)} 条）…",
                file=sys.stdout,
                flush=True,
            )
            _add_documents_in_batches(
                collection,
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas,
                collection_label=COLLECTION_NAME,
            )
        except Exception:
            logger.exception("写入 ChromaDB 失败，ids 数量=%d", len(all_ids))
            raise

        logger.info(
            "入库完成：共 %d 块，集合=%s，来源文件=%s，持久化目录=%s",
            len(all_documents),
            COLLECTION_NAME,
            ", ".join(files),
            self._persist_directory,
        )
        print(
            f"[RAG 入库] 集合 {COLLECTION_NAME!r} | 写入完成（{len(all_documents)} 块）。",
            file=sys.stdout,
            flush=True,
        )
        return len(all_documents)

    def ingest_named_collection(
        self,
        collection_name: str,
        source_filenames: tuple[str, ...],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        recreate_collection: bool = True,
    ) -> int:
        """
        将指定 Markdown 文件切块写入任意命名的集合（如 ``food_db``）。
        """
        client: chromadb.PersistentClient = self._get_client()
        if recreate_collection:
            try:
                client.delete_collection(name=collection_name)
            except Exception as exc:
                logger.debug("删除集合 %s（可能不存在）：%s", collection_name, exc)

        collection: Any = client.get_or_create_collection(name=collection_name)

        all_ids: List[str] = []
        all_documents: List[str] = []
        all_metadatas: List[dict] = []

        for filename in source_filenames:
            file_path: Path = self._base_dir / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"找不到知识库文件：{file_path}")

            try:
                raw_text: str = _read_utf8_file(file_path)
            except OSError as exc:
                logger.error("读取文件失败：%s", file_path)
                raise OSError(f"无法读取文件：{file_path}") from exc

            chunks: List[MarkdownChunk] = _chunk_markdown(
                raw_text, chunk_size=chunk_size, overlap=chunk_overlap
            )
            if not chunks:
                logger.warning("文件切块后为空，已跳过：%s", file_path)
                continue

            n_file: int = len(chunks)
            print(
                f"[RAG 入库] 集合 {collection_name!r} | 正在读取并切块：{filename!r}（共 {n_file} 块）",
                file=sys.stdout,
                flush=True,
            )
            for idx, chunk in enumerate(chunks):
                if _should_emit_chunk_progress(idx, n_file):
                    _print_ingest_progress(
                        filename,
                        idx + 1,
                        n_file,
                        collection_label=collection_name,
                    )
                chunk_id: str = f"{collection_name}:{filename}#{idx}"
                all_ids.append(chunk_id)
                all_documents.append(chunk.text)
                all_metadatas.append(
                    {
                        "source": filename,
                        "collection": collection_name,
                        "chunk_index": idx,
                        "chunk_chars": len(chunk.text),
                        "section_path": chunk.section_path,
                        "section_title": chunk.section_title,
                        "heading_level": chunk.heading_level,
                        "chunking": RAG_CHUNKING_VERSION,
                    }
                )

        if not all_documents:
            raise ValueError(
                f"集合 {collection_name!r} 没有可入库的文本块：请检查源文件。"
            )

        try:
            print(
                f"[RAG 入库] 集合 {collection_name!r} | 正在嵌入并写入向量索引（共 {len(all_ids)} 条）…",
                file=sys.stdout,
                flush=True,
            )
            _add_documents_in_batches(
                collection,
                ids=all_ids,
                documents=all_documents,
                metadatas=all_metadatas,
                collection_label=collection_name,
            )
        except Exception:
            logger.exception(
                "写入 ChromaDB 失败 collection=%s ids=%d",
                collection_name,
                len(all_ids),
            )
            raise

        logger.info(
            "入库完成：集合=%s，共 %d 块，来源=%s",
            collection_name,
            len(all_documents),
            ", ".join(source_filenames),
        )
        print(
            f"[RAG 入库] 集合 {collection_name!r} | 写入完成（{len(all_documents)} 块）。",
            file=sys.stdout,
            flush=True,
        )
        return len(all_documents)

    def search_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """
        根据用户查询返回最相关的若干文本块（纯文本）。

        :param query: 用户问题或检索语句
        :param top_k: 返回条数，默认 3
        :return: 按相关度排序的文本块列表（长度可能小于 top_k，例如库为空时）
        """
        trimmed: str = query.strip()
        if not trimmed:
            logger.warning("search_knowledge 收到空查询，返回空列表")
            return []

        client: chromadb.PersistentClient = self._get_client()

        # 若从未 ingest，集合可能不存在：get 会失败
        try:
            collection: Any = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            logger.error(
                "无法获取集合 %r：可能尚未执行 ingest_documents。底层错误：%s",
                COLLECTION_NAME,
                exc,
            )
            return []

        try:
            results = collection.query(
                query_texts=[trimmed],
                n_results=top_k,
            )
        except Exception:
            logger.exception("ChromaDB query 失败，query=%r", trimmed)
            return []

        # results["documents"] 为 List[List[str]]，外层对应每个 query_text
        documents_nested: Optional[List[List[str]]] = results.get("documents")
        if not documents_nested or not documents_nested[0]:
            return []

        chunks: List[str] = [d for d in documents_nested[0] if isinstance(d, str) and d]
        return chunks[:top_k]

    def search_knowledge_by_source(
        self,
        query: str,
        source_filename: str,
        top_k: int = 3,
    ) -> List[str]:
        """
        在主集合 ``psmf_knowledge`` 内按 ``metadata.source`` 过滤后向量检索。
        用于分别从 ``symptom_diagnostic_matrix.md`` / ``psmf_training_guide.md`` 取块。
        """
        trimmed: str = query.strip()
        if not trimmed:
            return []

        client: chromadb.PersistentClient = self._get_client()
        try:
            collection: Any = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            logger.error("无法获取集合 %r：%s", COLLECTION_NAME, exc)
            return []

        try:
            results = collection.query(
                query_texts=[trimmed],
                n_results=top_k,
                where={"source": source_filename},
            )
        except Exception:
            logger.exception(
                "ChromaDB query 失败 source=%s query=%r",
                source_filename,
                trimmed[:120],
            )
            return []

        documents_nested: Optional[List[List[str]]] = results.get("documents")
        if not documents_nested or not documents_nested[0]:
            return []

        chunks: List[str] = [d for d in documents_nested[0] if isinstance(d, str) and d]
        return chunks[:top_k]

    def search_in_collection(
        self,
        collection_name: str,
        query: str,
        top_k: int = 3,
    ) -> List[str]:
        """在指定集合中检索文本块。"""
        trimmed: str = query.strip()
        if not trimmed:
            return []

        client: chromadb.PersistentClient = self._get_client()
        try:
            collection: Any = client.get_collection(name=collection_name)
        except Exception as exc:
            logger.error("无法获取集合 %r：%s", collection_name, exc)
            return []

        try:
            results = collection.query(
                query_texts=[trimmed],
                n_results=top_k,
            )
        except Exception:
            logger.exception("ChromaDB query 失败 collection=%s", collection_name)
            return []

        documents_nested: Optional[List[List[str]]] = results.get("documents")
        if not documents_nested or not documents_nested[0]:
            return []

        chunks: List[str] = [d for d in documents_nested[0] if isinstance(d, str) and d]
        if collection_name == COLLECTION_FOOD:
            chunks = _rerank_food_chunks(trimmed, chunks)
        return chunks[:top_k]


def ingest_documents(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    base_dir: Optional[Path] = None,
) -> int:
    """
    便捷函数：使用默认参数执行一次全量入库，返回写入块数。

    便于脚本 ``python -c "from rag_system import ingest_documents; ingest_documents()"`` 调用。
    """
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase(
        persist_directory=persist_directory,
        base_dir=base_dir,
    )
    return kb.ingest_documents()


def search_knowledge(query: str, top_k: int = 3) -> List[str]:
    """
    便捷函数：在默认持久化路径上检索（需已 ingest）。
    """
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase()
    return kb.search_knowledge(query, top_k=top_k)


def search_knowledge_by_source(
    query: str,
    source_filename: str,
    top_k: int = 3,
) -> List[str]:
    """在主知识库中仅检索指定 Markdown 来源文件的切块。"""
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase()
    return kb.search_knowledge_by_source(query, source_filename, top_k=top_k)


def search_food_knowledge(query: str, top_k: int = 3) -> List[str]:
    """在 ``food_db`` 集合中检索食谱 / 食材相关知识。"""
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase()
    return kb.search_in_collection(COLLECTION_FOOD, query, top_k=top_k)


def ingest_food_database(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    base_dir: Optional[Path] = None,
) -> int:
    """将 ``psmf_food_database.md`` 全量灌入 ``food_db`` 集合。"""
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase(
        persist_directory=persist_directory,
        base_dir=base_dir,
    )
    return kb.ingest_named_collection(
        COLLECTION_FOOD,
        FOOD_SOURCE_FILES,
        recreate_collection=True,
    )


def _ensure_collection_nonempty(
    kb: PSMFRAGKnowledgeBase,
    collection_name: str,
    source_files: tuple[str, ...],
    *,
    use_default_ingest: bool = False,
) -> tuple[bool, int]:
    """
    若集合不存在、count 为 0，或缺少预期 source，则执行 ingest。

    :param use_default_ingest: True 时调用 ``ingest_documents``（仅用于主知识库）。
    """
    client: chromadb.PersistentClient = kb._get_client()
    n_existing: int = 0
    missing_sources: set[str] = set(source_files)
    chunking_ok: bool = False
    try:
        col: Any = client.get_collection(name=collection_name)
        n_existing = int(col.count())
        if n_existing > 0:
            got = col.get(include=["metadatas"])
            metadatas = got.get("metadatas") or []
            present_sources = {
                str(m.get("source"))
                for m in metadatas
                if isinstance(m, dict) and m.get("source")
            }
            chunking_versions = {
                str(m.get("chunking") or "legacy")
                for m in metadatas
                if isinstance(m, dict)
            }
            missing_sources = set(source_files) - present_sources
            chunking_ok = chunking_versions == {RAG_CHUNKING_VERSION}
    except Exception:
        n_existing = 0

    if n_existing > 0 and not missing_sources and chunking_ok:
        return False, n_existing
    if n_existing > 0 and not chunking_ok:
        logger.info(
            "RAG 集合 %s 使用旧切片版本，将重建索引：expected=%s",
            collection_name,
            RAG_CHUNKING_VERSION,
        )

    if use_default_ingest:
        ingested: int = kb.ingest_documents(recreate_collection=True)
    else:
        ingested = kb.ingest_named_collection(
            collection_name,
            source_files,
            recreate_collection=True,
        )
    return True, ingested


def ensure_local_index_ready(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    base_dir: Optional[Path] = None,
) -> tuple[bool, int]:
    """
    确保 ``psmf_knowledge`` 集合已就绪。

    :return: ``(是否刚执行了 ingest, 当前估计的文本块数量)``
    """
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase(
        persist_directory=persist_directory,
        base_dir=base_dir,
    )
    return _ensure_collection_nonempty(
        kb,
        COLLECTION_NAME,
        DEFAULT_SOURCE_FILES,
        use_default_ingest=True,
    )


def ensure_food_index_ready(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    base_dir: Optional[Path] = None,
) -> tuple[bool, int]:
    """确保 ``food_db`` 食谱集合已就绪。"""
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase(
        persist_directory=persist_directory,
        base_dir=base_dir,
    )
    return _ensure_collection_nonempty(
        kb,
        COLLECTION_FOOD,
        FOOD_SOURCE_FILES,
        use_default_ingest=False,
    )


def ensure_all_local_indexes_ready(
    persist_directory: str = DEFAULT_CHROMA_DIR,
    base_dir: Optional[Path] = None,
) -> dict[str, tuple[bool, int]]:
    """
    依次检查并必要时 ingest：主知识库 + 食谱库。

    :return: 例如 ``{\"psmf_knowledge\": (did_ingest, n_chunks), \"food_db\": (...)}``
    """
    print(
        "[RAG] 正在检查本地向量索引（Chroma）…",
        file=sys.stdout,
        flush=True,
    )
    kb: PSMFRAGKnowledgeBase = PSMFRAGKnowledgeBase(
        persist_directory=persist_directory,
        base_dir=base_dir,
    )
    p: tuple[bool, int] = _ensure_collection_nonempty(
        kb,
        COLLECTION_NAME,
        DEFAULT_SOURCE_FILES,
        use_default_ingest=True,
    )
    f: tuple[bool, int] = _ensure_collection_nonempty(
        kb,
        COLLECTION_FOOD,
        FOOD_SOURCE_FILES,
        use_default_ingest=False,
    )
    print(
        "[RAG] 向量索引检查完成："
        f" {COLLECTION_NAME}={'本次已执行入库' if p[0] else '已存在，未重灌'}（约 {p[1]} 条）, "
        f"{COLLECTION_FOOD}={'本次已执行入库' if f[0] else '已存在，未重灌'}（约 {f[1]} 条）。",
        file=sys.stdout,
        flush=True,
    )
    return {
        "psmf_knowledge": p,
        "food_db": f,
    }


if __name__ == "__main__":
    # 简单自测：入库后检索
    logging.basicConfig(level=logging.INFO)
    try:
        n: int = ingest_documents()
        print(f"入库块数：{n}")
        demo_q: str = "头晕心悸 电解质 缺钠"
        hits: List[str] = search_knowledge(demo_q, top_k=3)
        print(f"查询：{demo_q!r}")
        for i, h in enumerate(hits, 1):
            print(f"--- 命中 {i} ---\n{h[:500]}...\n")
    except Exception:
        print("自测失败，详见堆栈：")
        traceback.print_exc()
