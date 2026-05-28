# 项目亮点与能力映射

## 核心亮点

- 面向健康管理 / 减脂陪伴场景构建 AI Agent PoC，围绕 PSMF 减脂协议实现用户建档、饮食 / 训练打卡、阶段复盘和安全提醒等完整业务流程。
- 不是通用聊天机器人，也不是线性 LLM workflow，而是将 Planner、Tool Registry、RAG、长期用户状态和服务流程组合成 Tool-Using Health Management Agent。
- 同时提供 Streamlit Web Demo、Telegram Bot 和 CLI 三种入口，覆盖方案演示、私域触达和技术验证。
- 在健康类场景中显式加入 deterministic safety guardrail，对胸痛、呼吸困难、晕厥、意识模糊等高风险症状 hard stop，避免把风险反馈当作普通减脂建议处理。

## 技术能力

- 基于 Gemini 构建核心对话与内容生成能力。
- `agent/orchestrator.py` 实现 bounded ReAct-style tool loop：safety pre-check、profile load、planner decision、tool execution、observation、optional re-plan、final response、memory persistence。
- `agent/tool_registry.py` 提供统一 `ToolSpec` / `ToolCall` / `ToolResult` 抽象，PSMF 计算、RAG、记忆读写、饮食打卡、补剂打卡和总结都是真实可调用工具。
- `agent/schemas.py` 中的 `AgentStep` trace 可用于调试和演示，但只记录公开 step summary，不暴露完整 chain-of-thought。
- 使用 ChromaDB 搭建本地 RAG 知识库，将 PSMF 协议、食物数据库、训练指南和症状矩阵用于垂直知识问答。
- RAG 采用 Markdown 标题语义切分：按 `# / ## / ###` 章节建立 chunk，保留标题路径、来源文件和 section metadata，并对食材表、症状矩阵等表格结构做保护。
- 通过 `memory_manager.py` 维护用户长期记忆，包括体重、体脂、Category、蛋白质目标、补剂状态、历史打卡和对话上下文。
- `psmf_engine.py` 已瘦身为 legacy compatibility wrapper，保留旧 `GeminiPSMFAgent` 入口并委托给 `AgentOrchestrator`。
- 使用 `.env.example`、`.gitignore` 和本地缓存隔离，保持密钥、用户数据和向量库缓存不进入公开仓库。

## 解决方案能力

- 将客户高频咨询、长期陪伴、个性化建议和风险控制拆解为可演示的业务流程。
- 使用“客户痛点 - Agent 能力 - 业务价值”的方式解释方案价值，而不是只展示技术实现。
- 将专家资料沉淀为可复用知识库，降低对单个顾问经验的依赖。
- 通过多渠道入口展示同一套 Agent 核心如何适配 Web 演示、私域运营和本地验证。
- 将健康类 AI 应用的安全边界、敏感数据隔离和人工介入思路纳入方案设计。

## 可展示能力

- 客户业务痛点分析：识别成本、响应速度、留存、标准化和风险边界等关键问题。
- 行业场景拆解：将减脂陪伴拆解为建档、打卡、问答、复盘、提醒和风险识别。
- AI 方案架构设计：将 LLM Planner、工具调用、RAG、长期记忆、多入口交互和 deterministic safety guardrail 组合为完整 PoC。
- Demo Storyline 设计：围绕典型用户旅程组织演示路径。
- 技术价值转业务价值：将 RAG、记忆和 Agent 编排映射为降本增效、服务连续性和规模化交付。

## 后续扩展方向

- 接入企业 CRM、会员系统、客服工单系统或顾问工作台。
- 增加数据看板，用于追踪用户打卡率、留存率、风险提醒和服务质量。
- 替换为客户自有知识库，扩展到营养咨询、运动康复、企业健康管理或智能客服场景。
- 增加人工审核和转接机制，让 AI 建议与专业顾问协同。
- 增加演示截图、录屏或在线 Demo，提升公开仓库的可理解性和可信度。
