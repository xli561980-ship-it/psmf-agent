# **基于临床运动营养学工程化的 PSMF 食材数据库与 AI Agent 架构规范报告**

在现代临床营养学与代谢工程的交汇点上，蛋白质保留改良禁食（Protein-Sparing Modified Fast, PSMF）作为一种极端但高效的干预手段，正逐渐从医疗机构的严格监管走向基于人工智能辅助的个性化管理。PSMF 的核心逻辑在于利用极低热量摄入（Very Low-Calorie Diet, VLCD）诱导深度酮症，同时通过高生物价值蛋白质的摄入来抵消内源性蛋白质的降解，从而实现“减脂而不减肌”的代谢目标 1。本报告旨在为开发 PSMF AI Agent 提供一套标准化的食材数据库（psmf\_food\_database.md）及其配套的组装算法逻辑，确保干预方案在临床安全红线内运行。

## **第一部分：PSMF 的生理机制与数据工程化背景**

PSMF 的生理学基础源于对饥饿反应的生化改良。在完全断食状态下，人体会通过糖异生作用分解肌肉组织以获取氨基酸，进而转化为葡萄糖供给大脑。PSMF 通过摄入外源性蛋白质（通常为 ![][image1] 至 ![][image2] 克/公斤理想体重），有效地抑制了这一过程 2。数据工程在其中的角色是确保摄入的每一克蛋白质都伴随着最少的附加脂肪和碳水化合物，即最大化“蛋白质/热量比” 6。

由于 PSMF 期间日均热量通常低于 ![][image3] 至 ![][image4] 千卡，任何数据误差都可能导致代谢状态从“蛋白质保留”滑向“代谢损伤” 1。因此，数据库必须基于“生重（Raw Weight）”进行标准化。生重数据规避了烹饪过程中水分流失（如煎炸导致的脱水）或添加剂引入（如油脂、酱汁）带来的数值漂移，为 AI Agent 提供了最纯净的计算底座 6。

## **第二部分：psmf\_food\_database.md 结构化数据**

以下数据均参考 USDA FoodData Central (SR Legacy 与 Foundation Foods 数据集)，数值精确到小数点后一位，作为 AI Agent 的核心检索源 6。

### **表 1：核心高优蛋白质来源（目标：高蛋白、极低脂、零碳水）**

本表收录的食材是 PSMF 的能量支柱。其筛选标准为蛋白质供能比需超过 ![][image5]，且单项食材的脂肪含量应尽可能接近零 6。

| 食材名称 | 蛋白质(g) | 脂肪(g) | 净碳水(g) | 总热量(kcal) | 备注说明 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 鸡胸肉 (Chicken Breast, Skinless) | 22.5 | 2.6 | 0.0 | 120.0 | 经典的极低脂选择；烹饪建议：低温慢煮或喷雾油香煎以防干柴 10。 |
| 火鸡胸 (Turkey Breast, Skinless) | 24.0 | 1.0 | 0.1 | 114.0 | 比鸡胸肉更优的蛋白质/热量比；适合烤制或切片冷餐 13。 |
| 蛋清 (Egg White, Raw) | 10.9 | 0.2 | 0.7 | 52.0 | 生物价值最高的蛋白源；可作为增加饱腹感体积的基底 6。 |
| 瘦牛肉（眼肉盖，Beef Eye of Round） | 22.0 | 3.0 | 0.0 | 119.0 | 必须去除所有可见白色脂肪；富含铁、锌及维生素 B12 16。 |
| 罗非鱼 (Tilapia, Raw) | 20.1 | 1.7 | 0.0 | 96.0 | 性价比极高的白肉鱼类；肉质细嫩，适合快速快炒 19。 |
| 鳕鱼 (Cod, Atlantic, Raw) | 18.0 | 0.7 | 0.0 | 82.0 | 脂肪含量极低，极适合深度减脂期；建议清蒸保持营养 22。 |
| 虾 (Shrimp, Raw) | 20.1 | 0.5 | 0.0 | 85.0 | 极高蛋白且富含微量元素；需注意胆固醇但脂肪极低 25。 |
| 扇贝 (Scallops, Raw) | 12.1 | 0.5 | 3.2 | 69.0 | 含有少量天然糖原；口感丰富，可缓解饮食单调感 28。 |
| 水浸金枪鱼罐头 (Canned Tuna in Water) | 25.5 | 0.8 | 0.0 | 116.0 | 极致的便携蛋白源； Agent 必须排斥油浸版本 6。 |
| 脱脂希腊酸奶 (Nonfat Greek Yogurt) | 10.0 | 0.4 | 4.0 | 59.0 | 兼顾乳清蛋白与酪蛋白；需严格警惕隐藏添加糖 31。 |
| 脱脂干酪 (Cottage Cheese, Nonfat) | 11.6 | 0.3 | 4.6 | 72.0 | 含有大量缓释酪蛋白；适合作为晚间餐点预防夜间分解 34。 |

### **表 2：PSMF 许可绿叶与十字花科蔬菜（目标：高纤维、低碳水）**

蔬菜在 PSMF 中不作为能量来源，而是作为微量元素补充剂与物理饱腹感填充物。 Agent 应利用其纤维素属性来维持肠道健康 2。

| 食材名称 | 净碳水(g) | 膳食纤维(g) | 蛋白质(g) | 总热量(kcal) |
| :---- | :---- | :---- | :---- | :---- |
| 菠菜 (Spinach, Raw) | 1.4 | 2.2 | 2.9 | 23.0 |
| 西蓝花 (Broccoli, Raw) | 4.0 | 2.6 | 2.8 | 45.0 |
| 芦笋 (Asparagus, Raw) | 1.8 | 2.1 | 2.2 | 20.0 |
| 芹菜 (Celery, Raw) | 1.4 | 1.6 | 0.7 | 15.0 |
| 黄瓜 (Cucumber, Raw) | 1.5 | 0.5 | 0.6 | 15.0 |
| 生菜 (Lettuce, Raw) | 1.1 | 1.3 | 1.4 | 15.0 |
| 羽衣甘蓝 (Kale, Raw) | 5.2 | 3.6 | 4.3 | 49.0 |
| 白萝卜 (Daikon Radish, Raw) | 2.5 | 1.6 | 0.6 | 18.0 |
| 青椒 (Green Pepper, Raw) | 2.9 | 1.7 | 0.9 | 20.0 |

### **表 3：PSMF “极低卡”调味与补剂底座**

调味品的工程化处理常被忽视，但在 PSMF 的极低热量框架下，不规范的调味品累加可能导致全天碳水超标 7。

| 名称 | 每份(约10g/ml)净碳水(g) | 热量(kcal) |
| :---- | :---- | :---- |
| 苹果醋 (Apple Cider Vinegar) | 0.1 | 2.0 |
| 无糖芥末酱 (Sugar-free Mustard) | 0.5 | 5.0 |
| 新鲜柠檬汁 (Fresh Lemon Juice) | 0.9 | 3.0 |
| 干香料 (黑胡椒、大蒜粉等) | 1.5 | 6.0 |
| 赤藓糖醇 (Erythritol) | 0.0 | 0.0 |

## **第三部分：蛋白质源的临床深度分析与数据解读**

在 PSMF 的数据组装过程中， AI Agent 必须理解不同蛋白质源的生化特性，而非仅仅进行数值叠加。蛋白质的氨基酸构成（PDCAAS 或 DIAAS 评分）决定了其在极低热量环境下维持氮平衡的效率 3。

### **禽类蛋白质的稳定性分析**

鸡胸肉与火鸡胸被选为核心蛋白源，是因为其极低的肌间脂肪含量 6。研究表明，火鸡胸在每 100 千卡热量中提供的蛋白质高达 21.6 克，而鸡胸肉约为 18.8 克 6。这种细微的效率差异在长期干预中会产生累积效应。从数据工程角度看，鸡胸肉的生熟比约为 ![][image6]，这意味着 AI Agent 在将食谱呈现给用户时，必须明确告知 100 克生重对应的熟重体积，以防止用户因称重习惯错误而导致摄入不足 8。

### **水产类蛋白质的微量营养优势**

鳕鱼、罗非鱼和虾类在提供蛋白质的同时，绕过了红肉中常见的饱和脂肪风险 22。鳕鱼的蛋白质/热量效率在水产中位居前列，每 100 克生重仅含 0.7 克脂肪 22。然而，扇贝等贝类虽然含有约 3.2 克的净碳水，但这属于天然糖原，在 Agent 组装逻辑中，这部分碳水必须从全天 50 克的总配额中扣除 28。此外，水产类通常富含碘和硒，这对维持 PSMF 期间可能下降的甲状腺转化率（T4 转 T3）具有潜在的保护作用 21。

### **乳制品在保留阶段的特殊角色**

脱脂希腊酸奶和脱脂干酪被引入数据库，是为了满足用户对饮食质地的需求 31。干酪中的酪蛋白（Casein）具有缓慢吸收的特性，能在胃部形成凝胶，延长氨基酸释放时间，这在缓解 PSMF 带来的强烈饥饿感方面具有显著价值 35。 Agent 在调用这些食材时，应设定“每日一份”的频次上限，以防止乳糖累加导致碳水红线受压 31。

## **第四部分：蔬菜摄入的体积工程与纤维素逻辑**

蔬菜在 PSMF 协议中并非“可选”，而是“强制” 1。由于蛋白质摄入量高且总食物体积显著下降，肠道蠕动放缓（便秘）是常见的副作用。

### **纤维素与肠道稳态**

表 2 中的蔬菜选择主要基于其“净碳水/纤维比”。菠菜和羽衣甘蓝不仅提供了必要的膳食纤维，还含有高浓度的维生素 K、叶酸和镁 42。镁的补充在 PSMF 期间尤为重要，因为电解质流失常导致肌肉抽搐 2。 Agent 在计算方案时，应优先分配蔬菜额度给菠菜和生菜，因为它们的净碳水极低（低于 1.5 克/100克），允许用户摄入更大的物理体积 42。

### **十字花科蔬菜的权衡**

西蓝花和青椒虽然营养丰富，但其净碳水含量较高（4.0 克与 2.9 克） 36。 Agent 必须精准计算其碳水占比，避免在蛋白质源已经包含部分碳水（如扇贝或脱脂酸奶）的情况下，导致每日 50 克红线被击穿。蔬菜的体积填充效应遵循胃牵张感受器的逻辑：摄入高体积、低能量的食物能诱导饱腹感信号，提高干预的依从性 1。

## **第五部分：电解质与微量元素的补救逻辑**

由于胰岛素水平在 PSMF 期间显著下降，肾脏会排出大量的水分、钠和钾 1。这一过程被称为“禁食性钠排泄”。

### **钠、钾、镁的平衡**

AI Agent 在生成食谱时，必须在文案中集成“电解质强制补充”指令。临床指南建议每日摄入至少 5 克盐（钠）、1 克钾和 400 毫克镁 2。缺乏这些补剂会导致典型的“酮流感”症状，包括头痛、眩晕和疲劳 1。

### **补充剂底座的工程化**

表 3 列出的调味品实质上充当了“微量元素载体”。例如，柠檬汁提供了微量的维生素 C，而干香料（如大蒜粉）在提供风味的同时，含有极少的碳水化合物 39。 Agent 必须通过算法监控这些微量添加，确保全天热量漂移控制在 50 千卡以内。

## **第六部分：AI Agent 约束与算法指南（技术规范）**

为实现 Agent 的自动化组装功能，以下逻辑必须硬编码至其执行引擎中。

### **1\. 蛋白质需求锚定算法（The Protein-First Principle）**

Agent 不得以“平衡膳食”为起点，而必须以“氮平衡需求”为基石。

* **输入变量**：用户当前体重（![][image7]）、体脂率（![][image8]）、目标体重（![][image9]）。  
* **计算公式**：  
  ![][image10]  
  针对极高运动量用户，系数可上调至 ![][image11] 2。  
* **组装逻辑**：Agent 从表 1 中检索食材，进行多项式组合。例如：若 ![][image12]，Agent 可组合“300g 鸡胸肉 \+ 200g 蛋清 \+ 100g 鳕鱼”。  
* **误差容限**：单日总蛋白质输出值必须在计算值的 ![][image13] 范围内。超出此范围的方案必须重算。

### **2\. 红线阻断机制（Red Line Interlock）**

这是 Agent 的核心安全模块，任何违反以下条件的方案均标记为“非法方案”。

* **脂肪阻断**：全天所有来源（包含蛋白质附带脂肪与补剂）的脂肪总和 ![][image14] 必须拒绝。  
  * *逻辑扩展*：若用户选择了瘦牛肉，Agent 必须自动削减后续餐点中酸奶或水产的比例，以维持脂肪平衡 1。  
* **碳水阻断**：全天净碳水总和 ![][image15] 必须拒绝。  
  * *深度约束*：在严苛模式下，此红线应设为 ![][image16] 以确保快速诱导酮症 5。  
* **热量天花板**：总热量 ![][image17] 必须触发警告（临床推荐通常在 ![][image18] 以下） 4。

### **3\. 宏量平衡算法（Volume & Satiety Optimization）**

该算法用于决定餐盘的物理排布，而非仅仅是宏量数值。

* **填充逻辑**：在满足蛋白质需求后，Agent 应尽可能多地分配表 2 中的蔬菜进入每一餐。  
* **分配比例**：建议每餐蛋白质/蔬菜的体积比为 ![][image19] 至 ![][image20]。  
* **隐形碳水核算**：明确指出，蔬菜的碳水并非“免费”。 Agent 必须在全天碳水账单中逐克计入蔬菜的净碳水。若蔬菜加入导致全天碳水超过 ![][image21]，Agent 必须优先削减西蓝花或青椒，代之以生菜或黄瓜 36。

### **4\. 补剂与饮水逻辑（Hydration & Micros Engine）**

* **强制提醒**：Agent 生成的每一份食谱必须附带以下文案：  
  * “饮水：今日需饮用至少 ![][image22] 无糖液体 1。”  
  * “盐分：今日需额外摄入至少 ![][image23] 食盐以防止电解质失衡 2。”  
  * “补剂：强制服用一份多元维生素 2。”

## **第七部分：PSMF 的阶段性演变与 Agent 适应性**

PSMF 并非长期饮食模式，其周期性特征要求 AI Agent 具备“状态感知”能力 1。

### **强化期（Intensive Phase）的数据处理**

在此阶段（通常持续几周到六个月）， Agent 需严格执行上述 ![][image24] 红线 2。数据精度至关重要。对于严重肥胖者（BMI ![][image25]）， Agent 应根据其基础代谢率（BMR）微调蛋白质系数，防止肌肉过量流失 4。

### **恢复期（Refeeding Phase）的逻辑切换**

一旦达到目标体重， Agent 必须自动切换至“再摄食算法” 2。

* **碳水递增**：第一个月碳水上调至 ![][image26]，第二个月上调至 ![][image27] 1。  
* **蛋白质递减**：每月减少约 ![][image28] 的蛋白质摄入，同时引入健康的复合碳水化合物和优质脂肪 1。  
* **数据源扩展**：Agent 此时可访问更广泛的食材库，但仍需监控体重反弹趋势。

## **第八部分：临床安全性与禁忌证的逻辑预检**

作为专业的临床运动营养学数据工程师， Agent 在开始组装食谱前，必须通过对话或数据接口确认用户不具备以下禁忌证 2：

* **绝对禁忌**：近期心肌梗死、严重心律失常、肾衰竭、肝衰竭、恶性肿瘤、妊娠期及哺乳期 3。  
* **相对禁忌**：BMI ![][image29]、胆石症史、未受控的痛风（PSMF 引起的血尿酸升高可能诱发痛风发作） 2。

通过集成上述结构化数据与技术规范， PSMF AI Agent 能够将复杂的临床干预转化为精确的、可操作的数字化方案。这套数据库不仅提供了食材的“零件清单”，更通过约束算法赋予了 Agent “工程化灵魂”，使其在追求极致减脂效率的同时，始终锚定在临床安全的基准线上。由于 PSMF 的严苛性，所有的数值计算（蛋白质、脂肪、碳水）均应以 ![][image30] 克为最小步进单位，确保在数据层面的绝对严谨 2。

#### **引用的著作**

1. 683435c0d192b850664144b8\_6, 访问时间为 五月 2, 2026， [https://cdn.prod.website-files.com/68053eae83c4661ee05a7c54/683435c0d192b850664144b8\_65520743076.pdf](https://cdn.prod.website-files.com/68053eae83c4661ee05a7c54/683435c0d192b850664144b8_65520743076.pdf)  
2. The Beginner's Guide To A Protein-Sparing Modified Fast (PSMF) | Diet vs Disease, 访问时间为 五月 2, 2026， [https://www.dietvsdisease.org/protein-sparing-modified-fast/](https://www.dietvsdisease.org/protein-sparing-modified-fast/)  
3. Protein-sparing modified fast (diet) \- Wikipedia, 访问时间为 五月 2, 2026， [https://en.wikipedia.org/wiki/Protein-sparing\_modified\_fast\_(diet)](https://en.wikipedia.org/wiki/Protein-sparing_modified_fast_\(diet\))  
4. The protein-sparing modified fast for obese patients with type 2 diabetes \- Cleveland Clinic Journal of Medicine, 访问时间为 五月 2, 2026， [https://www.ccjm.org/content/ccjom/81/9/557.full.pdf](https://www.ccjm.org/content/ccjom/81/9/557.full.pdf)  
5. The Real-Life Use of a Protein-Sparing Modified Fast Diet by Nasogastric Tube (ProMoFasT) in Adults with Obesity: An Open-Label Randomized Controlled Trial \- PMC, 访问时间为 五月 2, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10674249/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10674249/)  
6. Protein Per 100g in 65+ Foods — USDA Data, Ranked & Filterable, 访问时间为 五月 2, 2026， [https://proteinatlas.ca/tools/protein-ranker/](https://proteinatlas.ca/tools/protein-ranker/)  
7. A Guide to the Protein-Sparing Modified Fast or PSMF Diet \- Keto Lifestyle \- Ketogenic.com, 访问时间为 五月 2, 2026， [https://ketogenic.com/a-guide-to-the-protein-sparing-modified-fast-or-psmf-diet/](https://ketogenic.com/a-guide-to-the-protein-sparing-modified-fast-or-psmf-diet/)  
8. Simple Cooking 101: Chicken, Seafood and Meat \- Mark Hyman, MD, 访问时间为 五月 2, 2026， [https://drhyman.com/blogs/content/simple-cooking-101-chicken-seafood-meat](https://drhyman.com/blogs/content/simple-cooking-101-chicken-seafood-meat)  
9. Food Search | USDA FoodData Central, 访问时间为 五月 2, 2026， [https://fdc.nal.usda.gov/food-search?query=chicken%20breast\&type=SR%20Legacy](https://fdc.nal.usda.gov/food-search?query=chicken+breast&type=SR+Legacy)  
10. 100 Grams Of Chicken Breast Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/chicken-breast-451?a=0.847457627118644%3A0](https://www.eatthismuch.com/calories/chicken-breast-451?a=0.847457627118644:0)  
11. Chicken Breast, skinless, raw \- DataYourEat.com, 访问时间为 五月 2, 2026， [https://www.datayoureat.com/r/PaWYrLtSJ2bg/Chicken\_Breast\_skinless\_raw?locale=en](https://www.datayoureat.com/r/PaWYrLtSJ2bg/Chicken_Breast_skinless_raw?locale=en)  
12. Nutrition Facts for Raw Chicken Breast, 访问时间为 五月 2, 2026， [https://tools.myfooddata.com/nutrition-facts/171077/wt1](https://tools.myfooddata.com/nutrition-facts/171077/wt1)  
13. 100 Grams Of Turkey Breast Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/turkey-breast-560?a=0.1282051282051282%3A0](https://www.eatthismuch.com/calories/turkey-breast-560?a=0.1282051282051282:0)  
14. Turkey, breast, without skin, raw \- Matvaretabellen, 访问时间为 五月 2, 2026， [https://www.matvaretabellen.no/en/turkey-breast-without-skin-raw/](https://www.matvaretabellen.no/en/turkey-breast-without-skin-raw/)  
15. Food Search | USDA FoodData Central, 访问时间为 五月 2, 2026， [https://fdc.nal.usda.gov/food-search?component=1100](https://fdc.nal.usda.gov/food-search?component=1100)  
16. 100 Grams Of Beef Eye Of Round Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/beef-eye-of-round-5484?a=3.527336860670194%3A0](https://www.eatthismuch.com/calories/beef-eye-of-round-5484?a=3.527336860670194:0)  
17. Beef, round, eye of round roast, boneless, separable lean and fat, trimmed to 0" fat, all grades, raw \- Nutrition Facts \- ReciPal, 访问时间为 五月 2, 2026， [https://www.recipal.com/ingredients/11691-nutrition-facts-calories-protein-carbs-fat-beef-round-eye-of-round-roast-boneless-separable-lean-and-fat-trimmed-to-0-fat-all-grades-raw](https://www.recipal.com/ingredients/11691-nutrition-facts-calories-protein-carbs-fat-beef-round-eye-of-round-roast-boneless-separable-lean-and-fat-trimmed-to-0-fat-all-grades-raw)  
18. Beef Eye of Round Roast \- Nutrivore, 访问时间为 五月 2, 2026， [https://nutrivore.com/foods/beef-eye-of-round-roast/](https://nutrivore.com/foods/beef-eye-of-round-roast/)  
19. 100 Grams Of Tilapia Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/tilapia-3499?a=3.527336860670194%3A0](https://www.eatthismuch.com/calories/tilapia-3499?a=3.527336860670194:0)  
20. Fish, tilapia, raw \- Nutrition Facts \- ReciPal, 访问时间为 五月 2, 2026， [https://www.recipal.com/ingredients/4611-nutrition-facts-calories-protein-carbs-fat-fish-tilapia-raw](https://www.recipal.com/ingredients/4611-nutrition-facts-calories-protein-carbs-fat-fish-tilapia-raw)  
21. Fish, tilapia, raw \- Nutrition Facts and Information \- Medindia, 访问时间为 五月 2, 2026， [https://www.medindia.net/nutrition-data/fish-tilapia-raw.htm](https://www.medindia.net/nutrition-data/fish-tilapia-raw.htm)  
22. Fish, cod, nutrition data \- Zoë Harcombe, 访问时间为 五月 2, 2026， [https://www.zoeharcombe.com/nutrition-data/fish-cod-nutrition-data/](https://www.zoeharcombe.com/nutrition-data/fish-cod-nutrition-data/)  
23. 100 Grams Of Atlantic Cod Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/atlantic-cod-3268?a=3.527336860670194%3A0](https://www.eatthismuch.com/calories/atlantic-cod-3268?a=3.527336860670194:0)  
24. Calories in Atlantic Cod, raw \- CalorieKing, 访问时间为 五月 2, 2026， [https://www.calorieking.com/us/en/foods/f/calories-in-fish-atlantic-cod-raw/V2bxhYJtRfacMQb82XQwEw](https://www.calorieking.com/us/en/foods/f/calories-in-fish-atlantic-cod-raw/V2bxhYJtRfacMQb82XQwEw)  
25. 100 Grams Of Shrimp Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/shrimp-3399?a=3.527336860670194%3A0](https://www.eatthismuch.com/calories/shrimp-3399?a=3.527336860670194:0)  
26. Crustaceans, shrimp, raw (not previously frozen) nutrition: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/crustaceans-shrimp-rawnotpreviouslyfrozen](https://foodstruct.com/food/crustaceans-shrimp-rawnotpreviouslyfrozen)  
27. Shrimp nutrition: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/shrimp-nutrition](https://foodstruct.com/food/shrimp-nutrition)  
28. 100 Grams Of Scallops Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/scallops-3421?a=3.527336860670194%3A0](https://www.eatthismuch.com/calories/scallops-3421?a=3.527336860670194:0)  
29. Scallop nutrition: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/scallop-nutrition](https://foodstruct.com/food/scallop-nutrition)  
30. Scallops nutrition: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/scallop](https://foodstruct.com/food/scallop)  
31. 100 Grams Of Greek Yogurt Nutrition Facts \- Eat This Much, 访问时间为 五月 2, 2026， [https://www.eatthismuch.com/calories/greek-yogurt-5811?a=0.4166666666666667%3A0](https://www.eatthismuch.com/calories/greek-yogurt-5811?a=0.4166666666666667:0)  
32. 100 Calories Plain Greek Fat Free Yogurt Nutrition \- Prospre, 访问时间为 五月 2, 2026， [https://www.prospre.io/ingredients/100-calories-plain-greek-fat-free-yogurt-176249](https://www.prospre.io/ingredients/100-calories-plain-greek-fat-free-yogurt-176249)  
33. Greek yogurt nutrition: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/greek-yogurt](https://foodstruct.com/food/greek-yogurt)  
34. Cottage cheese \- Wikipedia, 访问时间为 五月 2, 2026， [https://en.wikipedia.org/wiki/Cottage\_cheese](https://en.wikipedia.org/wiki/Cottage_cheese)  
35. Cottage cheese nutrition: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/cheese-cottage-creamed-largeorsmallcurd](https://foodstruct.com/food/cheese-cottage-creamed-largeorsmallcurd)  
36. Nutrition Information for Raw Vegetables | FDA, 访问时间为 五月 2, 2026， [https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/nutrition-information-raw-vegetables](https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/nutrition-information-raw-vegetables)  
37. Dark Green Leafy Vegetables : USDA ARS, 访问时间为 五月 2, 2026， [https://www.ars.usda.gov/plains-area/gfnd/gfhnrc/docs/news-articles/2013/dark-green-leafy-vegetables](https://www.ars.usda.gov/plains-area/gfnd/gfhnrc/docs/news-articles/2013/dark-green-leafy-vegetables)  
38. Protein-Sparing Modified Fasts \- Paleo Leap, 访问时间为 五月 2, 2026， [https://paleoleap.com/protein-sparing-modified-fasts/](https://paleoleap.com/protein-sparing-modified-fasts/)  
39. Skinny Apple Cider Dressing \- Revivelife Clinic, 访问时间为 五月 2, 2026， [https://revivelifeclinic.com/skinny-apple-cider-dressing/](https://revivelifeclinic.com/skinny-apple-cider-dressing/)  
40. PSMF (Rules And Meal Plan) \- AWS, 访问时间为 五月 2, 2026， [https://thinlicious.s3.us-east-2.amazonaws.com/PSMF+Rules+And+Meal+Plan.pdf](https://thinlicious.s3.us-east-2.amazonaws.com/PSMF+Rules+And+Meal+Plan.pdf)  
41. Tilapia nutrition facts: calories, carbs, GI, protein, fiber, fats \- Foodstruct, 访问时间为 五月 2, 2026， [https://foodstruct.com/food/tilapia](https://foodstruct.com/food/tilapia)  
42. Spinach — Calories per 100g · Protein, Carbs, Fat | CalZen, 访问时间为 五月 2, 2026， [https://calzen.ai/en/calories-in/spinach/](https://calzen.ai/en/calories-in/spinach/)  
43. Kale \- Wikipedia, 访问时间为 五月 2, 2026， [https://en.wikipedia.org/wiki/Kale](https://en.wikipedia.org/wiki/Kale)  
44. Spinach \- SNAP-Ed Connection \- USDA, 访问时间为 五月 2, 2026， [https://snaped.fns.usda.gov/resources/nutrition-education-materials/seasonal-produce-guide/spinach](https://snaped.fns.usda.gov/resources/nutrition-education-materials/seasonal-produce-guide/spinach)  
45. The 21 Best Low-Carb Vegetables \- Healthline, 访问时间为 五月 2, 2026， [https://www.healthline.com/nutrition/21-best-low-carb-vegetables](https://www.healthline.com/nutrition/21-best-low-carb-vegetables)  
46. The Protein-Sparing Modified Fast Diet: An Effective and Safe Approach to Induce Rapid Weight Loss in Severely Obese Adolescents \- PMC, 访问时间为 五月 2, 2026， [https://pmc.ncbi.nlm.nih.gov/articles/PMC4784653/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4784653/)  
47. Protein-Sparing Modified Fast Review: Does It Aid Weight Loss? \- Healthline, 访问时间为 五月 2, 2026， [https://www.healthline.com/nutrition/psmf-diet](https://www.healthline.com/nutrition/psmf-diet)  
48. Top 15 Vegetables Lowest in Sugar \- My Food Data, 访问时间为 五月 2, 2026， [https://www.myfooddata.com/articles/low-sugar-vegetables.php](https://www.myfooddata.com/articles/low-sugar-vegetables.php)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAAA9ElEQVR4Xu2UzQoBYRSGj7JlQ3EdSrgQ1yBElI07sHAHVlYWUu7CTikXoNjIhq3/95jB5x0zY4qFmqeeZnq/M+c0f59IyD/QgmUOfejAAzzCPq09GMA9vNhWXpc92cGMfZ6QZw9PggzJwTWMGVlWrB5TI3MQZIg+Jq2fU+57N0GGKEMYp+zrQ5iCWD10uCtaUOUwAHr9mUNGi2ocfsgInjh8hw6pc/gBJbjl0A0d0uDQhzxcUOb74psc2hRhirI0nFGmuA5JirXY5QUQEeenGTUydmLU3dDPbQNXcGkf9U/WrcZkLNbedqcnzuZ320ZdSMiPuAI0tkZFujNCFwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAAA/UlEQVR4XmNgGAVDAZQAcSa6IB7wAYibgVgWiFmA2AKIz6KogILlQPwLiP9DcRaqNF4A04OMC1FUYAHkWFIJxAuBOB1NDicg1ZI/6ALEAFIt+Y0uQAwg1ZKfQDwViD8C8QoGiH4rFBVYAEhRNrogHvAJiN2R+GYMEDOEkMQwAEhBLrogiQBkBshynACkIA9dEA9gRRdgQCRlnAAkWYAuiAPEMkDUl6KJE2UJrswUCsTiSPwkBoh6aSQxEACJ3UITgwMRBoiCHnQJIGBkwO5CdP4WLGJgsBqIXwPxEyB+DKVfMkCKGmSwgQFStiEDmMNASRhEf2eAOGgUjAI6AACjN0IgC5nhQwAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAXCAYAAAD3CERpAAABbUlEQVR4Xu2Uuy9EURDGR4hEY0WhoFEoJYhGIrKNRKOS8A9o+CMotlVJ6IhGVAqtRyWRoCLxqhQKpfcrQpjZmXN9ZxxxW3J/yZc73zdz7mR3716igv9MM+uS9cE6Z5XidsYU6471xBp3vUAHa4/0Xluul1FPujBQQ3qgBzLhhLUJ/oi1A14ok54NdDufceMDppP1Dr6R0ocla3J+Erzwytp1WXKwy/LAgfMByRasbjEvV2TD8ohrC/H7l08/AF763w5SnE9DjSxRIq+1MEgWDkcT+ZauQY3MUzqnVooXH8ftXEu3oUZmSfM2DIdYz1aX6etGh9lEvqUrUCNzpHkdhqnBC4rzPEt/+k0XyeWjPgAk77P63rxHslOr+83/+vTKCyB1MwHzMecDkvU6PwJeeGBduaw6WHGZePlvIjI3AX7GMmSd9QY+vN3aIcu4JW2e2XU5bldpIO3tkz5kL6Q39UjvkbVKOj8YtwsK/iqfRad9HIdrXEYAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAXCAYAAACI2VaYAAABxUlEQVR4Xu2UvUsdQRTFr2AKGxEJNhaxsLMRLBUs4l9gYZXOxg8MCmm1SiFEi0A0jWKXFFrY+tUIgqZKRExh50cRRDF+JlGM9zgz+87e3VHBQoT9weHdc+7sndn3dp9IQcHT807VbUNiUHWsOld1ml6gXrWm+q9aND1mVPVPdaBqMb2Er+IWYRjUk24nbKoWyG+oVsiDVnEzAo3GB36r3pO/UA2TzyV2uErJ3wRZlfH2m8eNr5J/LdlZ1TlZhtjhvkv+xcgmfF3jPT6ZeZ8Hwq9kQfbGhkzscMhjA0M+RDUzJekc9SX5APJ1GzKPOdws1cyYZA+Hl8qCHM9eFCzotaE87HDLVDMfxeW13qM+LLUTYnskoNlnQ4lfyPkXqplP4vJy71EfldoJyK9tyGDBWxvKww4Xe+YmJZ2j/ks+gHzLhgwW9NtQOZH8jZH99HWz9/e9rXfd6GcbMlgwYEOlQ+IDm4xvJw9OJf2MjUt2VpnPXpg84aW4BSO24UGvi/wHnzFzqivyYdM6ygCyBvLfJP8NlmnVvmpXteM/f4n7s2QqxA3FoB+qP+I2t6B3ppoRt74t3b7llbjekmpbtZduFxQUPB9uABU0mRTsAc05AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACMAAAAXCAYAAACBMvbiAAABrklEQVR4Xu2VTShFQRTHj5KPlCgbOyElFoqVspKVJBtfZYVSlnbKxkqxYGHFRtnZyU7sXj5KUSKykmShFIVE/E8zl3PPm3N7CqXur369Of+ZO3Pvm3fnEaX8PV06CNCkg5+mAT77z1fYEu/+5Bz26JA5grtwHA7BAdgP+7ySKXgPH+Gw6mPeYY1vl8Mn+AYnYAdc8mNu/JgsuNPyTow7gZuiPoYZUTN8jVVX+k/+5kz4gjZyX20trPbKiUpVHcFZmW9X+Fqi60kytifiQAdgBzaK+pCyJ2Y4W1a1RNb5lLA9Fq1wTWXRtml0/gIHfbsdjqq+b5PLohGhnOtVeCuyadgt6pxY8WpCizJWLimA16KuJ/c2bossCE9cpUOyF7VyCZ83EfJFKIGnoi/GCNkTW4taecQM7BT1JVwX9YVox+BT0Zr4gcJ9nFlPVwSvVMbjF0U9L9oxkp6yl8J9nDXr0MMnr0bfzIJox0i6GYb7xkQ967MQc+Rebc0e3BC1uU08sfyxaYrJjdkn93/Gx3pebISDt8dapJC+HoB/wPyX8quc6UBRR+4A3NIdKSn/lg/TO3tyE5g4JwAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADwAAAAXCAYAAABXlyyHAAACEElEQVR4Xu2WzUsXURSGj9TCP8DahBEapq2qrYsS3Au2UHDnTsW+KESKIkoQFEFR0UWEIIXowm3hylUfi+gfcFObiFpZC8nqfb136no8d2aQUVDuAw8/5j13jnPHe2dGJJFIHDfuwj4dVsh5+Bb+gWuqlgfHP4G9sAd2wy5vvR9zGy7Bi/64GS7CW/74Hy/hlrimtH93uTKuiuufcUkdxzgr/6/NknXy1Kh98LUoBzlh9tarhzf6jco0A3ACXoZNsBE2wCE4H4x7BKfgAnwATwS1KAc14dPievM35LXP83ilA3AS/lQZJ9mmskL2M2HuyyIeij2x52LnRfzSAbgvhzDh3+LOOaMLilWxJzYjdp7HLHysQzAMR8T1y25kuORNOIh7pixj4vZhEetiT2xSyt2wEKsPuSN7lz/H8ukehQMGdVgBL8S+0GlxOfdkGcbF7hODY3PHs3hDhxUQ28PPxM5jcOw7HXpqdAC2paA/i3te1BXQKq73fp7SIRw7qkMPa9+MLLc/i/xiKUutlN/z7N2psk34XWV82lp0iOtxUxc8rN0zsuiE68QVuU/KkjXMPufy4AMlfJ1wCfLcc0HG1cXsS5BlzImr8fPSgu/lU8HxNXHjLwTZDsvwK/wMP/lf/sEyT1/e9fc6zOEj/AFXxF1M++7yDhuwRYfgurhzruhCAJd09k+g/CJLJBKJxJHgL0l+jQzaJjQiAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAYCAYAAABtGnqsAAAC3ElEQVR4Xu2YS6hOURiGP5dcconORJKOKJJEDkVmDAwkl4lLUaKUTBgYOIpCjAyYGxi4pNxySbll4k6kGJi5JYWI3H1v6/v+/f2vvX9mzr9bT739e73v2pe1ztprr3VEMplMJuPsUL1X/TJ9Vr1VvVP9MO9Fo3bijfmuL6qBlr2i7IP5YCZluE9t8EYxIyX5H8mfYf4+8sEGSdkKDgy+Vi1Ag0+xaZR17jDzzpIPtkrKujlQTqv6s9nuLJDU4CkcKMMlZXi1GfhP2FR+SsoOkN9bdZm8WnBH/hxhjo8+NJ6B/528q6pBll2jDPNjLfFOGmuaoNpr3pFQj/HzHLyaZ0L2MmSdqs2hXCvQ2Euquao59rvSfO+QMrgD8TV3OIvHtcLnv6kcSBpRyJ5yYMROmqRaX5GtVk0LWbuAqeiv3JPWo4NHUiRmXCdm7bhsma+6z2YZrToItMo9w7pvYkV2XIpFdjuBgbWIzTLQyPNsGr5DGc2B4XnZjsI78DoHxEnVOdUzK89TnVCtadRIOyIwRHVYdVDS+hPnYXWA5RfKayXd74HVByNUnyQtqXAuWCqpg7pUd1WHpBhtGAj+7JCfU8pGSZWmkz9e0vYMGXYcVeDmqNOPAykeoBVo2Cg7Rt2hqi2qJaor5s+WYq2JTu4lxXXx26F6LWkOx/rTfTBZ9c2OAdoEsLJYp/oaMn5WLjexW9LJvuB1oYx1HT4ayxu1qzmqusim8Vy1nc3AMql+SHTsODvG9VeFDA3HyGMwUniZhOs/VN1QPZbmHRD+GLNCOT4LPqz/NP/9T26q9rNpxMZwJ+MfG2VvBer1LfGqiNkmad7G4s1aHMo9km2qXaGMZVCnHXvj4uu6kDKmzMcb1SeUL4TjWB/1sGX15ZpnmA8x7/ZYsIXERwRz2JjgH7Nsj6RGPQoZ6jIDJI0aBl9/dM5tKeZUgI0CdloOprRbqsFW3ilpz465OJPJZDKB3wE53cSvNFw/AAAAAElFTkSuQmCC>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAACJElEQVR4Xu2WvUsdQRTFrzGYIoiJRQqJNiJaiCQQQxorRTv/AEULUYM2QQvByhQpEiSkCPkj0ggKBpMinYWIWNmplYgg4kcUNX7ew8w87x7f7nMjxML9weHNnDs7X+/O7Ipk3D+aVE/YJBrYuGvOVH2qKdUkxQJtqj02mQ+qXdWF16FqW7UjbhB467nWVzyQq2cKqcg/81W16ssgxCdUraou1V/v3ZjQCVMhzt/ngOeHuPhTDijPJNonyuOmPm3KSB0ssFfVb/yCoFP8ffmIWxRIigGe+Jipv1c9NHVQMEUs7eI6fcEBcTuJGFIoH4idkFdtynbiB6pvpj5ryiDVpMGCxO9a2FHkNPNKXOwT+WumjIMWGBaXw4FzUx4Ud2hTESaHnYLqVF+89920Y5CjaFNqvCXVW1Nn5lTL4g5+mfeQ27ggUoPBf6taVM3+t9v79gAxYcGstCCFLD/FLaSc/Aghv19yQHkkLrbCAQ9i2Dn20jCk6jF1nJc3vsx9R1iU5MHidhGdw/9M/gbVk8C5wTsjUCPRsZ6r3pl6hLiJBeLiv8T5IU//Bb6pRuX6WHhP5AUNZ9j0hDdqFQckfkE3ZUTVSd6AXO8z7xnD9YSGjeTXqo597DXFQIncbuLFqk02lccS7bNSKFU+irtPcY+GCUCon4o7jB251lFwA+A7Zktcfv6R9As4YsOABdX7cuLh/N9gF/FBlcS8uE1JvA4zMjIybsclOOCefPSqjRgAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAYCAYAAABZY7uwAAACx0lEQVR4Xu2YS8hNURiGP3LNhBgp+ku5JblEGRi4ZOYyI4oiJbeBlAn9KclAKAOFYmDCRMhl4JIYGLikFBIGcklyyf3ue33f9+913rP37p/4/7O1n3rba73v2uectfbaa699RGpqampah22qd6rfrs+qN6q3qp/uPetobbxyP/RV1d+zF5S9dx9MowzfUxniRzNDxfwP5E91fy/5YJ1YtoQDhz+rEqBDJ9l08gZvoHtnyAdbxLLNHCinVH3ZbHXmiXVoAgfKILEMtx4D/z6byi+x7BD5PVWXyKsEN6R5hgQxe9A5Bv4P8i6rBnh2hTKsT5UkBmGEa7Rqt3tHk3ZMnBfg1jmdZM+TrE21KalXCnTmomq2apYfl7ofHc6DBwhPw4CztFwpYv2ZyIHYjED2kAMnHYRxqjUF2XLV5CRrFXqxkcctKb+6PBNS0ozbpFnZY53P6yp2qvawmUfZAICyPDLse8YWZMcl20TmUfTZ/xo8aQezmQd+4Dk2ndhhD+fAiTxvRxwDdI0DJ3K+ANhW4POwJi4Qe2CA26rvXj+huut+gD4cELsg6dNzodjOH/kk1Wpp/N5VWdNmNog1mkL+KLHXB2TYMRdxU6xNHw6kueN54Pz5Sb2HNJ6Dcm8vjxd7/VmvWiQ2WMET1Rgvr5XsohxWHfEyOO9HbHIxgwrZofom2YYuhDr2NViUF3e0LuaY6gKbzlPVVjYJHkB04GBS55zrAAOY+ndUc70M/6zqkTQO1C7p5PrT3XCHUR/m5TbVpyz6C7cHG1X3kjrPwDzwEt6p9ac7mSnZOvLYjx/9CHD7bVft8/pKsdnAzJDslkK5bICwjoHwl0XQquA2xzYjGCK29mFRxRV+rVrh2VXVSC8z+8UeGFh/rif+dNUXsb9r2hMfywJeibB3++/BUy94oJqT1Gsku11we71Mgxqjn9jg4B+Emq7gDzBq19JEzHaaAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABBElEQVR4XmNgGAVDAZQAcSa6IAHgAsS/gPg/EG9Hk4OD5QwIRSCchSqNF3gA8RwkvioDxAy8gFRLsBm4BohnogsiA3IsEUATWw/EC9DEUAA5loBwNpoYXkCqJbYMCIu+Q2keFBVYALqriAG5DAiLQLgFVRoTgBSBNBELNgPxXih7AgPCoji4CiwApCAPXRAHUGPADH9OqBi6OAoASRagC+IAV4F4MbogA8IinAAkWYguCAWhQCyOxN8IxFeQ+MgApyUiDBDJHnQJIGBkwAwGFihfBkkMBO4CcTiaGMNqIH4NxE+A+DGUfskAKWqQwQYGSNmGDKQZEJa/gtKBKCpGwSigGQAAcTFFYgtEghAAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG0AAAAYCAYAAADwF3MkAAADt0lEQVR4Xu2ZWaiNURTHF9c8PBgezFd5QEqmkjIkSV6ETGW4MqYUD4rIEJJEpkKmG5J44IkypgghyZQpZIjMQ+YM62/tfc7+1tn7nuu75R7nfL/6d/f6r286e397+PYlSkhISEhIyE/WsN6xfhl9Zr1W3r7U0YVDNdY3bTrUZR0hqZ8LrCrRdIpVJNd5xeqpchXGNpCmmMTHAxYCNyldF776AM1JcrVN3MjEVVNHCHjxlzoxOsRyJ64wuOlZbRrK+gH5ynsK/+aPrL3Ku8j64sT9KPP8hh4vNqNILjZAJ0jepqTRosAfoby5xrdgSPSdD2+MNuNwg/w3AAdIcoN1Is8JNVpvEl/PTyXGR28CKH9Pp1PAv6JNwyCSHtxWJ3yEelJfEn+tThQAoUabSeJ3Uf5w43c3Mcq4hgY+5jaXItYP1iQTX6KyG/cPttHest6QjM2Ir5JMsrlIfdaugHaydrBKWdtZW1lb5LRyE2q0xSR+R+UPMf5oE6OMVbjG1rX29nu8ycpLYeezCTpRBiUkDTtHJwLgRbCspMyHzkVCjTaFxO+k/GHGxwIEoIxOoIH/04nRWPo+3TxehFuU5YAAOKeGNgPMUHGc+/1rQo1m57Qeyh9rfHwOAJS/ptMp4N9Rsb7PQY8XwXdSeYhzDhjIuqfNGNRkrfhL/Q2hRsN94WdbPYbqFd5GFR91Yuu9UF4EHHBXmwEwx+1mrafoZNqAZAg8QbLKbGf8hazrrPYmBudI3krLc5JJ2DKb1dmJK4tQowH465R3yPiWDSoG2DWBV93xEKOeLPhAhzfd8SLMIzlgqk54wHH1TPkMa5Ep2wexuA9VQrKNszqdjhw7y+PpH1pZfKLws+heBRAP9XgdnPg8Za4oT5HsnABMNzgHytgWQ0/5QNI7sMLBF747OWomUmbF1jLlYySrMzfnki3Gi6CvXZmgXp6yHhk9IamjNu5BzB6SEQJ/8cz4FNAUk+SOsx6SXMsHPqnwkqDzZJ3Pystl1jYn1pXc0pRbk9zcgmHTHfqw46KHYizNl5gyJvlnTq4QQX0+1mYcMIyOM+WRJFs0TUl2wtFLLfgoXEbpifYkyXfLAhNj03k8yZBgwTzZx5Qx301zcvnObdY1J8ZuChqtq+NVCDTOA1YLkt5z2viNSZa2L0k+xDGUYDgFvUgWIrZRMFzcp+geZxHJN1/cT4//Gfxeu9vUysSb0uncBXMiaEbRXlsIYFopZR1mzVe5nKU/azOrDslCSP8/KiFHaUKyj5iQkF/8BhAnIAaFP2tUAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAXCAYAAABefIz9AAACPUlEQVR4Xu2WP0hWURjG36CsUIOiRSIcBEFcGiIi3CIcIqKhJXIRxDnUwZagGhoKarCGpsBBnEJaay9oCUEqJ7FwUEkHK0vreTj39r3f4/lO53MQhPuDh3vf57z33nPu+WtWUbFXXFEjwhk19gO90I/i+hs6W1/8j0/QNTVTHICuqxnhG3QPOg0dhM5D7+sy0jyCNqEVqE/KyB+oq7g/Dn2HtqAR6BL0vMhZKnKyOQqNqhmBL1fdqstozBp038Ws/AMXE76vUdxRXNnDTdMGjakZgR8ch15Aw1KW4qLtrPwJ8U5KTDS+bU0OzZJ2y2sg58Vu4LDUyhJ6NyX2+JhToumhWZLbwF9qZMKKxp6l/8HF/BE3inv2+pCU7ZrcBv6EJizMpykLFbxQlxGHeetqWvA5F9WbhJaddxe66uIkfEEz4opZwkr2u/ichRzOpxTMWVXTat9I0QJ9dXEPtAG9cd5/ye3BGKxgrHc8zOEWo9DfVlPw8/6Y1X5IKzTnypLkNvCQGpbXCyzn8Fbof1bTwW3ksosXoBkXz7v7JDkNHLBQIc3LbWAsh94zNQuOQIviMZ9rQMljd58kp4GDFj5wSnx6PDp57kj81HY2kKcnerFRQXiCUbSBT9x9kpwGEq3kq4j3svCmxafHM2bJO2s8dx9a2CaUtxa+WRIdouVwyZXvsfK0wW2CVy7x7AkPz5BfLCwInk4Lz7y2MJeYE4NDM1pxcNjCOwgXmVlXtm/4qIbQbWHT54+qqKio2Dv+Aih5o2EEPZaCAAAAAElFTkSuQmCC>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAG8AAAAYCAYAAAD04qMZAAADxUlEQVR4Xu2ZWahOURTHlzGSIQovlChDpogUT0hmRYYypAwZIg+iFA+8KC8SKYQ3RQmJDCkJUaZMGXINLzJkKvO0/nftfe/61rfPueecz+1e9zu/Wn1n/dc6++yz9zlnDx9RTk5OTk5OeXKa7U9Ca8h0ZXtKcp+32doURKtZz/aR7TPbAhOriT0k5T8nud4/I66DDrL9sGIDYhrbZeVXkLTFFKWBe2xnlH+H7aLy4/jNNkb5KH+s8jPThKSwqzbgaM123op1SFsrlAjufXpA0w8z3sTQww2tnRUNm6j43IkBLROrKfwktHC/aKyNOlDHbGb7yTbABjJiOwq8dlpv5990vgUaPodxIAdvrQV6Fyum5QMVV2wdWw933IytlYrVF9aQ1HuSDaRkKdsso30jKduPfaEOBlG6BvGzViTRt1nRgXvbTfJVjMVWAE9bTRWqT8whqe9yGygB2ybW90TpGsSPWZFEP2m0/k7v7nyMlfCnVmUo/HgXsv+NUST1xme1FHaQlNNTaVFtEqV7GpPED9sAif5I+S2dNlBpmExFlr+WJDhZaej9Q8qvTbqx3WB7YAMl0JdkdrzLBhLQgaQ9hho9qpOidA3iR6xIol9Qvn/LNMcDWhWfqDi4iKoHao3NiyNN7hW2uVYsgc4ka7ETNlADjUjqHZpERHVSlK5B/JQVSXSMa9o/qnyvYfIUJMnFPUnzQG3lxtGH5I3bawMJQT30xAyLcD/2hB5yAO2+FQ3IiZptznTHg50/vDpcCbQVRqukOUkQT34cvoNDHf2CbT/bF+dH5eJNfkZyE5jdNVUxW2ZaRpKUUcpy5isV1gncVcczKFxPaGh4DWbqGtyzPRefZa1hWQYfY6RnnNPwRShiK0nQ934c16l4x0FfHJ/aS+7Y5rZn+6V8fR7Wlo+VnwY/y1xsAynBdpV96Lxp4C9R/hanad45bZnScP/Q/LoZ4E2+pnyAnA3ueITzbfl0gORkXOgt23uSJ2+nTjLYQm6xbVf+KrZX7tjm4lpoaI+OY1tqtvKTsJKkjPE2kBHbYdo0fjaInSjcP9rMvhX92J4YDYwmORdjGtpdb8dpsKTAvmkvkvyXheFs2BuBrzdv0UF4+3xMo/35VLhgtblJwGy4HEDbLLRiWjCm+EG5wv1iYtDRHWPv038WQ7m6g7Cbgw1an+NjWM+UM99JxlUP/r3I8mAHQeFYj3kwuEM7x7ZP6cDmDmJ7w/aQbQjJwtQP8tiZh17OTCDpqE7On+f8YVUZOfUa/MuAjRHsxOhJUU5OTk5Odv4CcqoobLKZZ+AAAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGgAAAAYCAYAAAAWPrhgAAAD10lEQVR4Xu2ZWahNURjHP1OhEBESt8g8hgfTg+EBIUV5IVFERChkvA8oyVAyJEWiJKXIC/LkjaQQRUqGTJmnzL7//dY659vfWfvsc7Yr9zr7V//uWf9v7bX3XnvNlygjIyOjYunEWsY6wOqq/BHqd8Y/4CjrF+suaxKrO2sf6wlruIvVZ5azTrD6uHQv1jGSxmjZwfrKeskaZWJJnCSpq1us5iaWGhT4k9XKBpg1JPHrNlDP2EzyHlrXIjmEtyR5PZ9ZW1U6jiYkZXZx6UYu3TGXIyXfKbl3ID7Nmn+RZqzG1vxDqlm7WUdY60kq0DKOCuuiTcALcYn1yHjbqbRrY3lDUgAqpBh/dJOUHCd5vg42kBJ8lDHWNGBYC70rvFnWNCAPpgQN5u1QeSXRn+TiOzYQIPVNaoE9JL18kA2UyTpK/kB4z2/WJPGLDfF+OEMj0FQ5P270WcU6SOHeTD9ILg7NO3WRtSTPO8UGSgRz6RaSMg67v1ipauC9Mx6Aj7koDjQe5Flh/HbOX238Ac7v5tKY/ws+JAyovjGX5LkX2UACqLxzxkM5m0z6lUp7kupqLEl8qfFbOx+9xIPpBJ4eEaY7L0LSTes6E0meH0NXWmwd4DfmPQt8tPI4epDksUv2ts7XjcD3Fs1Z6/kx85k2Y7CFpQXzCFZEtcVAkmF6vw3E0MAalB/mPfj9RaU98LE/jANlIw+GUU1n589UHtKnVdp7L4xXYyZV/jDWHGumJOlepTKepKxqG0gA12DjaT37gULPCS+pISBP3CrO74WGuPTIXA4B3hLj0T0XCK4gSPznxrtK0kVbkqz+cMqACtPgROIy675Lowz/4qGXLxU/98yzgRLBtSsDnn4mVLB9Rt87sBH1NGUtVmmAerErPazSdHm4DumGyvNDdaiH1wQw9NiPNJgKuxy6KwrGNbgxwK75fS6H7MJxhAJaUP6YZCdrl/tdLphjcM8JNlAmn0hWVZ7RJOX2VB6A11el0djsys5/WH9sBPCu9uMijc2x9Ta63/4ae12EC5TP9NH9nR/JEUUXto3yXb+fi51nfWAt8JlIxvpy5x98VDQe7NdqCwxx/l0hfRjsqSKJXWQ9YD2OhmuYyrpiTcr3GJz5YdN7KBrOcYakwaAxI//TaDg9OEBFwR4U7rs+Hu6himmKtpAA6KntrfmfgrpJO2wXgNNfvVLxFY+NWG/WaxXDEnM2yZmaz3czH65I0KtmqPQGKr/xFgVDjp7gMGToSt/Luk0yZg9VPoa8GypdieBfOPgYfmRA40UaK+WMOsJkkv8XnWItNLGMjIyMSuA3e5YEymSNEsYAAAAASUVORK5CYII=>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAYCAYAAACBbx+6AAACFElEQVR4Xu2WsUsdQRCHR0U0CmLAICIqKEhIEQIWBrWyEwOCQpqAjSkEMQSLiGBsLLUykCZiYyWBQMTOUoSgjQli/oEUglYKUQQ183s7c8zNu336rB7hfTBw883u3t7e7d0RlSnz/zDAccpxy/GDozJdTvjIccbxl2PC1e7iK4XxjzjqXK0oPnF8Njkmg4E7jQM40bbJDzl2TR6jmsJ47ZJXSd6StCgSdO7NcAilweUKXKOXjh2OP84tU/Z4d1JP+ZMD3h24XIFb9dKBNvYOgj7xD2KRo985P2GfKzGv6O2fd75D/KjzygeOLxT63wsMduPyrInFvPKCQn3G+SfiZ51/Lr5Lcsyh0IXl+EWhkd3JsYnFvDJIof7O+cfisYrKI3G4SGVMXBRsPjTAClhiE4t5pZtC/b3zTeLxOCq6mpatDJegV13jCxSfWMwrFRTqc863iX9jHPLvJld34lwOfCj8idfN8Tnl1wHcby8daBN7S+i7uEfyrM0/7VwOu8GUa3P8muITxsmUWo4pkwOMjX1hwVvAjod+yO0Xdkgc7lKKKylkhQX5pMmXxFm03zPj8NnPareS4RbkWPv4ftRqCj4uTDugu3iP4yfHJeVf/QjHvnNAV3SDwgKtpcsJmxR+DZ5SaH+cLpc2mPBbL0sFrDr2iYK/wrzHoVQYpjC5ZsnHJX+ZtChBXlH4X/5G6Y1dpkwh/gECM6NNjcfWsgAAAABJRU5ErkJggg==>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI8AAAAYCAYAAADDAK5oAAAFEElEQVR4Xu2ZeeimUxTHj32LDFnLDP5iGGvZExGy/8VMw0iE8o8kFEVI/pEl2dNPklL2IgxNTEZEEbLEz75lF7I7n7n3/N7znOe+68z7m8X91Om953vu+yz3uc+9595HpFKpVCqVSmXsrK+2exTHyLpqu0bR85TavwPa/4l71HaOouNatT/UvlU7KMSMQ9XeltR294aY505JdT5Wmxlixncyvc/hCxnifL0qbivdY6sTi9X+kU5bzG6Gp/hR7Srn/6Z2jfPhfEnHMs6RchtS5wjnU+co53uIvRHFMcL5eEF6spakiq/FgKN04yuSWVFYjuwl3TvPYdJui80KGn4cudB8J7sya55jC5qBflIUxwjnuzqKkQskVTw+6Oe5crcbWpG8oPaB2gYxsIz06jy8iaW2QDsll0/MfuR3aeqU33K+gb5d0OZlfbrYQdL5NoqBCMNwvLCj1RY4f19XXtl4TO0ntW1iYER6dR70P6MoSX89l5/NfmRS2p1nofMN9BuD9m7WDWaLO6TdyYw5averXRQDGUaw29U2joHMhJTvoQWVsP0lJXkXZn9V41a1v9T2iIEh6dd56KgRdHIf+CH7kTel3Xkedb6B/kRBeyeXmdroOAdn3cNIgXZD9rkHX8fiPGewRPy+qRoJtIHznUlJ89tNap9mbVXlcknXf0zQB8U6zy4xIEmnwSPo1ma+7CGnNH3NXH6wE54C/b2CNl/tNEl5l2k+KV8na9c5LV4LZb/y2yRrcWZB84uCIgxrVIwZ/vuuvLnaGs5fz5X70XfOzJyp9rekuX15cbakezs1BvpgnYehP4LOyBLxD/Lz7EdYKcUH+bDzDfTnnT83a4xGezudF9/zjbTP66elV6UdP6OgzcrahkFv8bO0/0wPJucxSPQ8sX4vxlV3EI6TdMyLY6AP1nl2iwFJemwPQLfRolvOwwvpdcpPOt9AZ1oybK/IclMWOCWIvRJFB/F4XZMF7a6CVqR0QM8WkhrDmCHNobIXJ0jv5X+k13UMw1mSjnV6DAyIdZ5S7tStvdBuyeVLsh8ZZrV1cvDJl2Cn7PNcPCTO6H6REyFeym1iB0YrvSAN2H6m4ssx4CC+tit7M7hwdlpJ/l7MWre65CGcj7zKryiYNv1UOQqXSTqX33QbBes8e8aAcrM07weY0tEYsQ189n88aKwMDTpOPNY+BQ3f7+/g2x6SX/mhb+9848j8S/xAp1vedYike7g762i2v8NOc5HrJVWM+ztgexWxB5KXMPoY5AXMtQadwh5ebAQahrjh43Q62ycZltskNWIpwR0FpmyujVVNCWL+XC9JewX2paQpwdha0v98B6NzofHNyiCN8FMPL1tsR/MPl+aLwowQRxaW6w/lMv/jU4iBb8di2W7XhrafpOScFXgDDsiS1m/Fe0PnYbD0jEN36UZoGIMR6FwX8+DbvsSOar+G2LA8ICl53TIGRuRrta8kdfBP8i/aR76SdBLKZyR9j/qsGZ6Cl4rjPS6pPvccoQMQe0Tte7UlzfDSF/yXoJFk85+ngw5ctz1HEuStmuGlxyJme1KLsj+RfbBvbT5pX2Y2lXa+Ex+6+YxmdoGGrzuhdqnz43H6wZzv39jKSs4V0kkKbRPLP3Q+Z1hvfU46SZ/tcvq6lFkK8mYfoPahdPYvKqshJM7M7YucRh5DHsQ3Jr+6mSmpQ/gNKVYC7JIy3PKVmbmdTSpgOI27qpVKpVKpVCqVyrTzHxBaljb5hTv3AAAAAElFTkSuQmCC>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAYCAYAAACldpB6AAAC4UlEQVR4Xu2XS8hNURTHlzwi8iaP4lNkgJABJX3Kq2Qkj4GEDJCBlMiAgYmBKHlEkQhFkqFXKSKUorySRwwoA+9X3uv/7bXu/Z91zv2+O7kyOL9a7f3/r33O3mffffbZV6SkpOQ/or/GsGg2kL4ao6Lp9NZ4qfFH46lGj2y6wiaNDxpfNJaHnDNc44ake10MOWedxi9JbXaEXCOYrfFJUn+XQq6FTpImwGknqfF48sB9jQuk72pcJQ2aJV3rjAua6Scph/Jfgf6mRxO8i4YyWtIv5XSX4oeB1zPoVaTBd43rwQN7pfiejWKZtNJf0cDHmu/cDtqBd8DqeL+hUTLnzY/AK/IbxXNppb+3kpL8/mJ1TCFda8Dsb6Y6c0iKfXj7SKM/1pGlGic15gYftNfYqLEmJgj0dy2aDm7gD4PABMzJtKhvEs5QndkjeX+QeSgBNuMxGkc0dnsjY4OktiNMY9PdXk3LVkmvHOhq9dgfgDcjmowPyuNeNl3XJFymOrNTkj+YvP3mgddWTjVvoWngq6uD6YGmt5leZLqzaXDLPGZJgZdhlsZXqzdL9cHuVFrUNwnHqc7gl+UHAX7dN/JAt6DR5nTwhlId+Vek3cPnkHlmfk2Kki8k69czCbX2hIOS96EfW/lTo0823cJ6SfleMWHMk5RfEXx42B+idyV4FeZLfoAO/ElW/2g6Au+B1SebbuvrMMC0t3tkOvJEin3nrOTzQ8zrGHx4hecDgANRvJHD/oKgHXgTgo67N5bmG9K7JHuvE6QnSvqFQdEKcqZJ9TVjjpJ3zMrF5GHVjLR6BjTYEjxonA0YtFtJGptTHMQ5Scvb8dNnE3l+XHYOk+bJwmsAvwt5+JLh+iZJXwK+z0zT7nm5lup8AMzxXlLDh1ZiRiMYDHI3JW2a2NTwkBHkPmucktQ+LkN4OL0xP8z3z6Djr9hvK+M5AivCH3y1ed4W/4ccjAceVlpJSUlJSUlJ2/wF7+/vOUru6lgAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAXCAYAAABj7u2bAAABEElEQVR4XmNgGAWjgPqgBIgz0QWpCJKB+CsQ/wfiHWhycLAciH8xQBSBcBaqNNXAWSCOh7KZGRD2scNVYAG0chATEP8GYjkkMW4GiH2gwMAJaOUgDwZEiCADbGIogBwHqaAL4ABzgVgRTYzqDvrHANEjjS5BBOBigOh9jC6BDEAKstEF8YBuBgJpAA/4xkAgdEAApCAXXZAGIJ8BYhcot+EFIEV56IJUBsYMkKgmCoAcVIAuSEUgAsTf0cQIpqFCdEE8gIOB+DQHip736IIMeNIRyPUgyR50CTwAlm210CWwAJhadIwRfauB+DUQP2GABB+IfslAXO7xB+LT6IJYQDQDpkNgeCeSulEwCkYBTQEAY6xJsNVEQTAAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACQAAAAXCAYAAABj7u2bAAABKklEQVR4XmNgGAWjgPqgBIgz0QWpCIKB+AcQ/wfiLUDMhCoNAcuB+BcDRBEIZ6FKUw2sAOJkJP41Boh9XEhiGICWDoJ5GAb0ofznSGIYgJYO+g3E/5D45gwQ+84iiWEAchykgi5AJNjFALFPHF0CGZDqIJCPQXqk0SUIAFDGAemLQpdAByBF2eiCeEA3AyRDkAL6GCA57AsQa6DJYQCQg3LRBWkEXBgg9pWiSyADkII8dEEaAvSchwFAkgXoglQA/AwQs1PRxGEOYkQThwOQZCG6IB7AwUBcmpvMADH7G5o43hASYYBI9qBL4AEwA7XQJdAADwOmxaBSGyRWhybOsBqIXwPxEyB+DKVfMhCXe/yB+DS6IA6gxwBxAKiA/A5lExO6o2AUjAKqAQB1Z0fxK+vxvgAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAYCAYAAACBbx+6AAACKUlEQVR4Xu2WwUsVURTGT0lgLcQgkRYWKFS0CMFNWItsFwWBQRuhTS2EKEpQEdMW5ipXLgQpiqBNBEHiJlqKG11YIfoPtAiiTUFEkHW+d88Zz3zO9FKCHvF+8KHnd8+7c+e+mXkjUqfO/8FtzTPNcauPaZ5qbmUdm4xpPmu+aq7SWDWea35q1jT7aGxb3JM0UcxKriOBA70O9apmMdRl7JE05yGrG6w+mHVsk7uaac0TzR1JEzJNkg7CwDWzJBY078lNSfF8fwQW2cOSeCPFB4B7yJJAzwy5bvM7YlSqL9gvFabMO/71Y1Mih833kneGNA+k+NuWEc2kpAke29/ZXEf5wsq80ylpfIB8i/lh8ifMd1i9YXXuxDDZqygkNU1QXbSwMu+clTR+k/x+89hFZ685nKRzyVxVeCFcO2XeOSJpnB+RB8zHTfHdjMwXONnFQvkhf2fBmBvjuOwibeb7gkP9MtTuPpKryE8FLi7kC9UO3DpLAj1lTwl/FndZfSrrSMDdIFeRgwUuLvAy1Q4cDuY0aq6HGuCrfkcOT4E4Hz6Hendw58xtuQLwM4u71jkjqfFocACuP9T3zUX8RP1nHpw2F0GNHyt24/a/f4Y/l4FLwhuQ9vxwBb+LlzRvNd9k69lf1CyTA76jeGf5rnmUH86Yk7SBeJ9B/4f8cG2DBV9jWStg13GfOHgrLL0c/jXnJS2u1eorVp/MOmqQC5Lel19I/sauU+d3/AKAVp966ZEF+gAAAABJRU5ErkJggg==>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAYCAYAAACiNE5vAAACtklEQVR4Xu2Wy6tOURjGX/e7UBTScRsoBkIhygBlYiKMlGTCvyDkRFIuJdeBy4kyMhADuQ2QckkhMWCkZILc77f3Oet9v+/Z7147Xzkmp/2rp2+9v7Wsvde21zpbpKamphvRVzMtSme+5qXmt+ampmexu8EmzTvNJ83a0OdM1tySNNfl0Mfs1nzTvJJ0/f/BC0n3gZTYpzlINRaFgRPJgUeaS1Q/1NygGiyQ4kWmh9p5q9lG9WfNDqq7ElwfD7gEOmZnHN/w0FA7cMNCvZ5qgIviLXIWSnmuERnXVWDe7VEOso540ejuhdqBO2LtUVbjl7lo3sGDqJprVZT/yARJ82KdJbZq5gUXFx5rh/1majPHpTzXd6od+AdUL9PsonqMZqdmDbnemnbNIc1A8k6H5O+pEgz+FercBOzPUJs5IOWF44CMwGOvg6mSzoA95s9qtljfeUnjVkjz8Fwi+WvDZfd3Djx1/AN+gq0s/Bq1mb2S/Fir0X7d7G7Ac3203w5zi60GS83hATBwszKOD9FKcMhh8MjgW1n4KWoz+yV5vJYA7TfN7gbw/pZtIBf/x06YZ2aY60GuzVxuCxQYLmlgv9ghrS28ao8flaJH+yvVDvyTjIsnMpxvCeeqeeZYxpXAB0scdJLa76XcD+AeWxsHJOq/ner8sBg4HFLOaHODyQE4HGjRncu43AMuwAeZ85PaK6X6ZmeGGqcx80GKexofS3EuvKJwfcgdNsdMMtef3DhzQ6y+a79w/rbgC66E/13NhUG9jmr8aYljLmh+UO0LGk8OwOHkdm5L+aTHmC/B5fb3RnL4suxlbbg5mtWaueYa4KSNi/XEfTTAPG7yvqSb4sPEQR9O5dOSxi8qdnfSJqnviuaZ5nmxuxP0Lw/uqeZOcADfBRg/hRw+rOCuk6upqampqeku/AFqre7f/g4mPQAAAABJRU5ErkJggg==>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAYCAYAAAD6S912AAABJUlEQVR4Xu2UvUoDURCFx19iJ4JgYSVCxOAriDY+Quo8gajBQhQs1EJipVWqIIggCNYpxEfwSSxsrBQ9Z2fWzB0W9GrKfHDIne/OzrLh7oqMGBa7yB2yavUKcoPsfHdkcop8hjwnHZkcI5fINXKETKTb+XDIZpT/4VDyB54hnShLDkQb+N/17LebdAxoiu7PIlO2ZuZ80x7S90K06SQ4ngL6GeeuzP1Ieefo3oN7NZ8wFgX4kLSxYTXPrIfuPrhCvlQ4P7Bt9aRzXNOtOVdAuV/h/MCtUJPzClfwhsy7ekO0se4coVu39bbVlQMJH7lsYJbS7YJx5Em0l8eEffwGDIVp0YHLceM31EQvXnTu0dyf4CvmL76wesG5bFrIA3Ir+vqNUL4AlZpIcsJ3bHUAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAYCAYAAABDX1s+AAAC6ElEQVR4Xu2YzatNURjGXx8RInSNbiQpYkgZIO4/cAcG/AGIgeQjA6GbkIERSlcp6SoRZgYY3IFIBr4SJSMxQvKRz3y8z1lrnfPuZ7/r3K9zznVzfvV07/ustff77rXXXmftLdKmzVhjGRv/MxNVF9kcLVap3qr+qO6qxhebqxxQfVR9UW2ktpFySTXJxDslDNCSGC9WnVftqPao0dC6TqpOmRgnxcAsMB54qrpp4ieq2yYeKchpORw9qweFHoGG14VEKxzPFjiD4gS8mWwOAwz+IfJ6VCdU51T7VROKzRUaXtc0KV88YO8hxQl4Z9gcBi/YkDAIXWwSTakLd2YleTwgHCdyPuiWsAYs4gaHn2wo+2TgAcnlz/kAT8MF1RpuqAdO9ptiL4HnY2r/Um2K8X0JfR5XexRZp9rAprJXdUTCsWfj39OFHn5+kPM/SFgzAWYQ6vT6FUDh6DTVeLkEno/4quNtJi/xnY3ILtV18nAeu9Z4+YHnf1O9Jg998MuVBdMJneaQ7yUA7GMguN9yx7O8ZKMOnI/jBPvbY9xhPMxkeOlnvcQsCR0mc4OUEyTY5xhcc7zEQckXNI4NKU9xLx9gn2Owx/GqYCPGjX3m/09SbgfwnlFs9wTJe0NewjtnAm3vHM8eM5S6sL+y5I6tYBfQBO5GYr34B8Oz7x+Ie0ycBnqb8RLYlWL25MBxuIvs2TqGUle/iZN3mbwKP6SWiGVBvNXEx6JnuSVhJQe44HQeb/pfUU1h04A7ateytRLOxT/jg6kLC7H1MAEQLzVehc7Y4Omr6QdQPPx7qkcSVm3vQo9LuJgtUn/9yPkWPDK2Jn6dAIOta7fqveqoDLB+NBMkfcWmMl/CHmO0+CwtGJDnEl6sErMlJPW+ceCFrFX0S3mvg7q8t+aGgiR4XMC8GPfWmgtg3WoVqAPvPGB6jO/UmpvHXAlbbOwu8X0ix2oJ2/VWgRdYfN64IeExtd9c/gkWstGmzdjgLzPwAhox6O3XAAAAAElFTkSuQmCC>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABk0lEQVR4Xu2Vuy8FURDGPyIejY5INAqJKCQqDUHEX6Cj1ohSoVPR6CREJRKFQgTReIRGFCQaiVbcRtCQiEc8ghlzjsyOs+zei0L2l3zZne/bc+fsuWd3gYyM/80C6ZX0QBo2maeetA+5bstkvwo3rDT1o6qZDud7mk39QYs1CqQd0mhbeVfOa1Ae1wOqZvgm9oyHUlKOtEsqikZ5UQJpPq68O+f5Va12NR81m84PUkw6JJ2QKkxWKNxUNx4xtWcWYf8Ta6RrUo0NUsKreUR6QfTfWUF4IlMI+7HwHT2TmmyQgD7SNOmCtGqyHYQnMgHxa23wHWOQgfwA5MM5ZLxfxXlXWyYhPq98KgYhA3ttkJAhyPgzV8ftwRmE/VhGIQO6bPAFc5BtoWmD/I5v3urOUz3FGr6TJ1KjDRLgJ9KpvH7nnSqP6x5VMzekS+NFWIe8VKtskIJlyEOg4cY8oTLlbSC60rw/+Zo65b3DwQHpmFRusnxZgjTLueM9opPz8Hv3lrQIua47Ggv8qfuJL0hGRsZf8wZJgWKRIH3/fgAAAABJRU5ErkJggg==>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABcUlEQVR4Xu2UPS9EURCGx0dQSAQttqTnH2hUKpWEimj8BVGqNUIiUSuIBkGi8dFIFEIkYhsfiYQEhY+C8E5mzu7cce4W69rGPsmTPfOeOXc2d+9eoir/hRz88iF4giOwFbbAIfiY6PgFPDA2NOTWjkRHmWzDF0ofOgPnYL/bKxv+1mvwgdKHZk64aFZDB+Ey7PYbgRXYqetSQy/gGTyEH7A+0SHUwU84pvUxydmTQgdoh7umLjW00dQbmnk4W41k4z6wpA319JD0TZmMh/mzfT5bIDlsSRvKt81SS9J3bjKu/dl1n23CPWc4yOtF7bvUrElrplmzfZNxvWPqkN277Aexb3tF8v+1DJD0DZuM62lTh7sxabIosaH8ZOdd9g7fXMZ351nXDVS8Vk2hw8GP9h28Vnl9ZPZHSS5wq58HZs8yC1/hBEV+z0rAA298mCX84jg1dRvJ0F6TZQ4P4FvLdGk9X9z+G/hhW4JblHxhVKkc31ELcmN+nU1bAAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABhElEQVR4Xu2VzStEYRTGD0rZIJIVllZSPv4Ce2Vja4HGwt5K/gAbdpKytrGzQFmQjZBvZaHETimUbHycZ855x3Gcabo1LDS/epp5fu977zt37hdRhf9MLWeb88HZcGOWac4T54Uz6sYy0UOyWIP2fu2eC86m6WecXdMzgQWOA3dqer06D1yjl6VoIdlw0fkD9Ykj1xNwS16WYoRkwznnt9Qn8L3YopEHg5wVTqcf6KD4SG/UN2svtvPI13DeOGPaD0nmnBRmREIdgovKdk/k0VcDN27FgErcNmCK5CKCq1YX7Rx4j8X8vL7A5WkluT/POV2ca/o+0e884b3vYC1wIZj0bvqzOg/cpev2Xk7u3rnw16EPmT6szgPX6/qM6Tg9cJPG5YHEYy2xz3kwPYF5E6bPqrPscB71O66RdEBVhRlKtw7c6ieewRF1JON7JE+wVwp2xsyTHESOMpzPcoIF77wsJ1ckL4FEE/0852UHC+CvBe3aF76Gf4c2zjJnneS9W+Hv+QTRS351XQ0xmwAAAABJRU5ErkJggg==>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAYCAYAAAAGXva8AAABTUlEQVR4Xu2VMS9EQRSFLxKJRmS12JKe37GlP4BQ+AviB2ylI5GoFUQjgkQjGhJBhERLIVkNQjTCOZmZ3Zv77ha7s7axX3KSOWfmzdnMe/ueSI//Qhn6sWFkBXqDPqE5M5cFC73SO+hY+VvoTPm2OYI+pFg67GSE2YgNW2EM2oNepFhw5WSE2aYNWyFt6pU2O/JmOalA29CknUjsQONxnFs6AH1D89FfSlhzU18BRqET5XNL6XedbMEGmpxSltl1MzbbgKZ0IHml1pN9mx1Ap0bpQo7Tk/keMwuze+P1fzllNZMV8H7trJMRZtPGryrfH7Nllbl4pYTZkvLVmGl4Oq9xPCiNvfrqKwx8tJ+hxyiOL9T8kIQNzqFr6Ev8zdYkvJsXxbmf3YCFTzbsJA8SPgKJkhTvecdhAY+WTES/3pj+G/ga3YIOJXx3e3SfX2hxebjYDRHXAAAAAElFTkSuQmCC>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAXCAYAAAB50g0VAAABaElEQVR4Xu2UTytFQRjGX//KjWwUCzvdhfAF7JSFhbJSqLuhFGXtG1haKlnKShYoiXwCf7JiJ0kWLMhCFoTnNTPO9DTOved0pDS/+tU9z3NqpvPOHZFI5H9zAD/gI5ymTtFuE87CCpyEE3Dc+qvo4k32ty6sz/dJ/YVmP3nkvZeJEQ4C7MEtynbFLDzqZe9wAPbCMuy26nuZWYWvsIeLAG9iFhnzsj6b3dnneriU1N88wBYO0ziET7CDixS64Bplg2I2eEq5zwKc5zBEI7yA17BEXV72xWywnwuLnteqo20TM4ITMSMoigYxix9z4XEJpzh06Eie4Q4XBfEi6aNVUr+eHnw92CtcFMA53OCQWJcqG3S4L7nNRU70El6k7IaeFXf31UwrvBVzWdZRVyv6j5yjrBMuU6Zk3qBDD/cZvILN1KUxJMmi7LD3niP3Bn107HqJtnMRgDflG7odNNeJFcIMB5FI5A/5BNGqWJvPDKi0AAAAAElFTkSuQmCC>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABAUlEQVR4XmNgGAWDGXAD8S4g/g/Ep4GYEVWaKHAGXQAZSDNADOeE8oWhfCa4CtzgJANELQzjBF+BeCWaGMhVP9DE8IEiBgKWgCTD0MSqoOLEAryW2DFAJG3QxOOh4kJo4rgAXksKGCCSRmjioVBxczRxXACvJU0MEEk9NPFAqHg0mjgugNeSNAaIpAGaeAhU3BlNHBfAawksTizRxGOh4qDkTQzAawk7A0SSpqkLBECSk9DEtkHFkQEoMYijicEAQUuwuRrED0Lig4oZkBi6OhjoYoDIiaJLIIPlQPwXSoMUg5I2OtgAxCVoYr+A+AUQPwHix1D6NRAvRlY0CkYBbQAAvFZD8ftbiu8AAAAASUVORK5CYII=>