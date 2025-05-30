# **谷歌MedGemma：推动医疗人工智能发展的深度解析**

## **I. 引言：谷歌MedGemma与医疗人工智能的进步**

### **A. MedGemma概述及其重要性**

谷歌推出的MedGemma是一系列专为医学文本和图像理解而设计的开放模型，旨在加速医疗保健领域人工智能应用的开发进程 1。该模型被定位为谷歌在多模态医学文本与图像理解方面“能力最强的开放模型” 3。MedGemma于2025年5月20日左右发布 1，它的问世标志着谷歌在向更广泛的开发者社区提供专业化人工智能工具以赋能医疗健康创新方面迈出了战略性的一步。这种策略可能催生出超越谷歌内部研究范围的广泛创新。

### **B. MedGemma在谷歌人工智能战略中的定位**

MedGemma构建于Gemma 3架构之上，是谷歌轻量级、先进开放模型家族的一员 1。Gemma 3本身采用的技术与驱动Gemini模型的技术同源 7。MedGemma作为健康人工智能开发者基础（Health AI Developer Foundations, HAI-DEF）项目的一部分发布 3，延续了谷歌在医疗人工智能领域的持续投入，此前已有Med-PaLM、Med-PaLM2和Med-Gemini等项目 9。将MedGemma置于Gemma家族和HAI-DEF项目之下，凸显了谷歌致力于为特定高影响力行业提供开放、适应性强且负责任管理的AI工具的决心。

MedGemma并非一个即用型临床工具，而是被定位为一个“起点”或“基础” 1。这与人工智能领域基础模型的广泛趋势相符，这些模型需要针对特定任务进行大量的下游适应和验证。这意味着MedGemma提供的是一种强大但原始的能力，其真正的价值和安全性将高度依赖于开发者后续的适应工作的质量。

## **II. 技术架构与模型变体**

### **A.核心架构**

MedGemma基于Gemma 3架构 1，采用与Gemma 3相同的仅解码器（decoder-only）Transformer架构 1。其注意力机制为分组查询注意力（Grouped-Query Attention, GQA），并支持至少128K词元（tokens）的长上下文 5。模型创建于2025年5月20日，版本号为1.0.0 1。利用成熟的Gemma 3架构为MedGemma提供了坚实且经过测试的基础，使其能够受益于通用架构的进步，同时通过数据和微调实现专业化。长上下文长度对于处理冗长的医疗文档或患者历史记录尤为有利。

### **B. MedGemma 4B：多模态能力（文本与视觉）**

MedGemma 4B拥有40亿参数 1，能够接受文本和图像作为输入 1。其图像编码器采用SigLIP模型 1，该编码器专门使用多种经过身份匿名的医疗数据进行预训练，包括胸部X射线、皮肤病学图像、眼科学图像和组织病理学幻灯片 1。其大语言模型（LLM）组件则利用包括放射学图像、组织病理学图块、眼科学图像、皮肤病学图像及相关医学文本在内的多样化医疗数据进行训练 1。

MedGemma 4B提供两个版本：

* 预训练版（后缀：-pt），资源ID为google/medgemma-4b-pt，供希望进行更深度实验的用户使用 1。  
* 指令调优版（后缀：-it），资源ID为google/medgemma-4b-it，被推荐为大多数应用的更佳起点 1。

MedGemma 4B是谷歌为需要综合理解视觉和文本医疗数据的任务提供的解决方案。其图像编码器在医学影像上的专门预训练是其关键的差异化特征。

### **C. MedGemma 27B：纯文本专业化**

MedGemma 27B拥有270亿参数 1，是一个纯文本模型 1。它专门使用医学文本进行训练，并针对推理时计算进行了优化 1。该模型仅提供指令调优版，资源ID为google/medgemma-27b-text-it 1。谷歌指出，对于大多数需要医学知识的用例，这个较大的纯文本版本“通常会产生最佳性能” 11。MedGemma 27B专为需要对复杂医学文本进行深度理解和推理，同时关注推理计算效率的应用而设计。其更大的规模和纯医学文本训练表明其专为高风险文本任务而生。

提供MedGemma 4B（多模态）和MedGemma 27B（纯文本）两种不同的变体，反映了谷歌为满足医疗AI应用多样化的计算和功能需求的战略决策。某些任务固有地需要图像分析（如放射学、病理学），而另一些任务则是纯文本的，但要求深度的语义理解（如临床笔记摘要、医学问答）。这种双变体策略允许开发者选择最合适的工具，以优化能力和资源消耗。MedGemma 27B针对“推理时计算”的优化进一步表明其对文本型任务实际部署的关注。

专门提及MedGemma 4B中的SigLIP图像编码器“专门使用多种经过身份匿名的医疗数据进行预训练” 1 是一个关键的技术细节。通用图像编码器通常难以处理医学图像的细微差别。医学图像（如X射线、病理学幻灯片）具有独特的特征、模式和伪影，这些在通用图像数据集中并不常见。在通用图像上预训练的编码器可能无法提取与医学诊断或解释最相关的特征。而专门针对多种医学模态（胸部X射线、皮肤病学、眼科学、组织病理学）预训练SigLIP 1，使其能够识别医学相关的视觉特征。这种专业化的预训练很可能是MedGemma 4B在医学影像基准测试中表现优于配备通用图像编码器的Gemma 3 4B模型的主要原因。

下表总结了MedGemma模型变体的主要特性与规格：

**表1：MedGemma模型变体 – 关键特性与规格**

| 特性 | MedGemma 4B | MedGemma 27B |
| :---- | :---- | :---- |
| 参数量 | 40亿 | 270亿 |
| 模态 | 文本、视觉 | 纯文本 |
| 基础架构 | Gemma 3 (仅解码器Transformer) | Gemma 3 (仅解码器Transformer) |
| 图像编码器 | SigLIP (使用去识别化医疗数据预训练：胸部X光、皮肤科、眼科、组织病理学图像) | 不适用 |
| LLM组件训练重点 | 多样化医疗数据（放射学图像、组织病理学图块、眼科图像、皮肤科图像及相关医学文本） | 纯医学文本 |
| 关键优化 | 多模态理解 | 推理时计算效率 |
| 可用版本 (资源ID) | google/medgemma-4b-pt (预训练版)\<br\>google/medgemma-4b-it (指令调优版) | google/medgemma-27b-text-it (指令调优版) |

### **D. 技术创新与前景**

MedGemma的发布代表了谷歌在医疗AI领域推动开放模型发展的重要一步。通过提供基于先进Gemma 3架构并针对医疗数据进行优化的模型，谷歌旨在降低开发者构建复杂医疗AI应用的门槛。特别是MedGemma 4B中的医学专用SigLIP图像编码器，以及MedGemma 27B对医学文本的深度优化，是其核心技术亮点。这些模型有望催生更多创新的医疗应用，从辅助诊断到个性化治疗支持。然而，正如后续章节将详细讨论的，这些模型的“基础”特性要求开发者在实际应用中承担起验证、适配和确保伦理合规的重任。

## **III. 训练数据与评估方法**

### **A. 数据集概述**

Gemma基础模型预训练于大规模文本和代码语料库 5。MedGemma变体则进一步使用公开和私有（经过身份匿名和许可/内部收集）的医疗数据集进行训练/微调 1。在预训练特定组件（如SigLIP图像编码器）时，重点使用了经过身份匿名的医疗数据 1。对多样化和专业化医疗数据集的依赖，对于MedGemma的领域特定能力至关重要。在训练阶段使用身份匿名数据也体现了对隐私的考量。

明确提及“身份匿名的医疗数据” 1 和“专有数据集（身份匿名和许可/内部收集）” 5 具有高度重要性。虽然公共数据集很有价值，但高质量、多样化且经过良好整理的医疗数据，特别是针对专业影像的数据，通常是私有的。训练强大的医疗AI需要大量高质量、领域特定的数据。公共医疗数据集虽然有用，但在范围、多样性或注释质量方面可能存在局限。获取专有的、身份匿名的数据集使谷歌能够利用比单独使用公共数据更广泛的医疗状况、影像模态和患者人群来训练模型。这种数据获取途径是一项竞争优势，对于实现最先进的性能至关重要。然而，这也意味着训练数据中一部分的完整性质和潜在偏见无法公开审查，从而强化了开发者进行自身彻底验证的必要性 5。

### **B. MedGemma 4B的训练数据**

* **SigLIP图像编码器**：专门使用身份匿名的医疗数据进行预训练，包括胸部X射线、皮肤病学图像、眼科学图像和组织病理学幻灯片 1。  
  * 提及的公共数据集包括：MIMIC-CXR、PAD-UFES-20、SCIN、TCGA、CAMELYON、PMC-OA、Mendeley Digital Knee X-Ray 5。  
  * 专有（身份匿名/许可/内部）数据集包括：放射学（CT）、眼科学（眼底图像）、皮肤病学（远程皮肤病学、皮肤癌、非病变皮肤）、病理学（来自不同来源和组织类型的H\&E、IHC幻灯片）5。  
* **LLM组件**：使用多样化的医疗数据进行训练，包括放射学图像、组织病理学图块、眼科学图像、皮肤病学图像以及与这些图像类型相关的医学文本 1。 用于MedGemma 4B的医学影像和相关文本数据的广度和深度，凸显了其为创建真正的多模态医学AI基础所做的努力。纳入各种影像模态以及公共和专有数据源旨在实现模型的稳健性。

### **C. MedGemma 27B的训练数据**

MedGemma 27B专门使用医学文本进行训练 1。这种专注的训练使27B模型能够深入理解医学术语、概念以及临床文本数据中存在的推理模式。

### **D. 评估方法**

MedGemma在一系列临床相关的综合基准上进行了评估 1。评估覆盖超过15个数据集，涉及3项任务和4种医学影像模态 1，或更广泛地说，超过22个数据集，涉及5项任务和6种医学影像模态 5。评估既包括开放基准数据集，也包括策划数据集 1。在评估如胸部X光报告生成和放射学视觉问答（VQA）等任务时，重点关注了专家人工评估 5。这种多方面的评估方法，包括专家人工评估，旨在提供对MedGemma基线性能的整体看法，并建立对其超越自动化指标能力的信心。

MedGemma模型卡 5 明确警告了“数据污染”的风险，即模型在广泛的预训练过程中可能无意中接触到与基准数据集相关的信息，从而可能高估其性能得分。大型语言和多模态模型是在海量的网络规模语料库和专业数据集上训练的。基准数据集，即使没有直接包含在训练中，其组成部分（例如，讨论这些数据集的论文、相关概念）也可能存在于更广泛的预训练数据中。这种重叠可能导致模型在基准测试中表现良好，并非因为它具有良好的泛化能力，而是因为它“记住”或见过了类似的示例。谷歌明确承认这一点（例如，5中提到“开发者应在非公开可用的数据集上验证MedGemma……以减轻这种风险”）是负责任披露的体现。这突显了评估大型基础模型时的一个根本性挑战，并强调了开发者使用自己的私有、未见过的数据集进行独立验证的至关重要性。

下表概述了MedGemma训练数据的类别：

**表2：MedGemma训练数据类别概述**

| 数据类型/模态 | 具体示例/子类型 | 数据来源类型 (公开/专有去识别化) | 提及的关键公共数据集 (如适用) | 与MedGemma变体的相关性 (4B, 27B, 或两者) |
| :---- | :---- | :---- | :---- | :---- |
| 胸部X射线 (CXR) | 图像、报告 | 公开、专有去识别化 | MIMIC-CXR | 4B |
| 皮肤病学图像 | 临床图像、皮肤镜图像 (皮肤病变、皮肤癌、非病变皮肤) | 公开、专有去识别化 | PAD-UFES-20, SCIN | 4B |
| 眼科学图像 | 眼底图像 (糖尿病视网膜病变筛查) | 公开、专有去识别化 |  | 4B |
| 组织病理学幻灯片 | H\&E染色、IHC染色 (结肠、前列腺、淋巴结、肺等多种组织) | 公开、专有去识别化 | TCGA, CAMELYON | 4B |
| 放射学图像 (其他) | CT研究 (跨身体部位) | 专有去识别化 |  | 4B |
| 生物医学文献与图像 | 包含图像的文献 | 公开 | PMC-OA | 4B |
| 膝关节X射线 | 膝关节X射线图像 | 公开 | Mendeley Digital Knee X-Ray | 4B |
| 通用医学文本 | 与上述图像模态相关的医学文本、临床笔记、医学知识 | 公开、专有去识别化 | (多种来源，未具体列出所有文本数据集) | 4B, 27B |
| 放射学特定文本 | 与放射学图像和报告相关的文本 | 公开、专有去识别化 | (源自MIMIC-CXR等) | 4B |

## **IV. 性能基准与对比分析**

### **A. 影像评估结果 (MedGemma 4B vs. Gemma 3 4B)**

根据官方发布的数据 1，MedGemma 4B在各项医学影像基准测试中，相较于通用的Gemma 3 4B模型，展现出持续且显著的性能优势。

* **医学图像分类**:  
  * 在MIMIC CXR数据集上（针对前5种常见病症的平均F1分数），MedGemma 4B达到88.9，而Gemma 3 4B为81.1。  
  * 在CheXpert CXR数据集上（针对前5种常见病症的平均F1分数），MedGemma 4B为48.1，Gemma 3 4B为31.2。  
  * 在DermMCQA数据集上（准确率），MedGemma 4B为71.8，Gemma 3 4B为42.6。  
* **视觉问答 (VQA)**:  
  * 在SlakeVQA（放射学，词元化F1分数）上，MedGemma 4B为62.3，Gemma 3 4B为38.6。  
  * 在VQA-Rad（放射学，词元化F1分数）上，MedGemma 4B为49.9，Gemma 3 4B为38.6。  
  * 在PathMCQA（组织病理学，内部数据集，准确率）上，MedGemma 4B为69.8，Gemma 3 4B为37.1。  
* **知识与推理 (多模态)**:  
  * 在MedXpertQA（文本+多模态问题，准确率）上，MedGemma 4B为18.8，Gemma 3 4B为16.4。

这些结果有力地证明了MedGemma 4B针对图像编码器和LLM组件进行的专业化医学预训练所带来的益处。然而，诸如在CheXpert CXR上48.1的得分表明，要达到临床级性能仍有提升空间，这也印证了其作为基础模型的定位。

### **B. 文本评估结果 (MedGemma 27B & 4B vs. Gemma 3 27B & 4B)**

在基于文本的医学问答基准测试中，MedGemma模型（包括27B和4B版本）同样优于其对应的通用Gemma 3模型 1。

* **MedQA (4选项)**:  
  * MedGemma 27B：89.8 (best-of-5), 87.7 (0-shot)，而Gemma 3 27B为74.9。  
  * MedGemma 4B：64.4，而Gemma 3 4B为50.7。  
* **MedMCQA**:  
  * MedGemma 27B：74.2，而Gemma 3 27B为62.6。  
  * MedGemma 4B：55.7，而Gemma 3 4B为45.4。  
* **PubMedQA**:  
  * MedGemma 27B：76.8，而Gemma 3 27B为73.4。  
  * MedGemma 4B：73.4，而Gemma 3 4B为68.4。  
* **MMLU Med (纯文本)**:  
  * MedGemma 27B：87.0，而Gemma 3 27B为83.3。  
  * MedGemma 4B：70.0，而Gemma 3 4B为67.2。  
* **MedXpertQA (纯文本)**:  
  * MedGemma 27B：26.7，而Gemma 3 27B为15.7。  
  * MedGemma 4B：14.2，而Gemma 3 4B为11.6。  
* **AfriMed-QA**:  
  * MedGemma 27B：84.0，而Gemma 3 27B为72.0。  
  * MedGemma 4B：52.0，而Gemma 3 4B为48.0。

值得注意的是，MedGemma 27B的测试结果使用了测试时缩放（test-time scaling）以提升性能 5。有报道称，MedGemma 27B在MedQA上的表现可与GPT-4o等更大规模的模型相媲美，尽管在某些指标上其他一些模型表现更优 6。另有资料指出，MedGemma在临床知识和推理方面的基线性能与远大于其规模的模型相似 9。MedGemma 27B凭借其更大的规模和纯医学文本训练，通常在这些文本任务中取得MedGemma家族中的最高分。

关于MedGemma（特别是MedGemma 27B在MedQA上的表现）性能“可与大得多模型相媲美”或“相似”的说法 6，虽然对于其规模而言令人印象深刻，但需要审慎解读。模型规模通常与能力相关，但也与计算成本相关。用更小、更高效的模型实现与更大模型相似的性能是一项重大成就。然而，“相似”和“可比较”可能具有主观性。实现这种相似性的具体基准和指标非常重要（MedQA被引用）。同时，6也指出其他大型模型（如DeepSeek R1, Gemini 2.5 Pro, GPT-4o）在“某些指标”上优于MedGemma 27B。这表明，虽然MedGemma在其规模下能力很强，但它可能并非在所有任务上普遍达到或超过最大、最先进专有模型的性能。开发者需要考虑具体任务和所需的性能水平。MedGemma的“开放”性质和适应性在某些情况下可能是为了换取前沿性能而做出的权衡。

### **C. 胸部X射线报告生成性能**

在MIMIC CXR数据集上，使用RadGraph F1分数进行评估 5：

* MedGemma 4B (预训练版)：29.5。  
* PaliGemma 2 3B (针对CXR调优)：28.8。  
* PaliGemma 2 10B (针对CXR调优)：29.5。

MedGemma 4B的指令调优版 (-it) 和Gemma 3 4B的指令调优版 (-it) 在此指标上得分较低（分别为0.22和0.12），这归因于其报告风格与MIMIC基准报告的差异 5。预训练的MedGemma 4B在CXR报告生成方面表现出与专门的PaliGemma模型相当的竞争力。指令调优版本得分较低，凸显了生成模型评估中的一个常见挑战：指令调优可能会改变输出风格，即使内容临床相关，也可能与严格的基准指标不完全一致。这强调了在单一指标之外进行细致评估的必要性。

指令调优版MedGemma 4B在胸部X射线报告生成任务的RadGraph F1分数远低于其预训练版本 5，这是一个关键的细微之处。虽然指令调优通常使模型更易用且更善于遵循指令（并被推荐用于大多数应用 1），但它也可能导致模型偏离某些基准数据集（如MIMIC-CXR报告）的特定风格约定。预训练模型基于从原始数据中学习到的模式生成文本。指令调优模型则进一步训练以遵循人类指令，并通常生成更具对话性或结构化的输出。MIMIC-CXR基准的真实报告可能具有特定、简洁的报告风格。指令调优可能使MedGemma 4B生成的报告在事实上正确，但在风格上（例如，更冗长、结构不同）与MIMIC的真实报告不同。像RadGraph F1这样的指标对这种风格差异很敏感。这意味着，对于需要遵循非常特定的输出格式的任务，开发者可能需要试验预训练版本，或者对指令调优版本使用期望风格的示例进行非常有针对性的微调。这也凸显了在生成任务中仅依赖自动化指标的局限性。

下表总结了MedGemma在关键医学基准测试中的性能：

**表3：MedGemma在关键医学基准测试中的性能总结**

| 基准测试名称 | 任务类型 | 指标 | MedGemma 4B 得分 | MedGemma 27B 得分 | Gemma 3 4B 对比得分 | Gemma 3 27B 对比得分 |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| MIMIC CXR | 图像分类 | 平均F1 (前5病症) | 88.9 | 不适用 | 81.1 | 不适用 |
| CheXpert CXR | 图像分类 | 平均F1 (前5病症) | 48.1 | 不适用 | 31.2 | 不适用 |
| DermMCQA | 图像分类 | 准确率 | 71.8 | 不适用 | 42.6 | 不适用 |
| SlakeVQA (放射学) | 视觉问答 (VQA) | 词元化F1 | 62.3 | 不适用 | 38.6 | 不适用 |
| VQA-Rad (放射学) | 视觉问答 (VQA) | 词元化F1 | 49.9 | 不适用 | 38.6 | 不适用 |
| PathMCQA (组织病理学) | 视觉问答 (VQA) | 准确率 | 69.8 | 不适用 | 37.1 | 不适用 |
| MedQA (4选项) | 文本问答 (QA) | 准确率 | 64.4 | 89.8 (best-of-5) | 50.7 | 74.9 |
| MedMCQA | 文本问答 (QA) | 准确率 | 55.7 | 74.2 | 45.4 | 62.6 |
| PubMedQA | 文本问答 (QA) | 准确率 | 73.4 | 76.8 | 68.4 | 73.4 |
| MMLU Med (纯文本) | 文本问答 (QA) | 准确率 | 70.0 | 87.0 | 67.2 | 83.3 |
| MedXpertQA (纯文本) | 文本问答 (QA) | 准确率 | 14.2 | 26.7 | 11.6 | 15.7 |
| AfriMed-QA | 文本问答 (QA) | 准确率 | 52.0 | 84.0 | 48.0 | 72.0 |
| MIMIC CXR (报告生成) | 图像到文本 | RadGraph F1 | 29.5 (预训练版) | 不适用 | 0.12 (指令调优版) | 不适用 |

## **V. 预期用例与目标应用**

### **A. 总体定位**

MedGemma模型被设计为开发者构建和调整基于医疗保健的人工智能应用的起点 1。它们需要在开发者预期的用例上进行验证，并且可能需要进一步调整以提高性能 11。

谷歌一贯且强烈地强调MedGemma是一个“开发者模型” 11，它“需要在开发者预期的用例上进行验证” 5，这是一种深思熟虑的策略。它旨在管理关于开箱即用性能的期望，并且至关重要的是，将临床有效性和安全性的责任（以及相关法律责任）转移给那些基于MedGemma构建应用的开发者。医疗保健领域的人工智能是一个高风险领域，存在显著的监管和安全问题。直接发布用于临床的模型将使谷歌面临巨大的审查和监管负担（例如，作为医疗设备制造商）。通过将MedGemma定位为面向开发者的基础工具，谷歌在提供强大功能的同时，在法律和伦理上与最终的临床应用保持距离。“预期用途声明” 5 明确禁止直接将原始模型输出用于临床，并强制要求开发者进行验证和调整。这种方法使谷歌能够在生态系统中促进创新，而无需承担在临床环境中部署未经证实的人工智能的直接风险。它还为开发者社区设定了关于构建安全有效应用仍需完成工作的明确期望。

### **B. MedGemma 4B (多模态) 用例**

* **医学图像分类**：跨专业对医学图像进行分类，如放射学（例如胸部X射线）、数字病理学、眼底图像和皮肤图像（皮肤病学）8。  
* **医学图像解读**：生成医学图像报告，回答关于医学图像的自然语言问题（VQA）11。  
* 分析放射学图像或总结临床数据 6。

### **C. MedGemma 27B (纯文本) 用例**

* **医学文本理解与临床推理**：适用于需要深度医学知识的应用 8。  
* 具体任务：患者访谈、分诊、临床决策支持、医学文本/笔记摘要 11。  
* 谷歌指出，对于大多数基于文本的用例，27B模型通常会产生最佳性能 11。

虽然MedGemma提供了广泛的潜在用例 5，但这种多功能性意味着对于任何特定的、高性能的临床应用，都需要大量的专业化工作（微调、提示工程、与其他工具集成）。开发者必须在利用MedGemma广泛的基线能力与为特定细分任务可能使用/开发更专注的模型之间做出选择。MedGemma 4B接受多种图像模态和文本训练；MedGemma 27B接受广泛的医学文本训练 5。这赋予了它们一般的医学理解能力。列出的用例多种多样：放射学、病理学、皮肤病学的图像分类；报告生成；VQA；患者分诊；临床决策支持 5。然而，例如，在特定类型的病理学幻灯片中检测特定罕见疾病以达到专家级性能，可能需要的远不止基线MedGemma。开发者将需要投入资源来整理用于微调的特定数据集，并对该狭窄任务进行严格验证。这意味着对于许多复杂的临床问题，MedGemma更多的是一个强大的起点加速器，而非最终解决方案。

### **D. 不当使用 (明确说明)**

* 未经开发者针对其特定用例进行适当验证、调整和/或有意义的修改，不得使用 5。  
* MedGemma的输出**不**旨在直接为临床诊断、患者管理决策、治疗建议或任何其他直接临床实践应用提供信息 5。这是一个关键的免责声明。  
* 性能基准显示的是基线能力；即使在训练数据中占比较大的领域，也可能出现不准确的输出。所有MedGemma的输出都需要独立验证、临床关联和通过既定研发方法进一步调查 5。

## **VI. 开发者访问、部署与定制**

### **A. 可用性与访问**

MedGemma可通过Hugging Face访问 6，但需同意健康人工智能开发者基础（HAI-DEF）的使用条款 16。它也存在于谷歌云Vertex AI模型花园中 1。特定版本的资源ID（例如 google/medgemma-4b-it, google/medgemma-4b-pt, google/medgemma-27b-text-it）均已提供 1。在流行的开源平台（Hugging Face）和商业云（Vertex AI）上的双重可用性，满足了不同开发者对实验和可扩展部署的偏好。

要求在Hugging Face上访问MedGemma文件前接受“健康人工智能开发者基础使用条款” 16，这是一个至关重要的控制机制。即使对于一个“开放”模型，这一步骤也确保开发者在下载模型权重之前，至少了解了在医疗等敏感领域使用人工智能相关的具体责任和限制。Hugging Face是开源AI模型的主要中心，通常采用非常宽松的许可证。医疗AI比通用AI具有更高的风险和伦理考量。通过将访问权限置于同意特定使用条款之后，谷歌实施了一个检查点。这确保开发者承认参与规则（例如，HAI-DEF条款中概述的禁止用途、验证要求 18）。即使通过公共平台分发模型，这也是一种实施一定程度治理的实用方法。

### **B. 部署选项与要求**

MedGemma可在本地运行以供实验 6，也可通过谷歌云Vertex AI部署为可扩展的HTTPS端点用于生产环境 1。通过Vertex AI部署需要特定的角色：Vertex AI用户 (roles/aiplatform.user) 和Colab Enterprise用户 (roles/aiplatform.colabEnterpriseUser) 1。运行27B模型而不进行量化可能需要Colab Enterprise 5。HyperAI也为27B模型提供“一键部署”方案 20。谷歌为本地小规模使用和稳健的云端生产部署都提供了途径，反映了模型从研发到潜在应用的过程。

### **C. 调整与定制MedGemma**

开发者被期望对模型进行调整 11。

* **提示工程/情境学习**：对于某些用例，仔细设计提示（可能在提示中包含少量理想响应示例，即情境学习）可能就足够了。提示工程也可用于将任务分解为可单独执行的子任务 14。这种调整需要与其他类型的调整同等级别的验证 14。  
* **微调**：可以对MedGemma进行微调，以提高其在已训练任务上的性能，或为其增加新的任务能力 1。提供了使用LoRA（一种参数高效的微调技术）进行微调的示例笔记本 14。用户可以专门微调语言模型解码器组件，以帮助模型更好地解释图像编码器产生的视觉标记，或者同时微调两者 14。  
* **智能体编排 (Agentic Orchestration)**：MedGemma可作为智能体系统中的一个工具，与其他工具（如网络搜索、FHIR生成器/解释器、用于双向音频对话的Gemini Live，或用于函数调用/推理的Gemini 2.5 Pro）结合使用 6。它还可以用于在本地解析私有健康数据，然后再向Gemini 2.5 Pro等中心化模型发送匿名请求 14。

对各种调整方法的明确指导（提示、使用LoRA进行微调、智能体系统）赋予了开发者灵活性。建议使用MedGemma在与大型云模型交互前本地解析私有数据，这是一种值得注意的隐私保护模式。具体提及用于微调的LoRA 5 和“智能体编排”的概念 5 不仅仅是技术建议；它们表明谷歌将MedGemma视为复杂AI工作流程中高度适应性组件的愿景。对大型模型进行完全微调在计算上是昂贵且数据密集的。LoRA（低秩适应）提供了一种参数高效的模型调整方法，使得微调更易于实现。建议使用LoRA意味着谷歌希望降低开发者为特定医疗细分领域定制MedGemma的门槛。“智能体编排”是指将LLM用作大型系统中的推理引擎或组件，该系统可能涉及其他工具（数据库、网络搜索、其他API、其他模型如Gemini 2.5 Pro）。这将MedGemma定位为一个专门的“大脑”或“工具”，可以与其他功能结合以解决复杂问题，而非一个单一的解决方案。例如，MedGemma可以分析图像，然后一个智能体可以基于该分析查询FHIR数据库，之后再次使用MedGemma综合生成报告。使用MedGemma在向云模型发送匿名查询前本地解析私有健康数据 14 是这种具有隐私优势的编排的一个实际例子。

## **VII. 负责任的人工智能：伦理考量、局限性与治理**

### **A. 预期用途声明 (IUS) 及适当/不当使用**

根据MedGemma模型卡 5 和相关文档 14，其预期用途和限制如下：

* **适当用途**：MedGemma是一个开放的多模态生成式AI模型，旨在作为生命科学和医疗保健领域开发者构建下游医疗应用的*起点*。它利用其在各种医疗数据上的预训练，支持在任何医疗环境（图像和文本）中进行进一步开发。与其规模相似的模型相比，其强大的基线性能使其能够高效地适应下游医疗保健用例。预期通过提示工程、基础调整、智能体编排或微调等方式进行适配。  
* **不当用途**：  
  * **严禁**在未经开发者针对其特定用例进行适当验证、适配和/或有意义修改的情况下使用。  
  * 模型的输出**不得**直接用于临床诊断、患者管理决策、治疗建议或任何其他直接的临床实践应用。  
  * 性能基准仅代表基线能力；即使在训练数据中占比较大的领域，也可能出现不准确的输出，所有输出均需独立验证和临床关联。

预期用途声明是谷歌针对MedGemma的负责任AI策略的核心，它清晰地界定了界限和责任，并强烈强调MedGemma是开发者的工具，而非临床产品。

### **B. 偏见及其缓解**

MedGemma的训练数据包含多样化的医疗数据 1。然而，MedGemma模型卡 5 强调，开发者应确保下游应用使用能够适当代表预期使用场景（例如，年龄、性别、病情、影像设备等）的数据进行验证，以解决偏见问题。由于使用了专有数据集 5，所有训练数据特征的完全透明性无法实现，这使得开发者进行独立的偏见评估更为关键。通用AI偏见缓解策略（如27中所述，虽非专门针对MedGemma，但提供了背景信息）包括组建多元化团队，并从概念阶段就考虑对不同人群可能产生的非预期后果。尽管谷歌在整理多样化训练数据方面已采取措施，但在使用MedGemma构建的特定应用中，检测和缓解偏见的责任主要落在开发者身上，特别是考虑到部分专有训练集的不透明性。

### **C. 训练和使用中的数据隐私与安全**

MedGemma的图像编码器（SigLIP）和LLM组件均使用“经过身份匿名处理的”医疗数据进行预训练 1。谷歌声明其合作伙伴使用了经过严格匿名化或去身份化处理的数据集，以保护个体研究参与者和患者的隐私 5。健康人工智能开发者基础使用条款 18 要求开发者遵守适用法律，包括隐私和数据保护权利，并就违规行为对谷歌进行赔偿。MedGemma可用于在向中心化模型发送匿名请求之前，在本地解析私有健康数据，从而增强隐私保护 14。在设备上/离线运行模型（Gemma家族的普遍趋势，例如Gemma 3n 21）可以通过减少云端数据传输来增强隐私，尽管MedGemma的较大变体可能因其规模而难以完全在设备上部署。谷歌在其训练流程中强调了去身份化。然而，对于开发者使用MedGemma构建的应用程序，确保数据隐私和安全（尤其是在涉及受保护健康信息PHI时）是开发者的责任，并受HIPAA等法律以及所用任何平台服务条款（例如谷歌云BAA）的约束。

MedGemma在“去身份识别”数据上的训练 1 是模型开发的一项负责任实践。然而，这并*不*意味着用MedGemma构建的应用程序可以在没有进一步保障措施的情况下自动处理PHI。在模型创建阶段使用去身份识别数据可以最大限度地降低隐私风险。一旦MedGemma发布，开发者可能希望将其用于确实处理PHI的应用程序中（例如，分析患者特定的X射线和报告）。使用模型处理PHI的行为会使HIPAA（及类似法规）对开发者的应用程序生效。HAI-DEF条款 18 使开发者对法律合规性负责，包括数据保护。如果使用谷歌云服务（如Vertex AI）托管处理PHI的基于MedGemma的应用程序，开发者将需要与谷歌云就这些平台服务签订业务伙伴协议（BAA） 22。BAA涵盖平台的责任，而非MedGemma本身。这造成了双重责任：谷歌确保其基础模型训练中的去身份识别；开发者确保其使用该模型构建的特定应用程序中PHI的合规处理。

### **D. 安全性评估与局限性**

* **评估方法**：针对内容政策（儿童安全、内容安全、代表性伤害、一般医疗伤害）进行了结构化评估和内部红队测试 5。独立于开发团队的“保证评估”为发布决策提供信息 5。  
* **评估结果**：在儿童安全、内容安全和代表性伤害方面观察到安全的性能水平，策略违规极少（在无安全过滤器的情况下测试）。测试主要使用英语提示 5。  
* **已知局限性**：  
  * 主要在单图像任务上进行评估；未在多图像理解或多轮应用中进行评估 5。  
  * 可能比Gemma 3对提示更敏感 5。  
  * **数据污染风险**：模型在预训练期间可能无意中接触到相关的医学信息，从而可能高估其泛化能力。开发者应在非公开数据集上进行验证 5。  
  * 非临床级别；医学图像解读可能需要额外的微调 14。

谷歌的内部安全测试是一个积极的步骤，但已承认的局限性（单图像、单轮、提示敏感性、数据污染）对于开发者理解至关重要。这些局限性再次强调了为何MedGemma是一个基础模型，需要仔细的、针对特定情境的验证。模型卡 5 指出，MedGemma的安全性测试主要使用英语提示。考虑到人工智能的全球化愿景以及全球医疗保健领域语言环境的多样性，这是一个显著的局限性。人工智能的安全性和偏见在不同语言和文化中可能表现不同。医学概念、患者沟通方式和潜在危害可能有所不同。如果安全评估主要基于英语，那么模型在非英语环境中的行为和安全状况就知之甚少。这意味着旨在将MedGemma用于非英语应用的开发者，需要在其目标语言和文化背景下进行特别彻底的安全测试和验证。这也指出了谷歌自身在全球化模型评估流程中未来需要改进的一个领域。

### **E. 治理：健康人工智能开发者基础 (HAI-DEF) 使用条款**

访问MedGemma（例如在Hugging Face上）需要同意HAI-DEF条款 16。虽然完整的许可证文本在提供的材料中不可用 12，但使用条款 18 是关键。

* **关键禁止条款** 18：  
  * 如果任何“临床使用”（患者的诊断或治疗，包括作为研究的一部分）可能导致卫生监管机构将谷歌视为医疗设备的“制造商”，则禁止此类使用。  
  * 禁止任何违反适用法律的使用。  
  * （单独的）HAI-DEF禁止使用政策中描述的限制性用途。  
* **开发者分发责任** 18：必须在使用协议中包含使用限制，向接收者提供HAI-DEF使用条款，在适用时寻求监管授权，并注明修改。  
* **免责声明与责任限制**：模型按“原样”提供。谷歌不对损害承担责任。开发者承担所有风险 18。  
* **赔偿**：开发者因违规或因其使用/修改造成的损害而对谷歌进行赔偿 18。

HAI-DEF使用条款是主要的治理工具。对可能将谷歌归类为医疗设备制造商的“临床使用”的限制是一条关键的法律界限。这些条款坚决地将合规、安全和有效的应用开发的责任置于最终用户开发者身上。HAI-DEF使用条款中 18 关于禁止将HAI-DEF服务（包括MedGemma）用于任何可能导致卫生监管机构将谷歌视为医疗设备“制造商”的“临床用途”的规定，是一项非常重要的法律和监管条款。医疗设备受到严格监管（例如美国的FDA，欧洲的EMA）。如果软件旨在用于诊断或治疗等医疗目的，则可被视为医疗设备（医疗器械软件 \- SaMD）。成为医疗设备的“制造商”需要承担严格的义务：质量管理体系、上市前批准/许可、上市后监督、法律责任等。通过明确禁止会导致谷歌承担此类分类的用途，谷歌在法律上保护自己免受第三方开发者使用MedGemma作为组件创建的应用程序所带来的这些广泛的监管负担和相关法律责任。该条款有效地将任何临床AI应用程序寻求监管批准的责任推给了使用MedGemma作为组件创建该特定应用程序的开发者。

### **F. HIPAA合规性背景**

MedGemma模型本身并非“符合HIPAA标准”。HIPAA是针对受保护健康信息（PHI）处理的监管框架，适用于受保实体及其业务伙伴 22。谷歌云平台为某些服务（如Vertex AI、Google Workspace Enterprise）提供业务伙伴协议（BAA），这可以成为开发者符合HIPAA标准的架构的一部分 22。如果开发者使用MedGemma（例如部署在Vertex AI上）处理PHI，他们有责任确保其整个解决方案架构、数据处理政策和协议（如与谷歌云签订的BAA）符合HIPAA要求。谷歌工具的消费者版本（如免费的Gemini/Bard）不符合HIPAA标准，不应用于处理PHI 22。HAI-DEF条款 18 规定开发者有义务遵守所有适用法律，如果涉及PHI，则包括HIPAA。区分MedGemma模型本身和它可能运行的平台至关重要。虽然MedGemma是在去身份识别数据上训练的，但任何使用它处理PHI的应用程序都必须在符合HIPAA标准的框架内构建，开发者对此负责。谷歌提供了平台工具（如带有BAA的Vertex AI），这些工具可以成为此类框架的一部分。

下表总结了MedGemma的关键伦理考量和开发者责任：

**表4：MedGemma的关键伦理考量和开发者责任**

| 考量领域 | 谷歌的关键指南/警告 | 开发者义务/责任 | 相关资料来源 |
| :---- | :---- | :---- | :---- |
| 临床使用限制 | 输出不得直接用于临床诊断、治疗或管理；禁止可能导致谷歌被视为医疗器械制造商的临床用途。 | 严格遵守非临床用途；为任何临床级应用寻求独立的监管路径和验证。 | 5 |
| 性能验证 | 基准性能仅为基线；可能产生不准确输出；需要独立验证和临床关联。 | 针对特定用例，使用代表性数据进行彻底验证；不得依赖开箱即用的性能。 | 5 |
| 偏见缓解 | 开发者应确保下游应用使用代表预期使用环境的数据进行验证（年龄、性别等）。 | 主动评估和减轻应用中的偏见；考虑到专有训练数据的不透明性。 | 5 |
| 数据隐私 (PHI处理) | 训练数据经过身份匿名处理；开发者需遵守适用法律（包括数据保护）。 | 如果处理PHI，需确保整个应用架构符合HIPAA等法规，包括与平台提供商签订BAA（如适用）。 | 5 |
| 输出安全性 | 已进行内部安全测试，但主要针对英语提示；模型可能对提示敏感。 | 针对特定用例和语言环境进行独立的安全性和稳健性测试；仔细设计提示。 | 5 |
| 遵守使用条款 | 必须遵守健康人工智能开发者基础（HAI-DEF）使用条款。 | 阅读、理解并严格遵守HAI-DEF使用条款中规定的所有限制和义务。 | 16 |
| 数据污染意识 | 模型在预训练中可能接触过与基准相关的医学信息，可能高估泛化能力。 | 在非公开或机构特有的数据集上验证模型，以真实评估泛化能力。 | 5 |
| 非临床级别 | 模型非临床级别，医学图像解读等可能需要额外微调。 | 认识到模型是起点，投入必要的微调和适配工作以达到特定应用所需的性能和可靠性。 | 14 |

## **VIII. MedGemma在不断发展的医学AI领域中的地位**

### **A. 与谷歌先前医学AI成果的比较**

MedGemma建立在Med-PaLM、Med-PaLM2和Med-Gemini等先前模型的学习和进步之上 9。与一些早期的、可能更为封闭的研究模型不同，MedGemma作为“开放模型”（尽管有特定条款）发布，旨在赋能开发者 9。它补充了谷歌健康AI的其他举措，如用于诊断对话的研究性AI智能体AMIE（Articulate Medical Intelligence Explorer）6。MedGemma代表了向更开放、更易于获取的医疗保健基础模型的战略演进，旨在实现开发能力的民主化。

MedGemma（开放基础模型）和AMIE（用于诊断对话的研究AI智能体）6 的同时突显，表明了谷歌在健康AI领域的多方面战略。MedGemma为更广泛的社区提供工具，而AMIE代表了谷歌自身对更自主AI智能体的前沿研究。MedGemma是一个供开发者构建应用的开放模型 9。AMIE是谷歌（DeepMind）的一个研究AI智能体，用于诊断对话，研究表明其表现良好，甚至表现出同理心 6。这些是不同但互补的方法。MedGemma旨在装备生态系统。AMIE展示了谷歌在先进AI能力方面的内部雄心。未来版本的AMIE或类似的谷歌开发的临床AI智能体，内部可能利用或建立在像MedGemma或其后续模型的架构和学习之上。这种双重方法使谷歌既能促进外部创新，又能推动其在医学AI领域的自身研究前沿。

### **B. 对医疗创新与研究的潜在影响**

MedGemma旨在加速新型医疗保健AI应用和产品的开发 9。它可以帮助快速有效地分析大量医疗数据，从而可能加速治疗和患者护理方面的突破 21。它支持医学图像分类、报告生成、问答、患者访谈、分诊、摘要和临床决策支持等应用 11。此外，它还有助于制定个性化治疗方案并优化医疗服务提供 11。通过提供一个功能强大、以医学为重点的基础模型，MedGemma可以降低研究人员和开发人员的入门门槛，从而可能催生更广泛的创新工具并提高医疗保健效率。

虽然大型机构可能会开发自己的基础模型，但MedGemma的开放性质（附带条件）可以使小型研究团体、初创企业和资源有限环境中的开发者能够构建专业的医疗AI工具。这可能会在利基领域或服务不足的人群中催生创新，这些领域或人群否则可能会被忽视。从头开始开发大型基础AI模型非常昂贵且资源密集。这通常将此类开发限制在主要科技公司和资金充足的研究联盟。通过发布MedGemma，谷歌提供了一个强大且经过医学调整的起点。这降低了创建复杂医疗AI应用的入门门槛。较小的实体可以将其资源集中用于针对特定本地需求、罕见疾病或特定人群微调MedGemma，而不是从头开始。如果开发仅限于少数大型参与者，这可能会促进针对更广泛医疗问题的多样化AI应用的“长尾”发展。

### **C. 临床采用的挑战与监管考量**

这些模型并非临床级别，仅供研发使用；开发者必须进行验证和调整 11。AI在临床应用中普遍面临的挑战包括互操作性、工作流程整合、维护、可持续性、人力资源需求以及影响信任、责任和报销的“黑箱”性质 25。信任是采纳的重要催化剂，受到算法偏见、公平性等问题的影响 26。医疗AI的监管和法律框架仍在发展中 26。HAI-DEF条款 18 通过将特定应用的监管责任置于开发者身上来明确应对这一问题。尽管技术取得了进步，但使用MedGemma构建的应用要实现广泛、安全和有效的临床应用，其道路将充满所有医学AI共有的挑战，包括严格验证、监管导航和建立临床医生信任。

## **IX. 结论与未来展望**

### **A. MedGemma能力与贡献总结**

MedGemma作为基于Gemma 3的开放模型系列（4B多模态，27B纯文本），专为医学图像和文本理解而设计。其主要优势在于凭借专业化训练在医学基准测试中展现出强大的基线性能，并能适应各种医疗保健应用。它作为开发者的基础工具，旨在加速医疗保健领域的AI创新。

### **B. MedGemma及类似开放式医学AI模型的未来之路**

持续的社区参与、开发验证和微调的最佳实践至关重要。未来版本的MedGemma可能会扩展功能（例如，更多模态、改进的多轮对话、超越以英语为中心的安全测试的更广泛语言支持）。持续关注负责任的AI原则，解决偏见，确保隐私，并适应不断变化的监管环境，这一点至关重要。MedGemma的手稿/技术报告“即将发布” 5，这可能会提供更深入的技术细节和见解。

反复提及完整的技术报告或手稿“即将发布” 5 对于一份专家级报告来说是一个关键点。虽然提供的摘要和模型卡提供了大量信息，但同行评审的技术论文通常会提供更深入的架构细节、更详尽的训练和评估方法、消融研究以及对局限性更彻底的讨论。专家级报告依赖于全面、可验证的技术文档。模型卡和开发者页面很有用，但通常是总结性发现。一份完整的技术手稿（通常针对重要的模型发布，常发表在arXiv或期刊/会议上）将允许研究社区进行更深入的审查。在获得这些摘要时，该手稿的缺失意味着任何当前的评估都是基于初步或摘要信息。最终报告应承认这一点，并指出即将发布的手稿对于全面理解和验证MedGemma的主张和能力至关重要。

MedGemma的发布凸显了一个固有的张力。“开放”模型促进了可访问性、快速创新和透明度。然而，医疗保健是一个高风险、高度管制的领域，要求极度谨慎、严格验证和明确的问责制。“开源”精神通常鼓励广泛使用和修改。医疗保健应用可能产生生死攸关的后果。谷歌试图通过健康人工智能开发者基础使用条款 18、“开发者模型”定位和广泛的免责声明来管理这种张力。然而，一旦模型的权重“公开”可用（即使附带条款），控制所有下游用途就变得具有挑战性。此类模型在医疗保健领域的未来成功将取决于开发者生态系统在自我监管和遵守安全与伦理最佳实践方面的成熟度，以及针对AI驱动医疗工具不断发展的正式监管框架。这对采用这些模型的社区提出了重大的长期责任。

MedGemma是一个重要的进步，但它只是持续旅程的一部分。其最终影响将取决于开发者如何负责任地使用它，以及更广泛的生态系统如何应对在医疗保健领域安全、公平地部署AI所面临的挑战。

#### **引用的著作**

1. MedGemma – Vertex AI \- Google Cloud console,  [https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma)  
2. developers.google.com,  [https://developers.google.com/health-ai-developer-foundations/medgemma\#:\~:text=The%20MedGemma%20collection%20contains%20Google's,a%2027B%20text%2Donly%20version.](https://developers.google.com/health-ai-developer-foundations/medgemma#:~:text=The%20MedGemma%20collection%20contains%20Google's,a%2027B%20text%2Donly%20version.)  
3. Building with AI: highlights for developers at Google I/O,  [https://blog.google/technology/developers/google-ai-developer-updates-io-2025/](https://blog.google/technology/developers/google-ai-developer-updates-io-2025/)  
4. What you should know from the Google I/O 2025 Developer keynote,  [https://developers.googleblog.com/en/google-io-2025-developer-keynote-recap/](https://developers.googleblog.com/en/google-io-2025-developer-keynote-recap/)  
5. MedGemma model card | Health AI Developer Foundations,  [https://developers.google.com/health-ai-developer-foundations/medgemma/model-card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)  
6. Google Unveils MedGemma: Innovative AI Model for Healthcare \- The Munich Eye,  [https://themunicheye.com/google-unveils-medgemma-ai-innovations-healthcare-21754](https://themunicheye.com/google-unveils-medgemma-ai-innovations-healthcare-21754)  
7. Gemma 3 \- Google DeepMind,  [https://deepmind.google/models/gemma/gemma-3/](https://deepmind.google/models/gemma/gemma-3/)  
8. Google AI Launches MedGemma for Analyzing Health-related Text & Images,  [https://www.digitalhealthnews.com/google-ai-launches-medgemma-for-analyzing-health-related-text-images](https://www.digitalhealthnews.com/google-ai-launches-medgemma-for-analyzing-health-related-text-images)  
9. Google Research at Google I/O 2025,  [https://research.google/blog/google-research-at-google-io-2025/](https://research.google/blog/google-research-at-google-io-2025/)  
10. Silicon Valley: A global engine for health and AI innovation \- RealChange,  [https://www.realchange.com/news/silicon-valley-a-global-engine-for-health-and-ai-innovation](https://www.realchange.com/news/silicon-valley-a-global-engine-for-health-and-ai-innovation)  
11. Google Launches MedGemma for Healthcare AI Application Development,  [https://community.hlth.com/insights/news/google-launches-medgemma-for-healthcare-ai-application-development-2025-05-22](https://community.hlth.com/insights/news/google-launches-medgemma-for-healthcare-ai-application-development-2025-05-22)  
12. Google-Health/medgemma \- GitHub,  [https://github.com/google-health/medgemma](https://github.com/google-health/medgemma)  
13. MedGemma Release \- a google Collection \- Hugging Face,  [https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4](https://huggingface.co/collections/google/medgemma-release-680aade845f90bec6a3f60c4)  
14. MedGemma | Health AI Developer Foundations,  [https://developers.google.com/health-ai-developer-foundations/medgemma](https://developers.google.com/health-ai-developer-foundations/medgemma)  
15. Google launches MedGemma to help jumpstart medical AI development,  [https://firstwordhealthtech.com/story/5965162](https://firstwordhealthtech.com/story/5965162)  
16. google/medgemma-4b-it at main \- Hugging Face,  [https://huggingface.co/google/medgemma-4b-it/tree/main](https://huggingface.co/google/medgemma-4b-it/tree/main)  
17. google/medgemma-27b-text-it \- Hugging Face,  [https://huggingface.co/google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it)  
18. Health AI Developer Foundations Terms of Use | Google for ...,  [https://developers.google.com/health-ai-developer-foundations/terms](https://developers.google.com/health-ai-developer-foundations/terms)  
19. Google AI Releases MedGemma: An Open Suite of Models Trained for Performance on Medical Text and Image Comprehension (Marktechpost) \- Scope Forward,  [https://scopeforward.com/google-ai-releases-medgemma-an-open-suite-of-models-trained-for-performance-on-medical-text-and-image-comprehension-marktechpost/](https://scopeforward.com/google-ai-releases-medgemma-an-open-suite-of-models-trained-for-performance-on-medical-text-and-image-comprehension-marktechpost/)  
20. Google Releases MedGemma, Built on Gemma 3, Specializing in Medical Text and Image Understanding | News,  [https://hyper.ai/en/news/39645](https://hyper.ai/en/news/39645)  
21. Google's Newest Gemma AI Model Sparks Excitement and Controversy with Mobile Deployment \- OpenTools,  [https://opentools.ai/news/googles-newest-gemma-ai-model-sparks-excitement-and-controversy-with-mobile-deployment](https://opentools.ai/news/googles-newest-gemma-ai-model-sparks-excitement-and-controversy-with-mobile-deployment)  
22. Is Google Gemini HIPAA Compliant? \- Nightfall AI,  [https://www.nightfall.ai/blog/is-google-gemini-hipaa-compliant](https://www.nightfall.ai/blog/is-google-gemini-hipaa-compliant)  
23. Terms and Conditions of Use | Google AI Edge \- Gemini API,  [https://ai.google.dev/edge/litert/next/tensor\_ml\_terms](https://ai.google.dev/edge/litert/next/tensor_ml_terms)  
24. Google's MedGemma: Transforming Medical AI with Multimodal Comprehension \- UBOS,  [https://ubos.tech/news/googles-medgemma-transforming-medical-ai-with-multimodal-comprehension/](https://ubos.tech/news/googles-medgemma-transforming-medical-ai-with-multimodal-comprehension/)  
25. Meeting the Moment: Addressing Barriers and Facilitating Clinical Adoption of Artificial Intelligence in Medical Diagnosis \- NAM,  [https://nam.edu/perspectives/meeting-the-moment-addressing-barriers-and-facilitating-clinical-adoption-of-artificial-intelligence-in-medical-diagnosis/](https://nam.edu/perspectives/meeting-the-moment-addressing-barriers-and-facilitating-clinical-adoption-of-artificial-intelligence-in-medical-diagnosis/)  
26. Barriers to and Facilitators of Artificial Intelligence Adoption in Health Care: Scoping Review,  [https://humanfactors.jmir.org/2024/1/e48633](https://humanfactors.jmir.org/2024/1/e48633)  
27. Bias recognition and mitigation strategies in artificial intelligence healthcare applications \- PMC \- PubMed Central,  [https://pmc.ncbi.nlm.nih.gov/articles/PMC11897215/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897215/)