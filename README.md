# Frankenstories: Story Generation from Reusable Parts


## 1. 项目描述

本项目比较LLM生成ComicXML格式漫画故事的两种方法：

1. **Method A (Co-writing / Interactive Storytelling (IST))**: 人与AI进行真正的多轮对话，细化故事设定后生成ComicXML \
多轮对话是否能减少 hallucination？多轮交互是否能提高叙事质量？\
🟦 Method A 的优点：\
故事更完整、逻辑性更好\
角色行为、场景发展更自然\
用户能引导故事方向（更个性化）\
🟦 Method A 的缺点：\
容易 hallucinate 出库里没有的元素\
多轮对话越多错误越多\
更难保持 XML 格式正确\
难以高一致性选择角色变体
2. **Method B (Template Selection / Retrieval-Augmented Generation (RAG))**: AI直接从元素库中选择最合适的元素，填充4格模板生成ComicXML\
强调结构化输出，不强调创造性\
 🟧 Method B 的优点：\
非常稳定\
不太会产生不存在的 pose、setting\
更适合结构化输出\
容易达成高一致性\
🟧 Method B 的缺点：\
故事逻辑更弱\
不一定能从库中的相似元素中选择处最恰当的\
不一定能连续推进剧情\
过于机械（slot-filling）

### LLM 和 Prompt 选择
实验将设计成多因子：Method × Model × Prompt Type, 以观察：\
不同方法下模型表现是否一致\
Prompt 类型对生成质量的影响\
模型参数大小对故事逻辑/库元素选择能力的影响

- **GPT-4**: Chain-of-Thought; 实际采用：gpt-4.1-2025-04-14
- **GPT-o1**: Direct；实际采用：o1-2024-12-17
- **DeepSeek-7B**: Chain-of-Thought：实际采用DeepSeek-R1-Distill-Qwen-7B
- **LLaMA-3-70B**: Direct； 实际采用：Llama-3.3-70B

以上全部通过 aihubmix 代理实现api连接。

| | GPT-4 (CoT) | GPT-o1 (Direct) | DeepSeek-7B (CoT) | LLaMA-3-70B (Direct) |
| -------------- | ----------- | --------------- | ----------------- | -------------------- |
| **Method A**   | 多轮 CoT 生成   | 多轮 Direct 生成    | 多轮 CoT 生成         | 多轮 Direct 生成         |
| **Method B**   | 单轮 CoT 填充   | 单轮 Direct 填充    | 单轮 CoT 填充         | 单轮 Direct 填充         |


---

## 2. 数据准备

###  2.1 准备元素库约束：只能使用预定义元素库中的角色、姿势、场景等元素
提取所有元素库中元素至element_library.json\
**角色 (Characters)**: 3个角色，共7个变体

| 角色 | 变体名称 | 外观描述 | 出现次数 |
|------|---------|---------|---------|
| **Alfie** | `default` | 男性，浅棕色皮肤，深棕色头发，红色嘴唇 | 1,043次 |
| | `light_hair` | 男性，浅棕色皮肤，浅棕色头发，红色嘴唇 | 32次 |
| | `dark` | 男性，黑色皮肤，黑色头发，红色嘴唇 | 1次 |
| **Betty** | `default` | 女性，白色皮肤，金色头发，粉色嘴唇 | 846次（默认形象） |
| | `green_male` | 男性，绿色皮肤，绿色头发 | 48次（角色扮演时） |
| | `green_female` | 女性，绿色皮肤，深绿色头发 | 4次（角色扮演时） |
| **Gemma** | `default` | 女性，橄榄色皮肤，黑色头发 | 26次（完全固定） |

**其他元素**:

- **姿势 (Poses)**: 586个
- **场景 (Settings)**: 149个
- **朝向 (Facings)**: 2个（`left`, `right`）
- **位置 (Positions)**: 3个（`left`, `middle`, `right`）
- **气球状态 (Balloon Status)**: 1个（`speech`）
- **边框 (Border)**: 1个（`white`）
- **框架 (Framing)**: 1个（`picture panel`）

### 2.2 提取规范的ComicXML

依据specification中的结构，创建template.xml，涵盖所有规范结构\
占位符 + 属性清单 → 指明必须元素和结构\
组合规则 + 注释 → 指明可选/多样化情况\
明确告诉 LLM 输出格式\
便于后续自动检查生成的 XML 是否规范\
同时排除因为指令不清楚导致的错误\
删除了封面页有关的embedding、framing和border设定——因为本实验不需要测试封面

### 2.3 设计故事输入设定

#### 主题内容来源
主题应该与元素库相关以排除对结果Hallucination的干扰\
模型需要自己选择合适角色、pose、setting 来生成 ComicXML\
主题不应直接指明具体元素名称。防止LLM照搬元素库，无法测试模型的理解能力和Repetition指标

#### 主题数量与多次生成
为了统计 Prompt Sensitivity / Method Comparison，需要重复实验获取可量化结果。\
同一个主题应在两种方法下使用：Method A vs B 在同一输入下的输出差异才能有效量化\
准备 10–20 个主题（取决于实验规模和时间成本）\
对每个主题 生成 3 次 ComicXML（每次用不同 prompt 变体）\
既可以分析主题广度，也分析同一主题下模型/Prompt 的稳定性


#### 主题模糊度与边界
如果太具体，模型没有自由度，Diversity/Repetition/Prompt Sensitivity 等指标可能会失效\
如果太模糊，模型随机性太大，故事可能不连贯，Local Coherence / Progression 的评估困难

应给出 基本场景 + 简单事件，不指定角色或 pose\
控制长度：1–2 句即可\
尽量留出选择空间给模型选角色、pose、setting


## 3. 实验实现
### 3.1 API 调用与 Method 设计

为了实现 ComicXML 的批量生成并保证实验可重复性，本实验通过 **API 调用 LLM** 完成故事生成，同时区分 Method A 与 Method B 的不同设计思路

#### 3.1.1 Method A: 多轮模拟交互生成（Co-writing / IST）

1. **目标**

模拟真实的人机多轮对话，引导模型逐步细化故事内容并生成 ComicXML\
探索多轮交互是否能减少 hallucination，并提高故事逻辑和创造性

2. **输入结构**

初始轮 prompt 包含：
* 故事主题
* 元素库约束（角色、pose、setting 可选列表）
* 输出格式要求（ComicXML template）

模型不会直接读取 JSON 或文件，而是根据 prompt 中描述的可选元素生成内容

每轮模拟用户追问/补充示例：

     * “请描述主角在场景中的第一反应。”
     * “加入新的动作或角色互动。”
     * “强调事件发生后的角色情绪或反应。”

3. **对话轮数与上下文累积**

对话轮数可固定 3–5 轮，也可根据故事复杂度调整\
每轮输出会 累积到对话历史中，作为下一轮 prompt 的一部分，保证上下文连贯\
元素库信息只需在初始轮提供一次，后续轮次仅传递对话历史 + 当前追问，观察模型是否正确遵循初始约束

4. **API 调用设计**

每轮调用 LLM 接口（OpenAI API、DeepSeek、LLaMA），传入对话历史 + 当前轮 prompt\
输出为 JSON 或 XML 字符串（ComicXML），可直接解析或保存\
支持批量生成：同一主题可生成多版本，便于统计 Prompt Sensitivity、元素合法性等指标

5. **优势与注意事项**

保留多轮交互逻辑，增强故事连贯性和角色发展\
模拟多轮可降低实验时间成本，同时便于量化对比\
注意 XML 格式规范，防止连续轮次生成时格式错误


#### 3.1.2 Method B: 单轮模板填充生成（Template / RAG）

1. **目标**

强调结构化输出，使用预定义模板和元素库约束生成 ComicXML\
分析在严格 slot-filling 下，模型选择库元素的能力与稳定性

2. **输入结构**

单轮 prompt：包含故事主题、ComicXML 模板、元素库约束\
模板中预设占位符，模型只需选择合法角色、pose、setting 填入模板生成完整 ComicXML

3. **API 调用设计**

单次调用 LLM 接口，将主题 + 模板作为 prompt\
输出结果直接为 ComicXML，可批量生成多个版本\
可通过不同 prompt 变体测试 Prompt Sensitivity 和 Method Comparison

4. **优势与注意事项**

输出稳定，可控制 XML 格式正确性与元素合法性\
不依赖用户引导，便于大规模批量实验\
故事创造性和逻辑连贯性可能略低，适合对比 Method A 的效果差异



#### 3.1.3 批量生成流程示意

1. **准备**

主题列表（seeds文件夹里的20个seeds）\
ComicXML 模板（template.xml）\
元素库约束（element_library.json）

2. **Method A 批量生成**

对每个主题进行 3–5 次多轮模拟：
1. 初始化 prompt → 调用 API → 保存输出 
2. 根据输出生成下一轮 prompt → 调用 API → 保存输出
3. 重复至设定轮数 → 最终生成 ComicXML 

每次生成版本保存，用于后续指标统计

3. **Method B 批量生成**

对每个主题进行 3–5 次单轮模板填充：\
单轮 prompt（包含主题 + 模板 + 元素库约束） → 调用 API → 保存输出
 
输出 ComicXML 可直接解析

---


## 4. 评估维度
### 分阶段评估结构
第一阶段：量化可测、库约束可自动检查的指标（元素合法性 + 结构有效性 + 角色一致性）\
第二阶段：语义、连贯性、故事逻辑、创造性指标（故事推进、信息增益、元素多样性）

### 4.1 第一阶段评估指标
可直接从 XML/元素库/文本统计自动计算，强调规范性与稳定性。数据应尝试做成可视化表格。

#### 4.1.1 ComicXML输出ComicXML 输出指标

| 指标                                 | 定义                                                                   | 度量方法              |
|------------------------------------| -------------------------------------------------------------------- | ----------------- |
| **Element Validity**               | 每个元素（character/variant、pose、setting、facing、position、balloon）是否存在于元素库 | 合法元素数 / 总元素数      |
| **Hallucination Rate**             | 生成元素中属于库外的比例                                                         | 库外元素数 / 总元素数      |
| **Structural Validity**            | XML 格式是否正确（标签、必需属性、语法）                                               | 通过验证样本数 / 总样本数    |
| **Character Variant Consistency**  | 同一角色跨 panel 是否使用一致变体                                                 | 无矛盾变体次数 / 角色出现总次数 |
| **Repetition / Redundancy**    | 相同元素或台词重复率                                                           | 重复元素/台词数占比        |
| **Diversity**                      | 不同 pose、setting、角色组合等的类型数 / 面板数                                      | 类型数 / 面板数         |

#### 4.1.2 LLM/Prompt 评估指标

| 指标                           | 定义                             | 度量方法                                                                                                                     |
|------------------------------| ------------------------------ |--------------------------------------------------------------------------------------------------------------------------|
| **Prompt Sensitivity (整体)**       | 同一故事设定下，不同 prompt 是否导致元素或结构变化  | 对比不同 prompt 输出的 Element Validity、Structural Validity、Character Variant Consistency、Repetition/Diversity，计算差异率            |
| **Prompt Sensitivity (元素级)**      | 分元素类别统计敏感性，发现模型在哪些元素最容易受 prompt 影响 | 对每类元素（Variant、Pose、Setting、Facing、Balloon/Position）计算不同 prompt 输出不一致的比例：Sensitivity(element) = 元素类别出现不一致的面板数 / 该元素类别总面板数 |
| **Method Comparison Metric** | Method A 与 Method B 在元素和结构上的差异 | 对比 Hallucination Rate、Structural Validity、Character Variant Consistency、Repetition/Diversity                             |

### 4.2 第二阶段评估指标
关注故事推进、语义连贯性、创造性，可能需要人工辅助或 NLP proxy

#### 4.2.1 ComicXML输出ComicXML 输出指标

| 指标                                           | 定义                   | 度量方法                                        |
| -------------------------------------------- | -------------------- | ------------------------------------------- |
| **Character Presence Consistency** | 角色出现顺序或频率是否合理                                                        | 连续出现面板比例          |
| **Local Coherence / Contextual Consistency** | 相邻 panel 在文本或角色上是否连贯 | Sentence-BERT embedding 相似度、角色/实体重叠率；人工评分可选 |
| **Progression / Story Advancement**          | 故事是否推进，有新增事件或信息      | 事件计数（谓词提取）、信息增益或人工评分                        |
| **Creativity / Narrative Richness**          | 角色互动、场景变化、剧情新意       | 主观打分（可结合 panel 多样性统计）                       |
| **High-level Repetition / Redundancy**       | 故事内容重复、情节退步          | 人工评分或语义分析                                   |

#### 4.2.2 LLM/Prompt 评估指标
| 指标                                          | 定义                                    | 度量方法                                                                          |
| ------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| **Prompt Sensitivity**       | 同一故事设定下，不同 prompt 是否导致故事语义/情节差异       | 比较 Local Coherence、Progression、Creativity 指标，人工评分或语义 embedding 差异             |
| **Method Comparison Metric** | Method A 与 Method B 在故事逻辑/连贯性/创造性上的差异 | 对比高层指标，如 Local Coherence、Progression、Creativity、High-level Repetition，可结合人工评分 |

## 评估计划
