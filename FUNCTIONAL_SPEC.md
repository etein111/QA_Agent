## QA_Agent 功能说明文档

本文件从「产品/功能」视角，描述 QA_Agent 系统的整体能力、工作流程以及关键设计点，便于前后端、算法和产品同学对齐。

---

### 一、系统定位

QA_Agent 是一个面向学生的 **智能答疑助手**，支持：

- 基于教师上传的 **PPT/PDF 课件** 等内部资料进行检索与回答；
- 结合 **多轮对话上下文** 进行连续问答；
- 在允许的情况下，利用 **LLM 自身提供的联网搜索能力**（如浏览器/搜索工具）补充最新资料；
- 支持 **图片提问**（如拍黑板、截图题目）；
- 在答案中给出清晰的 **引用标记**，标明哪些内容来自哪份课件或网页。

---

### 二、核心功能模块

1. **内部知识库（KB）检索**
   - 预处理教师的 PPT/PDF 课件，切分为带 `doc_id / filename / title / page_idx / text_content / tags` 的 segment。
   - 使用 embedding 构建向量索引，支持基于语义的相似度检索。
   - 对于多概念问题（如「自然谬误和柠檬市场是什么」），通过 `search_multi` 分别对多个子 query 检索，并合并结果，保证每个概念至少有一条命中。
   - 通过相似度阈值（`WEB_SEARCH_MIN_SIM`）控制召回质量：
     - 若所有 segment 的相似度都低于阈值，则认为「知识库无可靠命中」，返回空结果。

2. **LLM 对话与提示词控制**
   - 使用 `gpt-5-mini`（或兼容 OpenAI API 的模型）作为主要对话模型。
   - 统一的基础提示词 `SYSTEM_PROMPT_BASE` 约束回答风格：
     - 先直观解释，再给出严格结论；
     - 优先利用内部资料和上下文；
     - 对引用加内联标记 `[引用: ...]`；
     - 字数控制 ~300 字，Markdown 格式，条理清晰；
     - 对不确定内容必须声明不确定性，避免编造。
   - **联网搜索开关由前端控制**：
     - `use_web_search = true`：在 system prompt 中第 9 条明确写明「本轮你可以使用自身具备的联网搜索/浏览器能力，在需要时主动检索最新资料，并在引用处标明来源」。
     - `use_web_search = false`：在 system prompt 中写明「本轮禁止使用任何联网搜索或浏览器能力，请仅根据对话历史和内部知识库作答，不要假设或编造外部网页内容」。
   - 后端不再主动调用 DDG / Google 等第三方搜索 API，所有联网行为交由 LLM 自身工具完成。

3. **引用补丁与说明文案**
   - 在 LLM 原始输出基础上，对答案进行后处理：
     - 将检索到的每条知识库 chunk 转换成 citation 文本，例如：  
       `【资料：高阶偏导数，文件名：partial_derivative.pdf，第 21 页，高阶偏导数】`  
       并包裹为内联标记 `[引用: ...]`，插入到合适段落末尾。
   - 若本轮**知识库完全未命中**：
     - 当 `use_web_search = true` 时，在答案末尾追加：  
       `（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识与联网搜索。）`
     - 当 `use_web_search = false` 时，追加：  
       `（说明：内部知识库未检索到与本问题直接相关的资料，以下回答主要基于通识推理。）`
     - 对图片问答，同理区分是否联网：
       - 开启联网时：`…通识、图片理解与联网搜索。`
       - 关闭联网时：`…通识与图片理解。`

4. **图片问答（多模态）**
   - `answer_with_image` 支持将图片以 base64 的形式作为多模态输入传给 LLM。
   - 当用户只上传图片（`user_query` 为空或只有少量文字）时：
     - 自动关闭 `use_web_search`，避免传空 query 触发无意义的网页搜索；
     - 仅依赖图片内容 + 历史对话 + 内部知识库回答。

5. **Web 服务与接口层**
   - 使用 FastAPI 提供 HTTP API：
     - `/api/chat`：简单非流式文本问答；
     - `/api/chat/stream`：SSE 流式问答接口；
     - `/api/chat/image`：图片问答接口；
     - `/api/qa`：规范化 JSON 协议接口（推荐前端 & 其他后端使用）。
   - `BACKEND_API_SPEC.md` 中详细定义了 `/api/qa` 的请求/响应格式，包含：
     - `session_id / model / prompt / question / options / context`；
     - `answer.text` 与结构化引用 `answer.citations` 的预留设计。

6. **日志与调试**
   - 使用 `debug-6dec40.log` 记录关键事件（服务启动、HTTP 请求、检索结果摘要等），方便排查问题。
   - 使用 `retrieval-debug.log` 记录知识库检索的相似度结果（当前仍保留，便于分析 KB 行为）。
   - 通过 Langfuse 可选地记录 LLM 调用 trace。

---

### 三、典型调用流程（非流式）

1. 前端调用方：
   - 根据用户输入构造符合 `/api/qa` 规范的 JSON 请求：
     - 生成/传入 `session_id`、`question.id`；
     - 填写 `question.text` 和（可选的）`attachments`；
     - 设置 `options.use_internal_kb` / `options.use_web_search` / `options.temperature` 等；
     - 传入近期 `context.history`。
2. FastAPI：
   - 解析请求体，调用 `_map_qa_options` 映射成 QAAgent 所需参数；
   - 写一条 `qa_request` 调试日志；
   - 调用 `_agent.answer(...)`。
3. QAAgent：
   - 根据 `use_internal_kb` / `use_web_search` 计算 system prompt；
   - 调用内部知识库检索 `_retrieve_parallel`，得到 `retrieved_chunks`；
   - 调用 LLMClient 生成原始回答；
   - 若知识库无命中，追加末尾说明文案；
   - 对答案文本进行 citation 补丁，将 `[引用: ...]` 标记插入合适段落。
4. FastAPI：
   - 将得到的完整文本答案封装为：

     ```json
     {
       "session_id": "...",
       "question_id": "...",
       "answer": {
         "id": "",
         "model": "gpt-5-mini",
         "text": "完整 Markdown 答案……",
         "citations": []
       },
       "usage": null,
       "trace": null
     }
     ```

   - 返回给调用方。

---

### 四、后续可扩展方向

1. **结构化引用抽取**
   - 在 QAAgent 中增加对 `[引用: ...]` 的正则解析，将其映射回具体的 `doc_id / page / section` 或 web URL，并填充 `/api/qa` 响应中的 `answer.citations`。

2. **多源知识融合**
   - 除课件外，引入「外部教学网站库」（维基、学校官网、课程网站）作为第二知识库层，统一走 KB 逻辑，再将实时 Web 搜索作为第三层兜底。

3. **异常和安全控制**
   - 针对联网搜索结果，增加更细致的过滤策略和来源白名单；
   - 对模型输出中的敏感内容（如隐私信息、违规内容）增加审核模块。

4. **更丰富的 API 形态**
   - 对 `/api/qa` 增加流式版本（如 `/api/qa/stream`），统一协议；
   - 补充错误码、详细 usage、trace 字段，便于监控和调优。

---

通过上述设计，QA_Agent 已经具备了：

- 清晰的前后端接口协议；
- 较强的可解释性（引用标记与末尾说明）；
- 对联网搜索行为的精细控制（由前端开关 + system prompt 决定）；
- 面向教学场景的可扩展空间。 

