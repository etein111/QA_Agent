## 学生高等数学问答 Agent（uv 项目）

本项目是一个使用 **uv** 作为依赖管理工具、面向「高等数学学习问答」场景的对话型 Agent。

- **工具 1：内部知识库检索**（基于 data 下 PPT/PDF 的预处理 Segment 元数据，可精确定位到文件、页码、章节）
- **工具 2：联网搜索**（基于 DuckDuckGo 搜索，并在回答中引用 URL）
- **外挂知识库：** 暂不做持久化记忆，将历史聊天记录整体作为上下文注入 LLM
- **LLM 调用：** 统一通过 aihub 中转站（OpenAI 兼容接口）调用

### 使用前准备

1. 安装 uv（如未安装）：

```bash
pip install uv
```

2. 在项目根目录安装依赖：

```bash
uv sync
```

3. 配置 aihub 中转站相关环境变量（示例）：

```bash
set AIHUB_BASE_URL=https://your-aihub-endpoint/v1
set AIHUB_API_KEY=your_api_key_here
set QA_MODEL=gpt-4.1-mini
```

> 上述变量名可按你实际的 aihub 配置修改，只要与 `qa_agent/config.py` 中读取的名称保持一致即可。

4. **（可选）Langfuse 监控与 Trace 名称**

   在项目根目录放置 `.env`（可参考 `.env.example`），并配置：

   - `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_BASE_URL`：Langfuse 项目凭证与地址  
   - **`LANGFUSE_TRACE_NAME`**：在 Langfuse 中显示的 trace 名称，可自行定义（如 `qa-agent-math`、`student-math-dev` 等）

   配置后，每次对话会以该名称作为根 trace 上报到 Langfuse；若不配置或未设置 `LANGFUSE_TRACE_NAME`，默认使用 `qa-agent`。关闭监控可设 `LANGFUSE_TRACING_ENABLED=false`。

5. **（推荐）内部知识库数据预处理**

   检索需要先对 `data/` 下的 PPT/PDF 做预处理，生成按页/章节的 Segment 元数据（复用 MinerU 提取 + 按页聚合成 Segment 的逻辑）：

   ```bash
   # 在 .env 中配置 MINERU_TOKEN（MinerU 开放平台 Token）
   uv run python -m qa_agent.preprocess
   ```

   默认会扫描 `data/` 下的 `.pptx`、`.pdf`，调用 MinerU API 解析，再按页生成 Segment，写出到 `data/preprocessed/<doc_id>_segments.json`。若已存在 `data/mineru_output/<文件名 stem>__mineru` 则直接复用，不再调 API。强制重新解析可加 `--no-use-existing`。

   - 输出目录：`data/preprocessed/`（供 KnowledgeBase 加载）  
   - MinerU 解压目录：`data/mineru_output/`

6. **（可选）知识库 SQLite 持久化**

   默认会将知识库元数据与 embedding 持久化到 SQLite（`data/kb.sqlite`），避免每次启动重算向量：
   - **生成时机**：运行 `uv run python -m qa_agent.preprocess` 时，每写出一个 `*_segments.json` 即同步写入 `kb.sqlite`。
   - 数据库路径：`QA_KB_DB_PATH`（默认 `data/kb.sqlite`）；禁用：`QA_KB_USE_SQLITE=false`。
   - 应用启动时仅从 DB 加载；若 DB 不存在或为空，则从 `data/preprocessed/*_segments.json` 回填一次。

### 运行对话 Agent（命令行）

```bash
uv run qa-agent-chat
```

运行后，将进入一个简单的命令行对话界面。特点：

- 面向 **高等数学学习** 提问；
- 如果用到了内部知识库（预处理后的 PPT/PDF Segment），会在回答中给出可精确定位的引用，例如：
  - `【资料：<标题>，文件名：xxx.pdf，第 3 页，<章节/ section>】`
- 如果用到了联网搜索结果，会引用 URL：
  - `【来源：https://example.com/xxx】`

### 运行 Web 前端（推荐用于公式与引用展示）

```bash
uv run qa-agent-web
```

然后在浏览器中访问 `http://localhost:8000/`：

- 中间区域展示对话记录（支持多轮对话）；
- Agent 输出中的 LaTeX 公式会通过 MathJax 自动渲染；
- 回答中出现的：
  - `http://` / `https://` 开头的 URL 会自动变成可点击链接（新标签页打开）；  
  - 引用中形如 `文件名：xxx.pdf` / `文件名：xxx.pptx` 的部分会自动转成 `/files/xxx.*` 的链接，点击即可直接在浏览器中打开对应的 PDF / PPTX 文件（前提是文件位于项目根的 `data/` 目录）。

### 目录结构（核心部分）

```text
qa_agent/
  config.py
  llm_client.py
  agent.py
  main.py
  web/
    app.py                # FastAPI Web 服务：/api/chat、/files、/static
  preprocess/              # 数据预处理（复用 MinerU Segment 逻辑）
    data_types.py          # MinerUElement, Segment, SegmentMetadata
    mineru_loader.py       # 从 MinerU *_content_list.json 加载元素
    mineru_api/            # MinerU API 客户端与 locate_outputs
    segmenter_simple.py    # 按页聚合成 Segment（不依赖 VLM）
    pipeline.py            # 单文件/整目录预处理入口
    __main__.py            # 命令行：uv run python -m qa_agent.preprocess
  tools/
    knowledge_base.py      # 基于 SegmentMetadata 的检索（文件/页码/章节）
    web_search.py
data/
  *.pptx, *.pdf            # 原始课件/习题
  mineru_output/           # MinerU 解析产物（可选，可复用）
  preprocessed/            # *_segments.json，供 KnowledgeBase 加载
```

后续如需扩展为 Web 服务，可以在此基础上新增 FastAPI / Django 等框架。

