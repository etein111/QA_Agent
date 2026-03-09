const chatWindow = document.getElementById("chat-window");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const imageInput = document.getElementById("image-input");
const imagePreview = document.getElementById("image-preview");

/** 当前对话历史：[{role, content}] */
const history = [];

/** 通过粘贴/拖拽选中的待发送图片文件（优先级高于文件选择框） */
let pendingImageFile = null;

function appendMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = role === "user" ? "你" : "Agent";

  const body = document.createElement("div");
  body.className = "message-content";
  body.innerHTML = renderContent(content);

  msg.appendChild(header);
  msg.appendChild(body);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // 触发 MathJax 重新渲染公式
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([msg]).catch(() => {});
  }
}

/**
 * 抽出数学公式（$$ 块级、$ 行内、\[ \] 块级、\( \) 行内），用占位符替换，避免被 Markdown 破坏；最后还原并由 MathJax 渲染。
 */
function extractMath(text) {
  const display = [];
  const inline = [];
  let s = text
    // 先匹配 $$ ... $$ 和 \[ ... \]，再匹配 $ ... $ 和 \( ... \)，避免误匹配
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => {
      display.push(formula.trim());
      return `\u200B__MATH_D_${display.length - 1}__\u200B`;
    })
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, formula) => {
      display.push(formula.trim());
      return `\u200B__MATH_D_${display.length - 1}__\u200B`;
    })
    .replace(/\$([^$]*?)\$/g, (_, formula) => {
      inline.push(formula.trim());
      return `\u200B__MATH_I_${inline.length - 1}__\u200B`;
    })
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, formula) => {
      inline.push(formula.trim());
      return `\u200B__MATH_I_${inline.length - 1}__\u200B`;
    });
  return { text: s, display, inline };
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * 轻量 Markdown：**加粗**、## 标题、- 无序列表、1. 有序列表、段落空行。
 * 支持数学公式 $...$ 与 $$...$$，并处理 URL、[引用: ... 文件名：xxx.pdf] 的链接。
 */
function renderContent(text) {
  if (!text || !text.trim()) return "";

  const { text: textWithoutMath, display: displayFormulas, inline: inlineFormulas } = extractMath(text);
  let html = escapeHtml(textWithoutMath)
    // **加粗**
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");

  const lines = html.split("\n");
  const out = [];
  let inList = false;
  let listTag = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed) {
      if (inList) {
        out.push(listTag === "ul" ? "</ul>" : "</ol>");
        inList = false;
      }
      out.push("");
      continue;
    }
    if (/^##\s+.+/.test(trimmed)) {
      if (inList) {
        out.push(listTag === "ul" ? "</ul>" : "</ol>");
        inList = false;
      }
      out.push("<h2>" + trimmed.replace(/^##\s+/, "") + "</h2>");
      continue;
    }
    if (/^###\s+.+/.test(trimmed)) {
      if (inList) {
        out.push(listTag === "ul" ? "</ul>" : "</ol>");
        inList = false;
      }
      out.push("<h3>" + trimmed.replace(/^###\s+/, "") + "</h3>");
      continue;
    }
    if (/^-\s+/.test(trimmed) || /^\*\s+/.test(trimmed)) {
      if (!inList || listTag !== "ul") {
        if (inList) out.push(listTag === "ul" ? "</ul>" : "</ol>");
        out.push("<ul>");
        listTag = "ul";
        inList = true;
      }
      out.push("<li>" + trimmed.replace(/^[-*]\s+/, "") + "</li>");
      continue;
    }
    const numBullet = trimmed.match(/^(\d+)\.\s+(.+)/);
    if (numBullet) {
      if (!inList || listTag !== "ol") {
        if (inList) out.push(listTag === "ul" ? "</ul>" : "</ol>");
        out.push("<ol>");
        listTag = "ol";
        inList = true;
      }
      out.push("<li>" + numBullet[2] + "</li>");
      continue;
    }
    if (inList) {
      out.push(listTag === "ul" ? "</ul>" : "</ol>");
      inList = false;
    }
    out.push("<p>" + trimmed + "</p>");
  }
  if (inList) out.push(listTag === "ul" ? "</ul>" : "</ol>");

  html = out.join("\n").replace(/\n\n+/g, "\n");

  // 1) 自动识别 URL
  const urlRegex = /(https?:\/\/[^\s\]<]+)/g;
  html = html.replace(
    urlRegex,
    (m) => `<a href="${m}" target="_blank" rel="noreferrer">${m}</a>`,
  );

  // 2) 处理引用中的 pdf/pptx 文件名
  const fileRegex = /(文件名：)([^\s，\]<]+\.(?:pdf|pptx?))/g;
  html = html.replace(fileRegex, (_, prefix, filename) => {
    const href = `/files/${encodeURIComponent(filename)}`;
    return `${prefix}<a href="${href}" target="_blank" rel="noreferrer">${filename}</a>`;
  });

  // 3) 还原数学公式为 \( \) 与 \[ \]，供 MathJax 识别渲染（公式内 < > 转义以免破坏 HTML）
  displayFormulas.forEach((formula, i) => {
    const safe = escapeHtml(formula);
    html = html.replace(`\u200B__MATH_D_${i}__\u200B`, `<span class="math-display">\\[${safe}\\]</span>`);
  });
  inlineFormulas.forEach((formula, i) => {
    const safe = escapeHtml(formula);
    html = html.replace(`\u200B__MATH_I_${i}__\u200B`, `<span class="math-inline">\\(${safe}\\)</span>`);
  });

  return html;
}

/**
 * 创建一条可增量更新的助手消息，返回用于更新内容的 body 元素。
 */
function createStreamingMessage() {
  const msg = document.createElement("div");
  msg.className = "message agent";

  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = "Agent";

  const body = document.createElement("div");
  body.className = "message-content";

  msg.appendChild(header);
  msg.appendChild(body);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return { msg, body };
}

/**
 * 解析 SSE 流中的 data 行（data: {...}），返回解析后的对象或 null。
 */
function parseSSEData(line) {
  const s = line.trim();
  if (s.startsWith("data:")) {
    try {
      return JSON.parse(s.slice(5).trim());
    } catch (_) {
      return null;
    }
  }
  return null;
}

/** 从页面读取当前参数（温度、是否联网、知识库/联网条数） */
function getParams() {
  const temperature = Math.max(0, Math.min(1, parseFloat(document.getElementById("param-temperature").value) || 0.3));
  const use_web_search = document.getElementById("param-use-web").checked;
  const kb_top_k = Math.max(1, Math.min(20, parseInt(document.getElementById("param-kb-top-k").value, 10) || 3));
  const web_max_results = Math.max(1, Math.min(15, parseInt(document.getElementById("param-web-top-k").value, 10) || 5));
  return { temperature, use_web_search, kb_top_k, web_max_results };
}

/** 读取文件为 base64（去掉 data:image/...;base64, 前缀） */
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        const parts = result.split(",");
        resolve(parts.length > 1 ? parts[1] : "");
      } else {
        resolve("");
      }
    };
    reader.onerror = () => reject(reader.error || new Error("读取图片失败"));
    reader.readAsDataURL(file);
  });
}

function clearPendingImage() {
  pendingImageFile = null;
  if (imageInput) imageInput.value = "";
  if (imagePreview) imagePreview.innerHTML = "";
}

function setPendingImageFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    clearPendingImage();
    return;
  }
  pendingImageFile = file;
  if (!imagePreview) return;
  const reader = new FileReader();
  reader.onload = () => {
    const result = reader.result;
    if (typeof result !== "string") {
      return;
    }
    imagePreview.innerHTML = "";
    const wrapper = document.createElement("div");
    wrapper.className = "image-preview-inner";
    const img = document.createElement("img");
    img.src = result;
    img.alt = "待发送的图片";
    img.className = "image-preview-img";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "image-preview-close";
    close.textContent = "×";
    close.addEventListener("click", clearPendingImage);
    wrapper.appendChild(img);
    wrapper.appendChild(close);
    imagePreview.appendChild(wrapper);
  };
  reader.readAsDataURL(file);
}

/** 在对话窗口中展示一条带图片的用户消息 */
function appendUserImageMessage(text, dataUrl) {
  const msg = document.createElement("div");
  msg.className = "message user";

  const header = document.createElement("div");
  header.className = "message-header";
  header.textContent = "你";

  const body = document.createElement("div");
  body.className = "message-content";

  if (text && text.trim()) {
    const textHtml = renderContent(text);
    body.innerHTML = textHtml;
  }

  const img = document.createElement("img");
  img.src = dataUrl;
  img.alt = "上传的图片";
  img.className = "message-image";
  body.appendChild(img);

  msg.appendChild(header);
  msg.appendChild(body);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text && !pendingImageFile && !(imageInput && imageInput.files && imageInput.files[0])) {
    return;
  }
  userInput.value = "";
  sendBtn.disabled = true;

  const params = getParams();
  const file = pendingImageFile || (imageInput && imageInput.files && imageInput.files[0]);

  // 有图片时走非流式的 /api/chat/image 接口；否则走原来的流式接口
  if (file) {
    // 先在前端展示用户的图片（和文字）
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          resolve(reader.result);
        } else {
          resolve("");
        }
      };
      reader.onerror = () => reject(reader.error || new Error("读取图片失败"));
      reader.readAsDataURL(file);
    });
    appendUserImageMessage(text, dataUrl);
    history.push({ role: "user", content: text || "[图片问题]" });
    userInput.value = "";

    const msg = document.createElement("div");
    msg.className = "message agent";
    const header = document.createElement("div");
    header.className = "message-header";
    header.textContent = "Agent";
    const body = document.createElement("div");
    body.className = "message-content";
    msg.appendChild(header);
    msg.appendChild(body);
    chatWindow.appendChild(msg);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    let answer = "";
    try {
      const image_b64 = await readFileAsBase64(file);
      clearPendingImage();
      const resp = await fetch("/api/chat/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_query: text, history, image_b64, ...params }),
      });
      if (!resp.ok) {
        body.innerHTML = "请求失败，请稍后再试。";
        return;
      }
      const data = await resp.json();
      answer = data.answer || "";
      body.innerHTML = renderContent(answer);
      chatWindow.scrollTop = chatWindow.scrollHeight;
      history.push({ role: "assistant", content: answer });
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([msg]).catch(() => {});
      }
    } catch (e) {
      body.innerHTML = "请求出错，请稍后再试。";
    } finally {
      sendBtn.disabled = false;
    }
    return;
  }

  // 纯文本消息：先在对话窗口中展示用户的问题
  appendMessage("user", text);
  history.push({ role: "user", content: text });

  const { msg, body } = createStreamingMessage();
  let accumulated = "";

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_query: text, history, ...params }),
    });

    if (!resp.ok) {
      body.innerHTML = "请求失败，请稍后再试。";
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";

      for (const block of events) {
        const dataLine = block.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        const obj = parseSSEData(dataLine);
        if (!obj) continue;
        if (obj.content !== undefined) {
          accumulated += obj.content;
          body.innerHTML = renderContent(accumulated);
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        if (obj.done && obj.full_content !== undefined) {
          accumulated = obj.full_content;
          body.innerHTML = renderContent(accumulated);
          chatWindow.scrollTop = chatWindow.scrollHeight;
        }
      }
    }

    if (buffer.trim()) {
      const dataLine = buffer.split("\n").find((l) => l.startsWith("data:"));
      if (dataLine) {
        const obj = parseSSEData(dataLine);
        if (obj && obj.done && obj.full_content !== undefined) {
          accumulated = obj.full_content;
          body.innerHTML = renderContent(accumulated);
        }
      }
    }

    history.push({ role: "assistant", content: accumulated });

    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([msg]).catch(() => {});
    }
  } catch (e) {
    body.innerHTML = "请求出错，请稍后再试。";
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// 文件选择时更新预览
if (imageInput) {
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      setPendingImageFile(file);
    } else {
      clearPendingImage();
    }
  });
}

// 支持在输入框中 Ctrl+V 粘贴图片
userInput.addEventListener("paste", (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const item of items) {
    if (item.kind === "file" && item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) {
        setPendingImageFile(file);
      }
      break;
    }
  }
});

// 支持将图片拖拽到聊天窗口区域
["dragenter", "dragover"].forEach((type) => {
  chatWindow.addEventListener(type, (e) => {
    e.preventDefault();
    e.stopPropagation();
  });
});

chatWindow.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  const files = e.dataTransfer && e.dataTransfer.files;
  if (!files || !files.length) return;
  const file = Array.from(files).find((f) => f.type.startsWith("image/"));
  if (file) {
    setPendingImageFile(file);
  }
});

