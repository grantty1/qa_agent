import os

import streamlit as st

from qa_agent import DeepSeekWebSearchClient, KnowledgeBase, QAAgent, env_int


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="ignore")


def search_badge(meta: dict) -> str:
    if meta.get("web_search_error"):
        return f"联网搜索失败：{meta['web_search_error']}"
    if meta.get("searched"):
        return "本次回答已联网搜索"
    if meta.get("attempted"):
        return "本次联网搜索未返回结果"
    return "本次回答未联网搜索"


@st.cache_resource
def build_agent(model, base_url, web_search_mode, search_count, kb_paths, kb_texts):
    try:
        kb = None
        if kb_paths:
            kb = KnowledgeBase(list(kb_paths))
        if kb_texts:
            if kb is None:
                kb = KnowledgeBase([])
            for name, content in kb_texts:
                kb.add_document(name, content)
        if kb is not None and not kb.chunks:
            kb = None

        web_client = None
        if web_search_mode != "off":
            try:
                web_client = DeepSeekWebSearchClient(count=search_count)
            except RuntimeError:
                web_client = None
        agent = QAAgent(
            model=model,
            base_url=base_url,
            kb=kb,
            web_search_client=web_client,
            web_search_mode=web_search_mode,
        )
        return agent, None
    except Exception as exc:
        return None, str(exc)


st.set_page_config(page_title="智能问答 Agent", layout="wide")
st.markdown(
    """
    <style>
    .qa-hero {
        text-align: center;
        padding: 1.4rem 0 0.2rem;
    }
    .qa-hero h1 {
        font-size: 2.7rem;
        font-weight: 700;
        letter-spacing: 3px;
        margin: 0 0 0.3rem;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .qa-hero p {
        margin: 0;
        color: #7a8699;
        font-size: 0.95rem;
        letter-spacing: 1px;
    }
    </style>
    <div class="qa-hero">
        <h1>🤖 智能问答 Agent</h1>
        <p>知识库 · 联网搜索 · 流式输出</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("配置")
    model = st.text_input(
        "主模型", value=os.getenv("OPENAI_MODEL", "x-ai/grok-4.7")
    )
    base_url = st.text_input(
        "Base URL", value=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    )
    web_search_mode = st.selectbox(
        "联网搜索模式",
        ["auto", "always", "off"],
        index=["auto", "always", "off"].index(
            os.getenv("WEB_SEARCH_MODE", "auto")
        ),
    )
    force = st.radio(
        "本次提问联网方式",
        ["自动", "强制联网", "禁用联网"],
        horizontal=True,
        disabled=web_search_mode != "auto",
    )
    if web_search_mode != "auto":
        st.caption("全局联网搜索模式为 always/off 时，本次提问联网方式不生效。")
    force_map = {"自动": None, "强制联网": True, "禁用联网": False}
    search_count = st.slider(
        "搜索结果条数",
        min_value=1,
        max_value=20,
        value=min(max(env_int("DEEPSEEK_SEARCH_COUNT", 5), 1), 20),
        help="DeepSeek 搜索接口每次返回的网页条数（接口允许 1-50）。",
    )

    st.subheader("知识库")
    uploaded_files = st.file_uploader(
        "上传电脑中的知识库文件",
        accept_multiple_files=True,
        type=["txt", "md", "csv", "json", "log"],
        help="从你的电脑选择文件上传。支持 UTF-8、GBK 等常见文本编码。",
    )
    kb_dir = os.path.dirname(os.path.abspath(__file__))
    disk_files = sorted(
        f for f in os.listdir(kb_dir) if f.lower().endswith(".txt")
    )
    selected_disk = st.multiselect("或选择脚本目录下的文件", disk_files)

    kb_paths = tuple(selected_disk)
    kb_texts = tuple(
        (uf.name, decode_text(uf.getvalue())) for uf in (uploaded_files or [])
    )
    if not kb_paths and not kb_texts:
        st.caption("提示：未加载知识库，仅使用大模型回答。")

    main_key_ready = bool(
        os.getenv("OPENAI_API_KEY") or os.getenv("API_Key_gty")
    )
    deepseek_key_ready = bool(
        os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek_api_key_gty")
    )
    st.caption(f"主模型 API Key: {'已设置' if main_key_ready else '未设置'}")
    st.caption(
        f"DeepSeek 搜索 API Key: {'已设置' if deepseek_key_ready else '未设置'}"
    )

    if st.button("清空对话记忆"):
        st.session_state["history"] = []
        st.rerun()

agent, build_err = build_agent(
    model, base_url, web_search_mode, search_count, kb_paths, kb_texts
)
if build_err:
    st.error(f"初始化失败：{build_err}")
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            st.caption(search_badge(msg["meta"]))

if prompt := st.chat_input("输入你的问题…"):
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("思考中…")
        try:
            agent.history = list(st.session_state["history"][:-1])
            answer_parts = []
            meta = None
            for delta, chunk_meta in agent.ask_stream_with_meta(
                prompt,
                force_web_search=(
                    force_map[force] if web_search_mode == "auto" else None
                ),
            ):
                if delta:
                    answer_parts.append(delta)
                    placeholder.markdown("".join(answer_parts))
                if chunk_meta is not None:
                    meta = chunk_meta
            answer = "".join(answer_parts)
            st.session_state["history"] = list(agent.history)
            if (
                st.session_state["history"]
                and st.session_state["history"][-1]["role"] == "assistant"
            ):
                st.session_state["history"][-1]["meta"] = meta
            placeholder.markdown(answer)
            st.caption(search_badge(meta))
        except Exception as exc:
            st.session_state["history"].pop()
            placeholder.error(f"出错：{exc}")
