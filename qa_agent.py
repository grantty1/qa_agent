import argparse
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from openai import OpenAI


DEEPSEEK_SEARCH_URL = "https://api.deepseek.com"
DEEPSEEK_SEARCH_PATH = "/v1/tools/search"


def env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to `default`."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


WEB_SEARCH_HINTS = (
    "today",
    "now",
    "latest",
    "recent",
    "news",
    "current",
    "price",
    "weather",
    "stock",
    "score",
    "schedule",
    "release",
    "update",
    "search",
    "web",
    "internet",
    "2026",
    "\u4eca\u5929",
    "\u73b0\u5728",
    "\u6700\u65b0",
    "\u6700\u8fd1",
    "\u8fd1\u671f",
    "\u65b0\u95fb",
    "\u5f53\u524d",
    "\u76ee\u524d",
    "\u4ef7\u683c",
    "\u5929\u6c14",
    "\u80a1\u4ef7",
    "\u8d5b\u7a0b",
    "\u53d1\u5e03",
    "\u66f4\u65b0",
    "\u641c\u7d22",
    "\u8054\u7f51",
    "\u67e5\u4e00\u4e0b",
    "\u5b9e\u65f6",
)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    # Keep alphanumeric words and basic CJK characters as tokens.
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


def should_search_web(question: str) -> bool:
    normalized = question.lower()
    return any(hint in normalized for hint in WEB_SEARCH_HINTS)


@dataclass
class Chunk:
    source: str
    content: str
    tokens: set


class KnowledgeBase:
    def __init__(self, paths: Optional[List[str]] = None, chunk_size: int = 500):
        self.chunks: List[Chunk] = []
        self.chunk_size = chunk_size
        for path in (paths or []):
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if not text:
                continue
            self.add_document(os.path.basename(path), text)

    def add_document(self, name: str, content: str, chunk_size: Optional[int] = None) -> None:
        chunk_size = chunk_size or self.chunk_size
        text = content.strip()
        if not text:
            return
        for piece in self._chunk_text(text, chunk_size):
            self.chunks.append(
                Chunk(
                    source=name,
                    content=piece,
                    tokens=set(tokenize(piece)),
                )
            )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> List[str]:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        chunks: List[str] = []
        current = []
        current_len = 0
        for p in paragraphs:
            plen = len(p)
            if current and current_len + plen > chunk_size:
                chunks.append("\n".join(current))
                current = [p]
                current_len = plen
            else:
                current.append(p)
                current_len += plen
        if current:
            chunks.append("\n".join(current))
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Chunk]:
        if not self.chunks:
            return []
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []

        scored: List[Tuple[int, Chunk]] = []
        for c in self.chunks:
            overlap = len(q_tokens & c.tokens)
            if overlap > 0:
                scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


class DeepSeekWebSearchClient:
    """Web search backed by DeepSeek's own Search API.

    Endpoint: POST https://api.deepseek.com/v1/tools/search
    Guide:    https://api-docs.deepseek.com/zh-cn/guides/search

    The endpoint returns structured results (title / url / summary / content),
    so there is no model to pick: the search itself is served by DeepSeek's
    search engine and is billed per successful request (¥5 / 1000 requests).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        count: int = 5,
        need_content: bool = True,
        content_chars: int = 1000,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("deepseek_api_key_gty")
        )
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY or deepseek_api_key_gty.")

        base_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_SEARCH_URL
        self.api_key = api_key
        self.endpoint = base_url.rstrip("/") + DEEPSEEK_SEARCH_PATH
        # The API accepts 1..50 results per request.
        self.count = max(1, min(int(count), 50))
        self.need_content = need_content
        self.content_chars = content_chars
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))

    @staticmethod
    def _describe_error(response: "requests.Response") -> str:
        """Turn a non-200 response into a message a human can act on."""
        try:
            error = response.json().get("error") or {}
            message = error.get("message") or response.text
        except ValueError:
            message = response.text
        message = (message or "").strip()[:200]

        hints = {
            401: "API Key 无效",
            402: "账户余额不足",
            403: "请求被安全策略拒绝",
            429: "超出 QPS 限制",
        }
        hint = hints.get(response.status_code)
        suffix = f"，{hint}" if hint else ""
        return f"HTTP {response.status_code}{suffix}：{message}"

    def _request(
        self, query: str, ori_question: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query": query[:128],
            "count": self.count,
            "need_content": self.need_content,
        }
        # The original question helps the engine understand search intent.
        if ori_question:
            payload["ori_question"] = ori_question[:128]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        retryable = {408, 429, 500, 502, 503, 504}
        delay = 1.0
        last_error = "未知错误"
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = f"网络错误：{exc}"
            else:
                if response.status_code == 200:
                    return response.json().get("results") or []
                last_error = self._describe_error(response)
                if response.status_code not in retryable:
                    raise RuntimeError(last_error)
            if attempt < self.max_retries:
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(last_error)

    def search_results(self, question: str) -> List[Dict[str, Any]]:
        """Return the raw result dicts from the Search API."""
        return self._request(question, ori_question=question)

    def search(self, question: str) -> str:
        """Return the search results as a compact context block."""
        results = self.search_results(question)
        if not results:
            return ""

        blocks: List[str] = []
        for index, item in enumerate(results, start=1):
            lines = [f"[{index}] {item.get('title') or '(no title)'}"]
            if item.get("url"):
                lines.append(f"URL: {item['url']}")

            meta = [str(item[key]) for key in ("site_name", "date") if item.get(key)]
            if meta:
                lines.append(" | ".join(meta))

            summary = (item.get("summary") or "").strip()
            if summary:
                lines.append(summary)

            content = (item.get("content") or "").strip()
            if content and self.content_chars > 0:
                if len(content) > self.content_chars:
                    content = content[: self.content_chars].rstrip() + "…"
                lines.append(f"正文摘录：{content}")

            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


class QAAgent:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        kb: Optional[KnowledgeBase] = None,
        web_search_client: Optional[DeepSeekWebSearchClient] = None,
        web_search_mode: str = "auto",
        memory_turns: int = 6,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_Key_gty")
        if not api_key:
            raise RuntimeError(
                "Missing API key. Please set OPENAI_API_KEY or API_Key_gty."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.kb = kb
        self.web_search_client = web_search_client
        self.web_search_mode = web_search_mode
        self.memory_turns = memory_turns
        self.history: List[dict] = []
        self.system_prompt = (
            "You are an intelligent QA assistant. "
            "Answer clearly and accurately. "
            "Use provided web search findings for current information. "
            "When sources or URLs are available in the context, cite them briefly. "
            "If information is uncertain or missing, say so directly. "
            "Mirror the user's language."
        )

    def _build_context(self, question: str, web_context: str = "") -> str:
        sections = []
        if web_context:
            sections.append(f"Web search findings:\n{web_context}")

        if self.kb:
            chunks = self.kb.retrieve(question, top_k=3)
            if chunks:
                lines = ["Relevant knowledge snippets:"]
                for i, c in enumerate(chunks, start=1):
                    lines.append(f"[{i}] source={c.source}")
                    lines.append(c.content[:600])
                    lines.append("")
                sections.append("\n".join(lines).strip())

        return "\n\n".join(sections).strip()

    def _should_use_web(self, question: str, force_web_search: Optional[bool]) -> bool:
        if force_web_search is not None:
            return force_web_search
        if self.web_search_mode == "always":
            return True
        if self.web_search_mode == "off":
            return False
        return should_search_web(question)

    def ask(self, question: str, force_web_search: Optional[bool] = None) -> str:
        answer, _ = self.ask_with_meta(question, force_web_search)
        return answer

    def _prepare_request(
        self, question: str, force_web_search: Optional[bool]
    ) -> Tuple[List[dict], bool, str, str]:
        """Run optional web search and build chat messages.

        Returns (messages, attempted, web_context, web_search_error).
        """
        web_context = ""
        web_search_error = ""
        attempted = False
        if self._should_use_web(question, force_web_search):
            attempted = True
            if self.web_search_client is None:
                web_search_error = (
                    "Web search unavailable: missing DEEPSEEK_API_KEY or "
                    "deepseek_api_key_gty."
                )
            else:
                try:
                    web_context = self.web_search_client.search(question)
                except Exception as exc:
                    web_search_error = f"Web search unavailable: {exc}"

        context = self._build_context(question, web_context=web_context)
        content = question if not context else f"{context}\n\nQuestion:\n{question}"

        messages = [{"role": "system", "content": self.system_prompt}]
        if self.history:
            messages.extend(self.history[-self.memory_turns * 2 :])
        messages.append({"role": "user", "content": content})
        return messages, attempted, web_context, web_search_error

    def ask_with_meta(
        self, question: str, force_web_search: Optional[bool] = None
    ) -> Tuple[str, Dict[str, Any]]:
        messages, attempted, web_context, web_search_error = self._prepare_request(
            question, force_web_search
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
        )
        answer = response.choices[0].message.content or ""

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        meta: Dict[str, Any] = {
            "attempted": attempted,
            "searched": bool(web_context),
            "web_search_error": web_search_error,
        }
        if web_search_error:
            return f"{web_search_error}\n\n{answer}", meta
        return answer, meta

    def ask_stream_with_meta(
        self, question: str, force_web_search: Optional[bool] = None
    ) -> Iterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """Stream the answer incrementally.

        Yields (text_delta, None) for each chunk of generated text; the final
        yield is ("", meta) carrying the web-search metadata.
        """
        messages, attempted, web_context, web_search_error = self._prepare_request(
            question, force_web_search
        )
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            stream=True,
        )
        parts = []
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except (AttributeError, IndexError):
                delta = ""
            if delta:
                parts.append(delta)
                yield delta, None

        answer = "".join(parts)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        meta: Dict[str, Any] = {
            "attempted": attempted,
            "searched": bool(web_context),
            "web_search_error": web_search_error,
        }
        yield "", meta


def parse_args():
    parser = argparse.ArgumentParser(description="Simple intelligent QA agent")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "x-ai/grok-4.7"),
        help="Model name. Example: openai/gpt-5.2",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        help="API base URL",
    )
    parser.add_argument(
        "--kb",
        nargs="*",
        default=[],
        help="Optional local text files for retrieval (utf-8 preferred)",
    )
    parser.add_argument(
        "--web-search",
        choices=["auto", "always", "off"],
        default=os.getenv("WEB_SEARCH_MODE", "auto"),
        help="Web search mode",
    )
    parser.add_argument(
        "--search-count",
        type=int,
        default=env_int("DEEPSEEK_SEARCH_COUNT", 5),
        help="DeepSeek 搜索返回条数（1-50），默认 5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kb = KnowledgeBase(args.kb) if args.kb else None
    web_search_client = None
    if args.web_search != "off":
        try:
            web_search_client = DeepSeekWebSearchClient(count=args.search_count)
        except RuntimeError as exc:
            print(f"Search disabled: {exc}")
            web_search_client = None

    agent = QAAgent(
        model=args.model,
        base_url=args.base_url,
        kb=kb,
        web_search_client=web_search_client,
        web_search_mode=args.web_search,
    )

    print("QA Agent is running.")
    print("Commands: /exit to quit, /clear to clear memory")
    print("Search: auto by default, /search to force, /nosearch to skip once")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            break
        if question.lower() == "/clear":
            agent.history.clear()
            print("Memory cleared.")
            continue

        force_web_search = None
        lowered = question.lower()
        if lowered == "/search":
            print("Please enter a question after /search.")
            continue
        if lowered.startswith("/search "):
            question = question[len("/search ") :].strip()
            force_web_search = True
        elif lowered == "/nosearch":
            print("Please enter a question after /nosearch.")
            continue
        elif lowered.startswith("/nosearch "):
            question = question[len("/nosearch ") :].strip()
            force_web_search = False
        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = agent.ask(question, force_web_search=force_web_search)
            print(f"\nAgent: {answer}")
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
