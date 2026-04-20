import http.client
import json
import os
import time
import traceback
from typing import Any, List, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retrieval.retriever import RAGInterface
from prompt_template import build_prompt_raft


class HttpsApi:
    def __init__(self, host: str, key: str, model: str, timeout: int = 20, **kwargs: Any) -> None:
        """Simple HTTPS API client for OpenAI-compatible chat completions."""
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args: Any, **kwargs: Any) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 4096),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt,
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'IRAG-Frontend/1.0',
                    'Content-Type': 'application/json',
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)
                return data['choices'][0]['message']['content']
            except Exception:
                print(
                    f'Error when calling LLM API: {traceback.format_exc()}.'
                    f'You may check your API host and API key.'
                )
                time.sleep(2)
                continue


# ── 合同名提取 ─────────────────────────────────────────────────────────
def _extract_contract_name(source: str) -> str:
    """从 metadata.source（PDF 路径）提取合同名，如 'sourcepdf/A/B/name.pdf' → 'name'"""
    return os.path.splitext(os.path.basename(source or "未知合同"))[0] or "未知合同"


# ── 支持本地 HTTP（非 HTTPS）的推理客户端 ──────────────────────────────
class LocalHttpApi:
    """与 HttpsApi 接口相同，但使用 HTTPConnection（适用于 localhost）"""

    def __init__(self, base_url: str, model: str, timeout: int = 60, **kwargs):
        url = base_url.rstrip("/")
        _scheme, rest = url.split("://", 1)
        if ":" in rest:
            host, port_str = rest.rsplit(":", 1)
            self._host = host
            self._port = int(port_str)
        else:
            self._host = rest
            self._port = 80
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs

    def draw_sample(self, prompt, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt.strip()}]
        payload = json.dumps({
            "model": self._model,
            "messages": prompt,
            "max_tokens": self._kwargs.get("max_tokens", 512),
            "temperature": self._kwargs.get("temperature", 0.2),
            "top_p": self._kwargs.get("top_p", 0.9),
        })
        headers = {"Content-Type": "application/json"}
        while True:
            try:
                conn = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = json.loads(res.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"LocalHttpApi error: {e}")
                time.sleep(2)


# -----------------------------
# FastAPI app & global deps
# -----------------------------

app = FastAPI(title="IRAG QA API")

# Mount static frontend assets directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 15
    mode: str = "raft"
    history: List[Message] = []


class RefChunk(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    refs: List[RefChunk]


# Initialize RAG + LLM client once at startup
rag = RAGInterface()

_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
_LLM_KEY      = os.environ.get("LLM_KEY", "")
_LLM_MODEL    = os.environ.get("LLM_MODEL", "qwen3-raft")

if _LLM_BASE_URL.startswith("http://"):
    http_client = LocalHttpApi(base_url=_LLM_BASE_URL, model=_LLM_MODEL)
else:
    _host = _LLM_BASE_URL.replace("https://", "") or "api.bltcy.top"
    http_client = HttpsApi(host=_host, key=_LLM_KEY, model=_LLM_MODEL)
LLM_CACHE: Dict[str, str] = {}


@app.get("/")
async def index() -> FileResponse:
    """Serve the Vue frontend."""
    return FileResponse("frontend/index.html")


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Main QA endpoint for the frontend.

    1) 基于最近若干轮用户问题 + 当前问题进行 RAG 检索；
    2) 用 prompt_template 生成当前轮 Prompt；
    3) 将历史对话 + 当前 Prompt 一起发送给 LLM 生成答案。
    """
    # 1) 构建用于检索的 query（最近若干轮用户问题 + 当前问题）
    history_user_questions = [
        m.content for m in (req.history or []) if m.role == "user"
    ]
    recent_user_questions = history_user_questions[-3:]  # 只取最近 3 轮用户问题
    rag_query_parts = recent_user_questions + [req.question]
    rag_query = "\n".join(rag_query_parts)

    # 2) RAG 检索
    results = rag.retrieve(rag_query, top_k=req.top_k)

    # 3-5) 构建 messages（raft 模式使用专用格式，其他模式沿用原有逻辑）
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in (req.history or [])
        if m.role in {"user", "assistant", "system"}
    ]

    raft_chunks = [
        {
            "contract_name": _extract_contract_name(r["metadata"].get("source", "")),
            "text": r.get("text") or "",
        }
        for r in results
        if r.get("text")
    ]
    system_prompt, user_message = build_prompt_raft(req.question, raft_chunks)
    messages = [{"role": "system", "content": system_prompt}] + \
               history_messages + \
               [{"role": "user", "content": user_message}]

    # 6) 调用 LLM（带简单缓存：仅在无 history 时缓存同一问题的回答）
    cache_key = None
    if not req.history:
        cache_key = json.dumps(
            {"q": req.question, "mode": req.mode, "top_k": req.top_k},
            ensure_ascii=False,
            sort_keys=True,
        )
    if cache_key is not None and cache_key in LLM_CACHE:
        answer_text = LLM_CACHE[cache_key]
    else:
        answer_text = http_client.draw_sample(prompt=messages)
        if cache_key is not None:
            LLM_CACHE[cache_key] = answer_text

    # 7) 返回统一结构
    converted_refs = [RefChunk(**r) for r in results]
    return AskResponse(answer=answer_text, refs=converted_refs)
