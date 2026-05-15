"""BanyanTree API — OpenAI-compatible FastAPI server."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.config import get, load_config
from app.api.llm_client import LLMClient
from app.api.orchestrator import FinEdgeAPIOrchestrator, GuardrailError
from agents.tools.document_store import save_uploaded_document
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatResponseChunk,
    DeltaContent,
    HealthResponse,
    MessageContent,
    Choice,
    ModelCard,
    ModelList,
    StreamChoice,
)

# ── App setup ─────────────────────────────────────────────────────────────────

load_config()  # eagerly expand env vars at startup

app = FastAPI(
    title="BanyanTree FinEdge API",
    version=get("app.version", "0.1.0"),
    description="OpenAI-compatible API wrapper around the BanyanTree fine-tuned LLM with agentic tool use.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get("api.cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = FinEdgeAPIOrchestrator()
_llm_client = LLMClient()

# Patterns that identify OpenWebUI internal system prompts (follow-up generation,
# title generation, etc.). These must bypass BanyanTree and go straight to the LLM
# — running them through the RAG pipeline causes garbage symbol extraction loops.
_OPENWEBUI_BYPASS_MARKERS = (
    "### Task:",
    "### Chat History:",
    '"follow_ups"',
    "Generate a concise, 3-5 word title",
    "Identify the primary language",
)

# ── Auth dependency ────────────────────────────────────────────────────────────


def _verify_api_key(authorization: str = Header(default="")) -> None:
    expected = get("api.api_key", "")
    if not expected:
        return  # auth disabled
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(
        status="ok",
        version=get("app.version", "0.1.0"),
        model=get("llm.model", ""),
    )


@app.post("/v1/embeddings", dependencies=[Depends(_verify_api_key)], response_model=None)
async def embeddings(request: dict) -> dict:
    """Stub embeddings endpoint — satisfies OpenWebUI's RAG_EMBEDDING_ENGINE=openai check.

    Returns zero vectors so OpenWebUI starts without needing a real embedding model.
    """
    inputs = request.get("input", [])
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [
        {"object": "embedding", "index": i, "embedding": [0.0] * 384}
        for i in range(len(inputs))
    ]
    return {"object": "list", "data": data, "model": request.get("model", "text-embedding-ada-002")}



@app.post("/v1/documents/upload", dependencies=[Depends(_verify_api_key)])
async def upload_document(file: UploadFile = File(...), session_id: str = Form(default="default")) -> dict:
    """Store a private user document for later MCP document-tool analysis."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")
    return save_uploaded_document(file.filename or "document", content, session_id=session_id)

@app.get("/v1/models", response_model=ModelList, dependencies=[Depends(_verify_api_key)])
async def list_models() -> ModelList:
    """Return available models (OpenAI-compatible)."""
    model_id = get("llm.model", "finedge")
    return ModelList(data=[ModelCard(id=model_id)])


@app.post("/v1/chat/completions", dependencies=[Depends(_verify_api_key)], response_model=None)
async def chat_completions(request: ChatRequest) -> StreamingResponse | ChatResponse:
    """Main chat endpoint — OpenAI-compatible, supports streaming.

    Args:
        request: ChatRequest with model, messages, and optional stream flag.

    Returns:
        StreamingResponse (SSE) when ``stream=true``, else ChatResponse JSON.
    """
    session_id = str(uuid.uuid4())  # stateless: fresh session per request
    model = get("llm.model", request.model)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    logger.info(f"[{request_id}] chat request — messages={len(request.messages)} stream={request.stream}")

    # Detect OpenWebUI internal prompts (follow-up generation, title generation, etc.)
    # and bypass BanyanTree entirely — send straight to LLM to avoid garbage symbol loops.
    last_content = (request.messages[-1].content if request.messages else "") or ""
    if any(marker in last_content for marker in _OPENWEBUI_BYPASS_MARKERS):
        logger.info(f"[{request_id}] OpenWebUI internal prompt detected — bypassing BanyanTree")
        messages = [{"role": m.role, "content": m.content or ""} for m in request.messages]
        reply = await _llm_client.complete(messages, max_tokens=512)
        return ChatResponse(
            id=request_id,
            model=model,
            choices=[Choice(message=MessageContent(role="assistant", content=reply), finish_reason="stop")],
        )

    if request.stream:
        return StreamingResponse(
            _stream_sse(request, session_id, request_id, model, request.document_ids),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: collect all chunks
    parts: list[str] = []
    try:
        async for chunk in _orchestrator.stream_response(request.messages, session_id, request.document_ids):
            parts.append(chunk)
    except GuardrailError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    full_text = "".join(parts)
    return ChatResponse(
        id=request_id,
        model=model,
        choices=[
            Choice(
                message=MessageContent(role="assistant", content=full_text),
                finish_reason="stop",
            )
        ],
    )


async def _stream_sse(
    request: ChatRequest,
    session_id: str,
    request_id: str,
    model: str,
    document_ids: list[str] | None = None,
):
    """Async generator that yields SSE-formatted chunks."""
    try:
        async for text_chunk in _orchestrator.stream_response(request.messages, session_id, document_ids):
            chunk = ChatResponseChunk(
                id=request_id,
                model=model,
                choices=[StreamChoice(delta=DeltaContent(content=text_chunk))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    except GuardrailError as exc:
        error_payload = json.dumps({"error": {"message": str(exc), "type": "guardrail_error"}})
        yield f"data: {error_payload}\n\n"
    except Exception as exc:
        logger.error(f"[{request_id}] Unhandled error during streaming: {exc}")
        error_payload = json.dumps({"error": {"message": "Internal server error", "type": "server_error"}})
        yield f"data: {error_payload}\n\n"
    finally:
        # Send stop chunk
        stop_chunk = ChatResponseChunk(
            id=request_id,
            model=model,
            choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
        )
        yield f"data: {stop_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
