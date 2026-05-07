from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .backends.llm.anthropic_llm import AnthropicLLM
from .models import QueryRequest, QueryResponse
from .rag_pipeline import RAGPipeline

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Both pipelines are created once at startup — no overhead per request
PIPELINES = {
    "local": RAGPipeline(config_name="local-phi3-chroma"),
    "anthropic": RAGPipeline(llm=AnthropicLLM(), config_name="anthropic-haiku-chroma"),
}


@router.get("/health")
def health() -> dict:
    """Return service health status and available backends.

    Returns:
        Dict with status and list of available backends.
    """
    return {"status": "ok", "backends": list(PIPELINES.keys())}


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Serve the chat UI.

    Args:
        request: Incoming FastAPI request.

    Returns:
        Rendered index.html template.
    """
    return templates.TemplateResponse(request, "index.html")


@router.post("/ask", response_model=QueryResponse)
def ask(query: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline and return an answer.

    Args:
        query: Validated request with question, max_contexts, and backend.
               backend is 'local' (Phi-3) or 'anthropic' (Haiku). Defaults to 'local'.

    Returns:
        QueryResponse with answer, retrieved contexts, config name, and latency.
    """
    pipeline = PIPELINES[query.backend]
    result = pipeline.answer_question(query.question, max_contexts=query.max_contexts)
    return QueryResponse(
        answer=result.answer,
        contexts=result.contexts,
        config_name=result.config_name,
        total_time_ms=result.total_time_ms,
    )
