"""FastAPI service exposing the AutoCare agent over HTTP."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import get_default_llm
from app.agent import run as run_single_agent
from app.agents.supervisor import run as run_supervisor

app = FastAPI(
    title="AutoCare Agentic Assistant",
    description=(
        "Multi-agent LangChain system (Supervisor -> Diagnostic + Scheduling sub-agents) "
        "with RAG, PyTorch computer vision, and deterministic tools for vehicle maintenance Q&A."
    ),
    version="2.0.0",
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    architecture: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Routes through the multi-agent Supervisor architecture by default."""
    try:
        reply = run_supervisor(request.message, llm=get_default_llm())
        return ChatResponse(reply=reply, architecture="multi-agent-supervisor")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/simple", response_model=ChatResponse)
def chat_simple(request: ChatRequest) -> ChatResponse:
    """Routes through the original single flat agent — kept for comparison."""
    try:
        reply = run_single_agent(request.message, llm=get_default_llm())
        return ChatResponse(reply=reply, architecture="single-flat-agent")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
