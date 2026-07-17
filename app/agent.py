"""
Assembles the AutoCare agent: an LLM + a set of tools (RAG, calculator,
structured lookup) combined via LangChain's tool-calling agent executor.

The LLM is injected rather than hard-coded, so the same agent-construction
logic can run against OpenAI in production and a FakeListLLM in tests —
this is what makes the agent's control flow unit-testable without hitting
a real model or paying for API calls in CI.
"""
from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.tools import get_all_tools

SYSTEM_PROMPT = """You are AutoCare, an assistant for vehicle owners and service advisors.

Rules:
- Ground every factual maintenance claim in the search_maintenance_docs tool. Do not invent maintenance facts.
- Use check_service_interval whenever mileage numbers are given, instead of estimating yourself.
- Use lookup_warning_light whenever a dashboard light or icon is mentioned.
- If a question needs more than one tool, call them in sequence and combine the results.
- Be concise and safety-conscious: flag anything urgent (red warning lights, overdue brake service) clearly.
"""


def build_agent_executor(llm: BaseLanguageModel, verbose: bool = False) -> AgentExecutor:
    """Wire an LLM together with the AutoCare tool set into a runnable agent."""
    tools = get_all_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose, handle_parsing_errors=True)


def get_default_llm() -> BaseLanguageModel:
    """
    Production default LLM, chosen from environment variables so no code change
    is needed to switch providers. Set LLM_PROVIDER=anthropic (or =openai)
    explicitly, or just set whichever API key you have — Anthropic is used if
    ANTHROPIC_API_KEY is present, otherwise OpenAI. Note: a claude.ai Pro
    subscription is separate from Anthropic API billing — you need a key
    from console.anthropic.com, not your claude.ai login.
    """
    from app.config import (
        ANTHROPIC_API_KEY,
        ANTHROPIC_MODEL,
        LLM_PROVIDER,
        OPENAI_API_KEY,
        OPENAI_MODEL,
    )

    provider = LLM_PROVIDER or ("anthropic" if ANTHROPIC_API_KEY else "openai")

    if provider == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Get one from console.anthropic.com "
                "(a claude.ai Pro subscription does not include API access) and add "
                "it to your .env file, or pass a different LLM instance to "
                "build_agent_executor()."
            )
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY, temperature=0)

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file, or pass a "
            "different LLM instance to build_agent_executor()."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)


def run(message: str, llm: BaseLanguageModel | None = None) -> str:
    executor = build_agent_executor(llm or get_default_llm())
    result = executor.invoke({"input": message})
    return result["output"]
