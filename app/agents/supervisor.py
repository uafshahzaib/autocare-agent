"""
Multi-agent AutoCare architecture: a Supervisor agent coordinates two
specialist sub-agents, using the "agents-as-tools" pattern (each sub-agent is
itself wrapped as a callable tool that the supervisor's LLM can invoke).

This mirrors the orchestration pattern popularized by Microsoft's AutoGen
(Wu et al., 2023, "AutoGen: Enabling Next-Gen LLM Applications via
Multi-Agent Conversation") and is the natural next step up from a single
flat agent once the tool count and responsibilities grow: instead of one
LLM call juggling five tools and their interactions, each sub-agent owns a
narrow, well-defined responsibility, and the supervisor only has to decide
*which specialist* the user's request belongs to (and combine their answers).

    Supervisor
    ├── diagnostic_agent   -> "what's wrong / what does this mean?"
    │                          tools: RAG search, image classification, warning-light lookup
    └── scheduling_agent   -> "what should I do about it, and how urgent is it?"
                               tools: service-interval calculator, risk triage

See app/agent.py for the simpler single-agent version this project started
from — both are kept so you can compare the two architectures directly.
"""
from __future__ import annotations

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools import get_diagnostic_tools, get_scheduling_tools

DIAGNOSTIC_SYSTEM_PROMPT = """You are the Diagnostic specialist inside AutoCare.
Your job is to figure out what's actually wrong or what something means -
never to recommend actions or timelines, that is the Scheduling specialist's job.

Rules:
- Ground every factual claim in the search_maintenance_docs tool. Do not invent maintenance facts.
- If the user gives you an image path instead of naming a warning light, use
  classify_warning_light_image first, then lookup_warning_light with the returned label.
- Be concise and precise about what a symptom or warning light indicates.
"""

SCHEDULING_SYSTEM_PROMPT = """You are the Scheduling & Risk specialist inside AutoCare.
Your job is to turn facts (mileage, an active warning light) into a concrete
recommendation: is anything due, how urgent is it, and what should the driver do next.

Rules:
- Use check_service_interval for any pure mileage/service-due question.
- Use assess_maintenance_risk whenever mileage AND a warning light are both relevant,
  or whenever the user wants an overall "what should I do" recommendation.
- Always state a clear priority and next action; do not hedge.
"""

SUPERVISOR_SYSTEM_PROMPT = """You are AutoCare, a vehicle maintenance assistant for owners
and service advisors. You do not answer directly — you have two specialist agents available
and your job is to route the user's request to the right one (or both, and then combine
their answers into one coherent reply):

- diagnostic_agent: for "what does X mean / what's wrong" questions (symptoms, warning
  lights, images of warning lights, general maintenance facts).
- scheduling_agent: for "is something due / what should I do / how urgent is this" questions
  (mileage-based service checks, overall risk/priority assessment).

Many real questions need both: first diagnose, then decide what to do about it. Call both
specialists in that case and combine their answers into a single, clear response for the user.
"""


def _build_sub_agent_executor(
    llm: BaseLanguageModel, tools: list[StructuredTool], system_prompt: str
) -> AgentExecutor:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)


class SubAgentQuery(BaseModel):
    request: str = Field(description="The user's request or sub-question to hand to this specialist.")


def _wrap_agent_as_tool(executor: AgentExecutor, name: str, description: str) -> StructuredTool:
    def _invoke(request: str) -> str:
        result = executor.invoke({"input": request})
        return result["output"]

    return StructuredTool.from_function(func=_invoke, name=name, description=description, args_schema=SubAgentQuery)


def build_supervisor_executor(llm: BaseLanguageModel, verbose: bool = False) -> AgentExecutor:
    """Assembles the full multi-agent system: two specialist sub-agents wrapped
    as tools for a top-level supervisor agent. Same LLM is reused for all three
    agents here for simplicity; in production each could use a different
    model sized to its task (e.g. a smaller/cheaper model for the supervisor's
    routing decision)."""
    diagnostic_executor = _build_sub_agent_executor(llm, get_diagnostic_tools(), DIAGNOSTIC_SYSTEM_PROMPT)
    scheduling_executor = _build_sub_agent_executor(llm, get_scheduling_tools(), SCHEDULING_SYSTEM_PROMPT)

    diagnostic_tool = _wrap_agent_as_tool(
        diagnostic_executor,
        name="diagnostic_agent",
        description="Ask the diagnostic specialist what a symptom, image, or warning light means.",
    )
    scheduling_tool = _wrap_agent_as_tool(
        scheduling_executor,
        name="scheduling_agent",
        description="Ask the scheduling/risk specialist whether something is due and what to do about it.",
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    supervisor_agent = create_tool_calling_agent(llm, [diagnostic_tool, scheduling_tool], prompt)
    return AgentExecutor(
        agent=supervisor_agent,
        tools=[diagnostic_tool, scheduling_tool],
        verbose=verbose,
        handle_parsing_errors=True,
    )


def run(message: str, llm: BaseLanguageModel) -> str:
    executor = build_supervisor_executor(llm)
    result = executor.invoke({"input": message})
    return result["output"]
