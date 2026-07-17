"""Integration test verifying the multi-agent supervisor assembles correctly,
using a fake chat model so no API key or network call is required.

Note: FakeListLLM does NOT work here because create_tool_calling_agent requires
an LLM with .bind_tools(), which only chat models implement."""
from app.agents.supervisor import build_supervisor_executor
from tests.fakes import FakeToolCallingChatModel


def test_supervisor_builds_with_two_sub_agent_tools():
    executor = build_supervisor_executor(FakeToolCallingChatModel())
    tool_names = {tool.name for tool in executor.tools}
    assert tool_names == {"diagnostic_agent", "scheduling_agent"}
