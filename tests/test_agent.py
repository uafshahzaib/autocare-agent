"""
Integration test for agent assembly using a fake chat model, so the agent's
construction/wiring is verified without any real API key or network call.

Note: FakeListLLM (a plain completion-style fake) does NOT work here because
create_tool_calling_agent requires an LLM with a .bind_tools() method, which
only chat models implement. GenericFakeChatModel does implement it.
"""
from app.agent import build_agent_executor
from app.tools import get_all_tools
from tests.fakes import FakeToolCallingChatModel


def test_agent_executor_builds_with_all_tools():
    executor = build_agent_executor(FakeToolCallingChatModel())
    tool_names = {tool.name for tool in executor.tools}
    assert tool_names == {
        "search_maintenance_docs",
        "classify_warning_light_image",
        "lookup_warning_light",
        "check_service_interval",
        "assess_maintenance_risk",
    }


def test_all_tools_have_descriptions():
    for tool in get_all_tools():
        assert tool.description
        assert tool.name
