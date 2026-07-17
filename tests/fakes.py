"""
Test-only fake chat model that supports .bind_tools(), unlike LangChain's
built-in FakeListLLM (completion-style, no tool support) or
GenericFakeChatModel (implements bind_tools but raises NotImplementedError).

create_tool_calling_agent() calls llm.bind_tools(tools) at construction time,
so any fake used to test agent *assembly* (not full execution) needs a working
bind_tools that just returns something invokable — the tests here only check
that the executor and its tool list are built correctly, they never actually
call the model.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool


class FakeToolCallingChatModel(BaseChatModel):
    """Minimal chat model stand-in: always replies with a fixed text message
    and accepts (and ignores) tool bindings, so agent construction succeeds
    without a real API call."""

    response_text: str = "I don't know."

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.response_text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "FakeToolCallingChatModel":
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling-chat-model"
