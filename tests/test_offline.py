"""
Offline validation tests - run with:  pytest -q

These tests need NO API key. A small scripted fake LLM stands in for Gemini/OpenAI so I
can check the parts I built myself: the tools, the router's conditional edges, the
tool-calling loop inside each agent, and MemorySaver keeping state across turns/threads.
"""

import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

import tools as T  # noqa: E402
from chatbot import CustomerServiceChatbot, RouteDecision  # noqa: E402

ORDER_RE = re.compile(r"ORD-\d{4}", re.I)


class FakeLLM(BaseChatModel):
    """Deterministic stand-in for a real chat model."""

    tool_names: list = []

    @property
    def _llm_type(self):
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return FakeLLM(tool_names=[t.name for t in tools])

    def with_structured_output(self, schema, **kwargs):
        def route(messages):
            text = messages[-1].content.lower()
            if any(w in text for w in ("refund", "return", "cancel")):
                r = "REFUND_POLICY"
            elif "ord-" in text or "order" in text or "arrive" in text:
                r = "ORDER_STATUS"
            elif any(w in text for w in ("laptop", "recommend", "cost", "price", "budget")):
                r = "PRODUCT_RECOMMENDATION"
            else:
                r = "GENERAL"
            return RouteDecision(route=r, reason="keyword match")

        return RunnableLambda(route)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if isinstance(last, ToolMessage):  # second LLM step: answer from the tool result
            msg = AIMessage(content=f"Here is what I found: {last.content}")
        else:
            # Find the most recent order ID anywhere in the history (tests memory)
            ids = [m for msg in messages for m in ORDER_RE.findall(str(msg.content))]
            order_id = ids[-1].upper() if ids else None
            call = None
            if "get_order_details" in self.tool_names and order_id:
                call = ("get_order_details", {"order_id": order_id})
            elif "check_refund_eligibility" in self.tool_names and order_id:
                call = ("check_refund_eligibility", {"order_id": order_id})
            elif "list_products" in self.tool_names:
                call = ("list_products", {"max_price": 1600, "max_shipping_days": 0})
            if call:
                msg = AIMessage(content="", tool_calls=[{"name": call[0], "args": call[1], "id": str(uuid.uuid4())}])
            else:
                names = [m for msg in messages for m in re.findall(r"my name is (\w+)", str(msg.content), re.I)]
                msg = AIMessage(content=f"Hi {names[-1] if names else 'there'}! How can I help?")
        return ChatResult(generations=[ChatGeneration(message=msg)])


# ------------------------------------------------------------------ tool tests
def test_order_lookup_and_not_found():
    assert "SpectraBook S" in T.get_order_details.invoke({"order_id": "ord-8276"})
    assert T.get_order_details.invoke({"order_id": "ORD-9999"}) == "ORDER_NOT_FOUND"


def test_quantity_update_rules():
    assert "UPDATE_REJECTED" in T.update_order_quantity.invoke({"order_id": "ORD-8276", "new_quantity": 4})  # delivered
    assert "UPDATED" in T.update_order_quantity.invoke({"order_id": "ORD-2050", "new_quantity": 3})  # in transit


def test_refund_eligibility_cases():
    assert T.check_refund_eligibility.invoke({"order_id": "ORD-3391"}).startswith("ELIGIBLE")  # 22 days, 30-day window
    assert T.check_refund_eligibility.invoke({"order_id": "ORD-5520"}).startswith("NOT ELIGIBLE")  # gaming, 25 > 15
    assert T.check_refund_eligibility.invoke({"order_id": "ORD-1182"}).startswith("NOT ELIGIBLE")  # 51 days
    assert "NOT been delivered" in T.check_refund_eligibility.invoke({"order_id": "ORD-6948"})


def test_product_filters():
    out = T.list_products.invoke({"max_price": 1500, "max_shipping_days": 0})
    assert "AlphaBook Pro" in out and "SpectraBook" not in out
    assert "64GB" in T.get_product_details.invoke({"laptop_name": "spectra"})


# ----------------------------------------------------------------- graph tests
def test_routes_to_all_three_specialists():
    bot = CustomerServiceChatbot(FakeLLM())
    t = str(uuid.uuid4())
    assert bot.chat("What is the status of order ORD-7311?", t)[0] == "ORDER_STATUS"
    assert bot.chat("Can I get a refund on it?", t)[0] == "REFUND_POLICY"
    assert bot.chat("Recommend a laptop under my budget", t)[0] == "PRODUCT_RECOMMENDATION"
    assert bot.chat("Thanks, bye", t)[0] == "GENERAL"


def test_memory_saver_keeps_context_across_turns_and_agents():
    bot = CustomerServiceChatbot(FakeLLM())
    t = str(uuid.uuid4())
    bot.chat("Hi, my name is Priya", t)                              # turn 1 (General)
    _, r2 = bot.chat("Where is my order ORD-3391?", t)               # turn 2 (Order agent)
    assert "NanoEdge Flex" in r2
    route, r3 = bot.chat("Can I return it?", t)                      # turn 3: no ID given!
    assert route == "REFUND_POLICY" and "ORD-3391" in r3 and "ELIGIBLE" in r3
    _, r4 = bot.chat("What's my name again?", t)                     # turn 4: recalls turn 1
    assert "Priya" in r4
    assert len(bot.history(t)) == 8                                  # 4 user + 4 bot messages stored


def test_threads_are_isolated():
    bot = CustomerServiceChatbot(FakeLLM())
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    bot.chat("Hi, my name is Priya", a)
    _, reply = bot.chat("What's my name?", b)
    assert "Priya" not in reply
