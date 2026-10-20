"""
chatbot.py - TechNova Customer Service Chatbot (multi-agent, LangGraph)

How it works
------------
                        +--> Order Status Agent ---------+
                        |                                 |
  user -> Router Agent -+--> Refund Policy Agent --------+--> reply (END)
                        |                                 |
                        +--> Product Recommendation Agent+
                        |                                 |
                        +--> General / Small Talk Agent --+

1. The Router Agent reads the latest message (plus recent history) and picks ONE
   route using structured output: ORDER_STATUS, REFUND_POLICY,
   PRODUCT_RECOMMENDATION or GENERAL.
2. A conditional edge sends the conversation to that specialist agent.
3. Each specialist is its own small LangGraph (LLM <-> tools loop) with its own
   system prompt and its own tools.
4. The parent graph is compiled with LangGraph's MemorySaver checkpointer, so the
   whole conversation is saved per thread_id. Every agent sees the full history,
   which is how "it", "that order" or "my name" work across turns and agents.

Run an interactive chat:   python chatbot.py
"""

import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from tools import ORDER_TOOLS, PRODUCT_TOOLS, REFUND_TOOLS, STORE_TODAY

# =============================================================================
# 1. SYSTEM PROMPTS
# =============================================================================

# Shared persona: defines the chatbot's ROLE. Every agent's prompt starts with it,
# so the bot sounds like one consistent customer service assistant.
BASE_SYSTEM_PROMPT = f"""You are Nova, the friendly and professional customer service and
sales assistant for TechNova Laptops, an online laptop store.
Your job is to help customers with their orders, refunds and returns, and with choosing
the right laptop. Be warm, concise (2-5 sentences unless listing items) and accurate.
Rules:
- Only use facts from your tools or from earlier in this conversation. Never invent
  order details, prices, specs or policy rules.
- Never reveal information about an order other than the one the customer asks about.
- Use the customer's name if they have shared it.
- Today's date is {STORE_TODAY.isoformat()}.
"""

ORDER_AGENT_PROMPT = BASE_SYSTEM_PROMPT + """
You are the ORDER STATUS specialist.
- Use get_order_details to look up an order. If the customer refers to "my order" or
  "it", use the order ID mentioned earlier in the conversation.
- If no order ID is known yet, ask for it (format ORD-1234).
- Use update_order_quantity to change quantities. Compute the new total yourself
  (e.g. "add one more" to a quantity of 3 means new_quantity=4) and confirm the result.
- If a tool returns ORDER_NOT_FOUND, say so politely and ask the customer to double-check.
"""

REFUND_AGENT_PROMPT = BASE_SYSTEM_PROMPT + """
You are the REFUND & RETURNS specialist.
- For general policy questions, call get_refund_policy and answer from it.
- For a specific order, call check_refund_eligibility (reuse an order ID from earlier
  in the conversation if the customer says "it" or "my order").
- Only call create_return_request after the customer clearly confirms they want to
  return the item, and include their reason. Share the RMA number it returns.
- Mention relevant fees or timelines from the policy (restocking fee, return shipping,
  5-7 business day refund timing).
"""

PRODUCT_AGENT_PROMPT = BASE_SYSTEM_PROMPT + """
You are the PRODUCT RECOMMENDATION (sales) specialist.
- Use list_products to see the catalog (filter by budget or shipping time if given)
  and get_product_details for specs of one laptop.
- Recommend the 1-2 best matches for the customer's needs and budget and explain why.
- If the customer's needs are unclear, ask one short question (use case or budget).
- Remember preferences the customer stated earlier (budget, use case) and keep using them.
- We only sell the laptops in the catalog; politely say so if asked about other brands.
"""

GENERAL_AGENT_PROMPT = BASE_SYSTEM_PROMPT + """
You are handling GENERAL conversation: greetings, thanks, goodbyes, questions about what
you can do, and questions about earlier parts of this conversation (for example "what is
my name?" or "what did I ask about?") - answer those from the conversation history.
Mention that you can help with order status, refunds and returns, and laptop
recommendations. For unrelated requests, politely explain you can only help with
TechNova topics.
"""

ROUTER_PROMPT = """You are the router for a laptop store's customer service chatbot.
Read the conversation and choose which specialist should answer the customer's LATEST
message:
- ORDER_STATUS: order status, delivery dates, tracking, order details, changing quantity.
- REFUND_POLICY: refunds, returns, cancellations, return windows, fees, "can I send it back".
- PRODUCT_RECOMMENDATION: which laptop to buy, features, specs, prices, comparisons, shipping time of a product.
- GENERAL: greetings, thanks, goodbyes, small talk, questions about the conversation itself
  (e.g. "what's my name?"), or anything off-topic.
Use earlier messages to resolve follow-ups: "how much is it?" after discussing a laptop is
PRODUCT_RECOMMENDATION; "can I return it?" is REFUND_POLICY.
Short follow-ups like "what about order ORD-1234?" or "and that one?" continue the SAME topic
as the previous customer message (e.g. after asking about return eligibility, "what about
order X?" means return eligibility for X, so REFUND_POLICY)."""

Route = Literal["ORDER_STATUS", "REFUND_POLICY", "PRODUCT_RECOMMENDATION", "GENERAL"]

AGENT_NAMES = {
    "ORDER_STATUS": "Order Status Agent",
    "REFUND_POLICY": "Refund Policy Agent",
    "PRODUCT_RECOMMENDATION": "Product Recommendation Agent",
    "GENERAL": "General Agent",
}


class RouteDecision(BaseModel):
    """Structured output the router LLM must return."""

    route: Route = Field(description="Which specialist agent should handle the latest message")
    reason: str = Field(description="One short sentence explaining the choice")


# =============================================================================
# 2. STATE
# =============================================================================
class ChatState(TypedDict):
    """State saved by MemorySaver for each conversation thread."""

    messages: Annotated[list[AnyMessage], add_messages]  # full chat history (appended each turn)
    route: str  # which agent handled the latest turn (for display / debugging)


class AgentState(TypedDict):
    """Private working state inside one specialist agent (includes its tool calls)."""

    messages: Annotated[list[AnyMessage], add_messages]


def message_text(message) -> str:
    """Get plain text from a message (Gemini can return a list of content parts)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
    return "".join(parts).strip()


# =============================================================================
# 3. SPECIALIST AGENT (one class, instantiated 4 times with different prompts/tools)
# =============================================================================
class SpecialistAgent:
    """A ReAct-style agent: LLM decides -> optionally calls tools -> LLM answers.

    Built as its own LangGraph:  agent_llm --(tool calls?)--> tools --> agent_llm ... --> END
    """

    def __init__(self, name: str, model, system_prompt: str, tools: list):
        self.name = name
        self.system_prompt = system_prompt
        # Give the LLM the schemas of ONLY this agent's tools
        self.llm = model.bind_tools(tools) if tools else model

        graph = StateGraph(AgentState)
        graph.add_node("agent_llm", self.call_llm)
        graph.add_edge(START, "agent_llm")
        if tools:
            graph.add_node("tools", ToolNode(tools))  # runs whichever tools the LLM asked for
            graph.add_conditional_edges("agent_llm", self.needs_tool, {True: "tools", False: END})
            graph.add_edge("tools", "agent_llm")  # send tool results back to the LLM
        else:
            graph.add_edge("agent_llm", END)
        # No checkpointer here: conversation memory lives in the parent graph
        self.graph = graph.compile()

    def call_llm(self, state: AgentState):
        # Put this agent's system prompt in front of the conversation history
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        return {"messages": [self.llm.invoke(messages)]}

    @staticmethod
    def needs_tool(state: AgentState) -> bool:
        return bool(getattr(state["messages"][-1], "tool_calls", None))

    def __call__(self, state: ChatState):
        """Node function for the parent graph: run the agent on the chat history and
        return ONLY its final answer (tool calls stay private to this agent)."""
        result = self.graph.invoke({"messages": state["messages"]}, {"recursion_limit": 12})
        final_text = message_text(result["messages"][-1])
        return {"messages": [AIMessage(content=final_text, name=self.name.replace(" ", "_"))]}


# =============================================================================
# 4. ROUTER + PARENT GRAPH WITH MEMORY
# =============================================================================
class CustomerServiceChatbot:
    def __init__(self, model, debug: bool = False):
        self.debug = debug
        self.router_llm = model.with_structured_output(RouteDecision)

        # Four distinct agents, each with its own prompt and tool set
        self.agents = {
            "ORDER_STATUS": SpecialistAgent("Order Status Agent", model, ORDER_AGENT_PROMPT, ORDER_TOOLS),
            "REFUND_POLICY": SpecialistAgent("Refund Policy Agent", model, REFUND_AGENT_PROMPT, REFUND_TOOLS),
            "PRODUCT_RECOMMENDATION": SpecialistAgent(
                "Product Recommendation Agent", model, PRODUCT_AGENT_PROMPT, PRODUCT_TOOLS
            ),
            "GENERAL": SpecialistAgent("General Agent", model, GENERAL_AGENT_PROMPT, []),
        }

        graph = StateGraph(ChatState)
        graph.add_node("router", self.route_message)
        for route, agent in self.agents.items():
            graph.add_node(route, agent)
            graph.add_edge(route, END)  # one agent answers per turn
        graph.add_edge(START, "router")
        # Conditional edge = the decision-making: go to the agent the router picked
        graph.add_conditional_edges("router", lambda s: s["route"], {r: r for r in self.agents})

        # MemorySaver stores the state (all messages) after every turn, keyed by thread_id
        self.memory = MemorySaver()
        self.graph = graph.compile(checkpointer=self.memory)

    def route_message(self, state: ChatState):
        """Router Agent: classify the latest message into one of the 4 routes."""
        recent = state["messages"][-7:]  # latest message + a little context for follow-ups
        try:
            decision = self.router_llm.invoke([SystemMessage(content=ROUTER_PROMPT)] + recent)
            route = decision.route
            if self.debug:
                print(f"  [router] {route}: {decision.reason}")
        except Exception as err:  # if the LLM returns something unparseable, fall back safely
            if self.debug:
                print(f"  [router] fallback to GENERAL ({err})")
            route = "GENERAL"
        return {"route": route}

    def chat(self, user_text: str, thread_id: str):
        """Send one user message in a conversation thread; returns (route, reply)."""
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({"messages": [HumanMessage(content=user_text)]}, config)
        return result["route"], message_text(result["messages"][-1])

    def history(self, thread_id: str):
        """Read back what MemorySaver has stored for a thread."""
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        return snapshot.values.get("messages", [])


def run_conversation(bot: CustomerServiceChatbot, user_inputs, thread_id=None, title=None):
    """Helper used by the notebook: run a scripted multi-turn conversation and print it."""
    thread_id = thread_id or str(uuid.uuid4())
    if title:
        print("=" * 78 + f"\n{title}\n(thread_id: {thread_id[:8]}...)\n" + "=" * 78)
    for turn, text in enumerate(user_inputs, start=1):
        route, reply = bot.chat(text, thread_id)
        print(f"\n[Turn {turn}] USER: {text}")
        print(f"   -> routed to: {AGENT_NAMES[route]}")
        print(f"NOVA: {reply}")
    print(f"\n(MemorySaver now holds {len(bot.history(thread_id))} messages for this thread)\n")
    return thread_id


if __name__ == "__main__":
    from config import get_model

    bot = CustomerServiceChatbot(get_model(), debug=False)
    thread = str(uuid.uuid4())
    print("TechNova customer service chat. Type 'quit' to exit, 'new' to start a new conversation.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "new":
            thread = str(uuid.uuid4())
            print("(new conversation - memory cleared)\n")
            continue
        if not text:
            continue
        route, reply = bot.chat(text, thread)
        print(f"   [{AGENT_NAMES[route]}]\nNova: {reply}\n")
