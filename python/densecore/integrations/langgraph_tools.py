"""
LangGraph integration for DenseCore.

This module provides utilities for building LangGraph workflows with DenseCore,
including:
- Node creation for LLM calls
- Tool execution with retry logic
- ReAct-style agent creation (create_react_agent)
- ToolNode for processing tool calls
- State management and checkpointing
"""

import logging
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Optional,
    TypedDict,
    TypeVar,
)

from langchain_core.messages import SystemMessage

try:
    from langgraph.graph import END, StateGraph
    from langgraph.graph.graph import CompiledGraph
    from langgraph.prebuilt import ToolExecutor, ToolInvocation
except ImportError as e:
    raise ImportError(
        "LangGraph integration requires langgraph. Install with: pip install densecore[langchain]"
    ) from e

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from .langchain import DenseCoreChatModel

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=dict[str, Any])


# Type for agent state
class AgentState(TypedDict, total=False):
    """Standard state for ReAct agents."""

    messages: Sequence[BaseMessage]
    next: str


def create_densecore_node(
    model_path: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    node_name: str = "llm",
    system_prompt: Optional[str] = None,
    **generation_kwargs: Any,
) -> Callable[[StateT], StateT]:
    """
    Create a LangGraph node that uses DenseCore for text generation.

    This factory function creates a stateful node that can be added to a LangGraph
    workflow. The node reads from the state's "messages" key and appends the
    generated response.

    Args:
        model_path: Path to GGUF model file
        hf_repo_id: HuggingFace repo ID (alternative to model_path)
        node_name: Name for the node (for debugging/logging)
        system_prompt: Optional system prompt to prepend
        **generation_kwargs: Additional generation parameters (temperature, max_tokens, etc.)

    Returns:
        A callable that can be used as a LangGraph node

    Example:
        >>> from langgraph.graph import StateGraph
        >>> from densecore.integrations import create_densecore_node
        >>>
        >>> # Define state schema
        >>> class AgentState(TypedDict):
        ...     messages: List[BaseMessage]
        ...
        >>> workflow = StateGraph(AgentState)
        >>>
        >>> # Add DenseCore node
        >>> llm_node = create_densecore_node(
        ...     hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        ...     temperature=0.7,
        ...     max_tokens=256
        ... )
        >>> workflow.add_node("llm", llm_node)
        >>>
        >>> # Build and compile graph
        >>> workflow.set_entry_point("llm")
        >>> workflow.add_edge("llm", END)
        >>> app = workflow.compile()
    """
    # Initialize the chat model
    chat_model = DenseCoreChatModel(
        model_path=model_path,
        hf_repo_id=hf_repo_id,
        **generation_kwargs,
    )

    def node_function(state: StateT) -> StateT:
        """Node function that processes messages."""
        messages = state.get("messages", [])

        # Optionally prepend system prompt
        if system_prompt and (not messages or not isinstance(messages[0], type(messages[0]))):
            from langchain_core.messages import SystemMessage

            messages = [SystemMessage(content=system_prompt)] + messages

        try:
            # Generate response
            logger.debug(f"Node '{node_name}' generating response for {len(messages)} messages")
            response = chat_model.invoke(messages)

            # Update state with new message
            updated_messages = messages + [response]
            return {**state, "messages": updated_messages}

        except Exception as e:
            logger.error(f"Error in node '{node_name}': {e}")
            # Add error message to state
            error_msg = AIMessage(content=f"Error: {str(e)}")
            return {**state, "messages": messages + [error_msg]}

    return node_function


class DenseCoreToolExecutor:
    """
    Tool executor for LangGraph that uses DenseCore for LLM calls.

    This class manages tool execution in LangGraph workflows, handling
    tool invocations and integrating them with DenseCore's LLM capabilities.

    Example:
        >>> from langchain_core.tools import tool
        >>> from densecore.integrations import DenseCoreToolExecutor
        >>>
        >>> @tool
        ... def search(query: str) -> str:
        ...     '''Search for information.'''
        ...     return f"Results for: {query}"
        >>>
        >>> tools = [search]
        >>> tool_executor = DenseCoreToolExecutor(tools)
        >>>
        >>> # Use in LangGraph
        >>> result = tool_executor.invoke(
        ...     ToolInvocation(tool="search", tool_input={"query": "AI"})
        ... )
    """

    def __init__(
        self,
        tools: list[BaseTool],
        handle_tool_errors: bool = True,
        max_retries: int = 3,
    ):
        """
        Initialize tool executor.

        Args:
            tools: List of LangChain tools to make available
            handle_tool_errors: Whether to catch and handle tool execution errors
            max_retries: Maximum number of retries for failed tool calls
        """
        self.tools = {tool.name: tool for tool in tools}
        self.handle_tool_errors = handle_tool_errors
        self.max_retries = max_retries
        self._base_executor = ToolExecutor(tools)

    def invoke(self, tool_invocation: ToolInvocation) -> Any:
        """
        Execute a tool invocation with schema validation.

        Args:
            tool_invocation: ToolInvocation with tool name and input

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool is not found or input validation fails
            Exception: If tool execution fails and handle_tool_errors=False
        """
        tool_name = tool_invocation.tool
        tool_input = tool_invocation.tool_input

        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            if self.handle_tool_errors:
                logger.error(error_msg)
                return {"error": error_msg}
            raise ValueError(error_msg)

        tool = self.tools[tool_name]

        # Validate input against Pydantic schema if present
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                # args_schema is a Pydantic model class
                schema_class = tool.args_schema
                # Validate by constructing the model
                validated = schema_class(**tool_input)
                # Use the validated data (converts to dict if needed)
                if hasattr(validated, "model_dump"):
                    # Pydantic v2
                    tool_input = validated.model_dump()
                elif hasattr(validated, "dict"):
                    # Pydantic v1
                    tool_input = validated.dict()
                logger.debug(f"Tool '{tool_name}' input validated against schema")
            except Exception as e:
                error_msg = f"Tool '{tool_name}' input validation failed: {e}"
                if self.handle_tool_errors:
                    logger.warning(error_msg)
                    return {"error": error_msg}
                raise ValueError(error_msg) from e

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Executing tool '{tool_name}' (attempt {attempt + 1}/{self.max_retries})"
                )
                result = tool.invoke(tool_input)
                return result

            except Exception as e:
                logger.warning(f"Tool '{tool_name}' execution failed (attempt {attempt + 1}): {e}")

                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    if self.handle_tool_errors:
                        return {
                            "error": f"Tool execution failed after {self.max_retries} attempts: {str(e)}"
                        }
                    raise

        # Should not reach here
        return {"error": "Unknown error during tool execution"}

    async def ainvoke(self, tool_invocation: ToolInvocation) -> Any:
        """
        Async execute a tool invocation.

        Args:
            tool_invocation: ToolInvocation with tool name and input

        Returns:
            Tool execution result
        """
        tool_name = tool_invocation.tool

        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"
            if self.handle_tool_errors:
                logger.error(error_msg)
                return {"error": error_msg}
            raise ValueError(error_msg)

        tool = self.tools[tool_name]

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    f"Async executing tool '{tool_name}' (attempt {attempt + 1}/{self.max_retries})"
                )
                result = await tool.ainvoke(tool_invocation.tool_input)
                return result

            except Exception as e:
                logger.warning(
                    f"Async tool '{tool_name}' execution failed (attempt {attempt + 1}): {e}"
                )

                if attempt == self.max_retries - 1:
                    if self.handle_tool_errors:
                        return {
                            "error": f"Tool execution failed after {self.max_retries} attempts: {str(e)}"
                        }
                    raise

        return {"error": "Unknown error during async tool execution"}


def create_agent_node(
    llm: DenseCoreChatModel,
    tools: list[BaseTool],
    node_name: str = "agent",
) -> Callable[[StateT], StateT]:
    """
    Create an agent node that can use tools.

    This creates a more sophisticated node that can decide when to use tools
    and when to respond directly.

    Args:
        llm: DenseCoreChatModel instance
        tools: List of available tools
        node_name: Name for the node

    Returns:
        A callable that can be used as a LangGraph node

    Example:
        >>> from langchain_core.tools import tool
        >>> from densecore.integrations import DenseCoreChatModel, create_agent_node
        >>>
        >>> @tool
        ... def calculator(expression: str) -> str:
        ...     '''Evaluate a math expression.'''
        ...     return str(eval(expression))
        >>>
        >>> llm = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
        >>> tools = [calculator]
        >>>
        >>> agent_node = create_agent_node(llm, tools)
    """
    tool_executor = DenseCoreToolExecutor(tools)

    def node_function(state: StateT) -> StateT:
        """Agent node function."""
        messages = state.get("messages", [])

        try:
            # Generate response (model decides whether to use tools)
            logger.debug(f"Agent node '{node_name}' processing {len(messages)} messages")

            # For simplicity, we'll just use the LLM directly
            # In a production system, you'd implement tool-use detection here
            response = llm.invoke(messages)

            # Update state
            updated_messages = messages + [response]
            return {**state, "messages": updated_messages}

        except Exception as e:
            logger.error(f"Error in agent node '{node_name}': {e}")
            error_msg = AIMessage(content=f"Error: {str(e)}")
            return {**state, "messages": messages + [error_msg]}

    return node_function


class GraphCheckpoint:
    """
    Simple checkpointing utility for LangGraph workflows.

    Provides basic save/load functionality for graph states to enable
    resumable workflows.

    Example:
        >>> from densecore.integrations import GraphCheckpoint
        >>>
        >>> checkpoint = GraphCheckpoint()
        >>>
        >>> # Save state
        >>> state = {"messages": [...], "step": 5}
        >>> checkpoint.save("my_workflow", state)
        >>>
        >>> # Load state
        >>> restored_state = checkpoint.load("my_workflow")
    """

    def __init__(self, checkpoint_dir: str = "./.checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        import os

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, checkpoint_id: str, state: dict[str, Any]) -> None:
        """
        Save a checkpoint.

        Args:
            checkpoint_id: Unique identifier for the checkpoint
            state: State dictionary to save
        """
        import json
        import os

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        # Convert state to JSON-serializable format
        serializable_state = self._make_serializable(state)

        with open(checkpoint_path, "w") as f:
            json.dump(serializable_state, f, indent=2)

        logger.info(f"Saved checkpoint '{checkpoint_id}' to {checkpoint_path}")

    def load(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Unique identifier for the checkpoint

        Returns:
            Loaded state dictionary, or None if not found
        """
        import json
        import os

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            logger.warning("Checkpoint '%s' not found", checkpoint_id)
            return None

        with open(checkpoint_path, encoding="utf-8") as f:
            state = json.load(f)

        logger.info("Loaded checkpoint '%s' from %s", checkpoint_id, checkpoint_path)
        return state

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, BaseMessage):
            # Convert LangChain messages to dicts
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
            }
        else:
            # Return as-is for primitive types
            return obj


def create_tool_node(
    tools: list[BaseTool],
    handle_errors: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Create a tool execution node for LangGraph.

    This node processes tool calls from the previous agent response and
    returns ToolMessages with the results.

    Args:
        tools: List of available tools
        handle_errors: Whether to catch and return tool errors as messages

    Returns:
        A callable node function for LangGraph

    Example:
        >>> from langchain_core.tools import tool
        >>> from densecore.integrations import create_tool_node
        >>>
        >>> @tool
        ... def calculator(expression: str) -> str:
        ...     '''Evaluate a math expression.'''
        ...     return str(eval(expression))
        >>>
        >>> tool_node = create_tool_node([calculator])
        >>> workflow.add_node("tools", tool_node)
    """
    tool_map = {tool.name: tool for tool in tools}

    def tool_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute tools based on the last AI message's tool calls."""
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]

        # Check if last message has tool calls
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return state

        # Execute each tool call
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{len(tool_messages)}")

            if tool_name not in tool_map:
                error_msg = f"Tool '{tool_name}' not found"
                if handle_errors:
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_id, name=tool_name)
                    )
                continue

            tool = tool_map[tool_name]
            try:
                result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
                )
            except Exception as e:
                if handle_errors:
                    tool_messages.append(
                        ToolMessage(content=f"Error: {e}", tool_call_id=tool_id, name=tool_name)
                    )
                else:
                    raise

        # Update state with tool messages
        return {"messages": messages + tool_messages}

    return tool_node


def should_continue(state: dict[str, Any]) -> str:
    """
    Routing function for ReAct agent.

    Checks if the last message has tool calls. If so, route to tools node.
    Otherwise, end the conversation.

    Args:
        state: Current graph state with messages

    Returns:
        "tools" if there are tool calls, "end" otherwise

    Example:
        >>> workflow.add_conditional_edges(
        ...     "agent",
        ...     should_continue,
        ...     {"tools": "tools", "end": END}
        ... )
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        return "tools"
    return "end"


def create_react_agent(
    model: DenseCoreChatModel,
    tools: list[BaseTool],
    *,
    system_prompt: Optional[str] = None,
    max_iterations: int = 10,
) -> CompiledGraph:
    """
    Create a ReAct-style agent using DenseCore and LangGraph.

    This function creates a complete agent graph with:
    - An agent node that uses the LLM with bound tools
    - A tool execution node
    - Conditional routing based on tool calls

    Args:
        model: DenseCoreChatModel instance
        tools: List of tools available to the agent
        system_prompt: Optional system prompt for the agent
        max_iterations: Maximum number of agent iterations (safety limit)

    Returns:
        Compiled LangGraph that can be invoked

    Example:
        >>> from densecore.integrations import DenseCoreChatModel, create_react_agent
        >>> from langchain_core.tools import tool
        >>> from langchain_core.messages import HumanMessage
        >>>
        >>> @tool
        ... def calculator(expression: str) -> str:
        ...     '''Evaluate a math expression.'''
        ...     return str(eval(expression))
        >>>
        >>> llm = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
        >>> agent = create_react_agent(llm, [calculator])
        >>>
        >>> result = agent.invoke({
        ...     "messages": [HumanMessage(content="What is 25 * 4?")]
        ... })
        >>> print(result["messages"][-1].content)
    """
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    # Create agent node
    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Agent node that calls the LLM with tools."""
        messages = list(state.get("messages", []))

        # Add system prompt if provided and not already present
        if system_prompt:
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt)] + messages

        # Call the model
        response = model_with_tools.invoke(messages)

        return {"messages": messages + [response]}

    # Create tool node
    tool_node = create_tool_node(tools)

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edge from agent
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile and return
    return workflow.compile()
