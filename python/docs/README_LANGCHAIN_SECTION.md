# LangChain & LangGraph Integration - README Section

> Add this section to the main Python README.md

## 🔗 LangChain & LangGraph Integration

DenseCore seamlessly integrates with LangChain and LangGraph, enabling you to build sophisticated LLM applications, chains, agents, and multi-step workflows.

### Quick Start

Install with LangChain support:

```bash
pip install densecore[langchain]
```

### LangChain Example

```python
from densecore.integrations import DenseCoreLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize DenseCore LLM
llm = DenseCoreLLM(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7,
    max_tokens=256
)

# Create a chain
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a haiku about {topic}:"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Generate
result = chain.run(topic="artificial intelligence")
print(result)
```

### Chat Model Example

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.messages import HumanMessage, SystemMessage

chat = DenseCoreChatModel(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain quantum computing in simple terms.")
]

response = chat(messages)
print(response.content)
```

### LangGraph Workflow Example

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from densecore.integrations import create_densecore_node

# Create workflow
workflow = StateGraph(State)

# Add DenseCore node
node = create_densecore_node(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=256
)

workflow.add_node("assistant", node)
workflow.set_entry_point("assistant")
workflow.add_edge("assistant", END)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [HumanMessage(content="Hello!")]
})

print(result["messages"][-1].content)
```

### Production RAG System

```python
from densecore.integrations import DenseCoreLLM
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Initialize components
llm = DenseCoreLLM(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
embeddings = HuggingFaceEmbeddings()

# Create vector store
vectorstore = Chroma.from_documents(documents, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Query
result = qa_chain("What is the main topic of the documents?")
print(result["result"])
```

### Features

- ✅ **LangChain LLM Wrapper** - Full compatibility with LangChain chains and agents
- ✅ **ChatModel Interface** - Conversational AI with message history
- ✅ **Streaming Support** - Real-time token streaming (sync & async)
- ✅ **Async Operations** - Native async/await throughout
- ✅ **LangGraph Nodes** - Easy workflow integration
- ✅ **Tool Execution** - Function calling with retry logic
- ✅ **Checkpointing** - Save and resume long-running workflows
- ✅ **Production-Ready** - Error handling, retries, monitoring

### Documentation

📚 **[Complete LangChain & LangGraph Guide](docs/LANGCHAIN_GUIDE.md)**

Comprehensive guide covering:
- Installation and setup
- All integration features
- Production examples (RAG, agents, workflows)
- Best practices and optimization
- Troubleshooting

### Examples

Full example scripts available in [`examples/`](examples/):

- **[langchain_example.py](examples/langchain_example.py)** - LangChain chains, memory, streaming
- **[langgraph_example.py](examples/langgraph_example.py)** - Workflows, agents, conditional routing
- **[production_rag.py](examples/production_rag.py)** - Complete RAG system

Run examples:
```bash
python examples/langchain_example.py
python examples/langgraph_example.py
python examples/production_rag.py
```

### Compatibility

Works seamlessly with:
- LangChain Expression Language (LCEL)
- LangChain Chains (all types)
- LangChain Agents
- LangChain Memory Systems
- LangChain Callbacks
- LangGraph Workflows
- LangSmith Tracing

### Learn More

- 📖 [LangChain Documentation](https://python.langchain.com/)
- 🔄 [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- 💡 [DenseCore Examples](examples/)
- 🚀 [Integration Guide](docs/LANGCHAIN_GUIDE.md)
