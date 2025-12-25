"""
LangChain Integration Examples for DenseCore

This script demonstrates how to use DenseCore with LangChain for various tasks:
1. Basic LLM usage
2. Chat model with conversation
3. Building chains
4. Streaming generation
5. Async operations

Requirements:
    pip install densecore[langchain]
"""

import asyncio

# LangChain imports
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# DenseCore imports
from densecore.integrations import DenseCoreChatModel, DenseCoreLLM


def example_1_basic_llm():
    """Example 1: Basic LLM usage"""
    print("\n" + "=" * 80)
    print("Example 1: Basic LLM Usage")
    print("=" * 80)

    # Initialize DenseCore LLM
    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",  # Or use model_path
        temperature=0.7,
        max_tokens=128,
    )

    # Simple generation
    prompt = "Explain quantum computing in one sentence."
    response = llm(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")


def example_2_chat_model():
    """Example 2: Chat model with conversation"""
    print("\n" + "=" * 80)
    print("Example 2: Chat Model with Conversation")
    print("=" * 80)

    # Initialize chat model
    chat = DenseCoreChatModel(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=200,
    )

    # Create conversation
    messages = [
        SystemMessage(content="You are a helpful AI assistant specialized in Python programming."),
        HumanMessage(content="What is a Python decorator?"),
    ]

    response = chat(messages)
    print(f"\nUser: {messages[1].content}")
    print(f"Assistant: {response.content}")

    # Continue conversation
    messages.append(response)
    messages.append(HumanMessage(content="Can you show me a simple example?"))

    response2 = chat(messages)
    print(f"\nUser: {messages[-1].content}")
    print(f"Assistant: {response2.content}")


def example_3_simple_chain():
    """Example 3: Building a simple chain"""
    print("\n" + "=" * 80)
    print("Example 3: Simple Chain")
    print("=" * 80)

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.8,
        max_tokens=150,
    )

    # Create prompt template
    prompt = PromptTemplate(input_variables=["topic"], template="Write a haiku about {topic}:")

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run chain
    topic = "artificial intelligence"
    result = chain.run(topic=topic)
    print(f"\nTopic: {topic}")
    print(f"Haiku:\n{result}")


def example_4_sequential_chain():
    """Example 4: Sequential chain"""
    print("\n" + "=" * 80)
    print("Example 4: Sequential Chain")
    print("=" * 80)

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # First chain: Generate a topic
    first_prompt = PromptTemplate(
        input_variables=["subject"],
        template="Suggest one specific topic about {subject}. Just the topic name:",
    )
    chain_one = LLMChain(llm=llm, prompt=first_prompt)

    # Second chain: Write about the topic
    second_prompt = PromptTemplate(
        input_variables=["topic"], template="Write one interesting fact about {topic}:"
    )
    chain_two = LLMChain(llm=llm, prompt=second_prompt)

    # Combine chains
    overall_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

    # Run
    result = overall_chain.run("machine learning")
    print(f"\nFinal result: {result}")


def example_5_streaming():
    """Example 5: Streaming generation"""
    print("\n" + "=" * 80)
    print("Example 5: Streaming Generation")
    print("=" * 80)

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=200,
    )

    prompt = "Write a short story about a robot:"

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


async def example_6_async():
    """Example 6: Async operations"""
    print("\n" + "=" * 80)
    print("Example 6: Async Operations")
    print("=" * 80)

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # Multiple async requests
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\nGenerating responses asynchronously for 3 prompts...")

    # Create tasks
    tasks = [llm.agenerate([prompt]) for prompt in prompts]

    # Wait for all
    results = await asyncio.gather(*tasks)

    # Print results
    for prompt, result in zip(prompts, results):
        print(f"\nQ: {prompt}")
        print(f"A: {result.generations[0][0].text[:100]}...")


async def example_7_async_streaming():
    """Example 7: Async streaming"""
    print("\n" + "=" * 80)
    print("Example 7: Async Streaming")
    print("=" * 80)

    chat = DenseCoreChatModel(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=150,
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain async/await in Python briefly."),
    ]

    print("\nStreaming response:")
    print("Assistant: ", end="", flush=True)

    async for chunk in chat.astream(messages):
        print(chunk.content, end="", flush=True)

    print("\n")


def example_8_memory():
    """Example 8: Conversation with memory"""
    print("\n" + "=" * 80)
    print("Example 8: Conversation with Memory")
    print("=" * 80)

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # Create memory
    memory = ConversationBufferMemory()

    # Create conversation chain with memory
    template = """You are a helpful assistant.

{history}
Human: {input}
Assistant:"""

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    # Have a conversation
    print("\nConversation 1:")
    response1 = chain.predict(input="My name is Alice.")
    print(f"Response: {response1}")

    print("\nConversation 2:")
    response2 = chain.predict(input="What is my name?")
    print(f"Response: {response2}")


def main():
    """Run all examples"""
    print("\n🚀 DenseCore + LangChain Examples")
    print("=" * 80)

    # Run synchronous examples
    example_1_basic_llm()
    example_2_chat_model()
    example_3_simple_chain()
    example_4_sequential_chain()
    example_5_streaming()
    example_8_memory()

    # Run async examples
    print("\n" + "=" * 80)
    print("Running Async Examples...")
    print("=" * 80)
    asyncio.run(example_6_async())
    asyncio.run(example_7_async_streaming())

    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
