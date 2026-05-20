"""
Phase 2 (Advanced): Agent with Tool Use

Goal: Build an agent that can use tools and trace each step.
This is where it gets interesting - multi-step agent reasoning!

To run:
    python examples/03_agent_with_tools.py
"""

import os
import json
from openai import OpenAI
import mlflow
from dotenv import load_dotenv

load_dotenv()


# Define a simple calculator tool
@mlflow.trace
def calculator(operation: str, a: float, b: float) -> float:
    """A simple calculator tool that performs basic math operations"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else "Error: Division by zero"
    else:
        return "Error: Unknown operation"


# Tool definition for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs basic math operations (add, subtract, multiply, divide)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The math operation to perform"
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
]


@mlflow.trace
def agent_with_tools(question: str, model: str = "gpt-5.4") -> dict:
    """An agent that can use tools to answer questions"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\nQuestion: {question}")

    messages = [{"role": "user", "content": question}]

    # First API call - agent decides if it needs tools
    # GPT-5.4 uses max_completion_tokens instead of max_tokens
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_completion_tokens=1024
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # If agent wants to use tools
    if tool_calls:
        print(f"Agent is using tools...")

        # Add assistant's response to messages
        messages.append(response_message)

        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"   Calling {function_name} with {function_args}")

            # Call the actual function
            function_response = calculator(**function_args)

            # Add tool response to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            })

        # Second API call - get final answer with tool results
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=1024
        )

        answer = final_response.choices[0].message.content
    else:
        # No tools needed
        answer = response_message.content

    print(f"Answer: {answer}")

    return {
        "question": question,
        "answer": answer,
        "model": model,
        "used_tools": bool(tool_calls)
    }


if __name__ == "__main__":
    # Test questions - some need tools, some don't
    questions = [
        "What is 25 multiplied by 4?",
        "If I have 100 dollars and spend 37, how much do I have left?",
        "What is MLflow?",  # This shouldn't need the calculator
    ]

    for q in questions:
        result = agent_with_tools(q)
        print(f"\n{'='*60}\n")

    print("\nAgent traces created! View them in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")
    print("\nLook for:")
    print("   - Traces showing tool calls")
    print("   - Multi-step reasoning (question -> tool -> answer)")
    print("   - Different execution paths (with/without tools)")
