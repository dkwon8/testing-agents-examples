"""
Level 3: Multi-Tool Agent with Complex Decision Making

WHAT THIS IS:
An agent that has access to multiple specialized tools and must decide:
- Which tool(s) to use
- In what order to use them
- How to combine results from multiple tools

This represents the highest complexity level for agents.

HOW IT WORKS:
1. User asks a complex question
2. Agent analyzes what's needed
3. Agent chooses appropriate tool(s)
4. Agent executes tools (possibly in sequence)
5. Agent synthesizes final answer from tool outputs

WHY THIS MATTERS:
- Handles complex, multi-step tasks
- Shows agent decision-making and planning
- More powerful than simple Q&A or RAG
- Common in production for customer service, data analysis, automation

COMPLEXITY COMPARED TO OTHER AGENTS:
- Simple: Question -> LLM -> Answer
- RAG: Question -> Retrieve -> LLM with context -> Answer
- Multi-Tool: Question -> Analyze -> Choose tools -> Execute (possibly multiple times) -> Synthesize -> Answer

To run:
    python examples/06_multitool_agent.py
"""

import os
import json
from openai import OpenAI
import mlflow
from dotenv import load_dotenv
from typing import Dict, List, Any
from datetime import datetime

load_dotenv()


# ============================================================================
# STEP 1: DEFINE MULTIPLE TOOLS
# ============================================================================

@mlflow.trace
def calculator(operation: str, a: float, b: float) -> Dict[str, Any]:
    """
    Tool 1: Mathematical calculator

    Performs basic arithmetic operations.

    Args:
        operation: One of [add, subtract, multiply, divide]
        a: First number
        b: Second number

    Returns:
        Dict with result and metadata
    """
    print(f"  [Calculator] {operation}({a}, {b})")

    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else None
    }

    result = operations.get(operation)

    if result is None:
        return {"error": "Invalid operation or division by zero"}

    return {
        "result": result,
        "operation": operation,
        "inputs": {"a": a, "b": b}
    }


@mlflow.trace
def search_knowledge_base(query: str) -> Dict[str, Any]:
    """
    Tool 2: Knowledge base search

    Searches a knowledge base for information.
    (Simplified version of the RAG agent's retrieval)

    Args:
        query: Search query

    Returns:
        Dict with search results
    """
    print(f"  [Search] Looking for: {query[:50]}...")

    # Simplified knowledge base
    knowledge = {
        "mlflow": "MLflow is an open-source platform for managing machine learning lifecycles.",
        "tracing": "MLflow Tracing captures execution of AI systems for debugging and observability.",
        "evaluation": "MLflow evaluation validates model quality using metrics and scorers.",
        "pricing": "MLflow is free and open-source. Enterprise support available from Databricks.",
    }

    # Simple keyword match
    query_lower = query.lower()
    results = []

    for topic, info in knowledge.items():
        if topic in query_lower or any(word in info.lower() for word in query_lower.split()):
            results.append({
                "topic": topic,
                "information": info
            })

    return {
        "query": query,
        "results": results,
        "num_results": len(results)
    }


@mlflow.trace
def get_current_date() -> Dict[str, Any]:
    """
    Tool 3: Date/time information

    Returns current date and time information.

    Returns:
        Dict with date/time data
    """
    print(f"  [DateTime] Getting current date/time...")

    now = datetime.now()

    return {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timestamp": now.isoformat()
    }


@mlflow.trace
def convert_units(value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
    """
    Tool 4: Unit converter

    Converts between different units of measurement.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Dict with converted value
    """
    print(f"  [Converter] {value} {from_unit} -> {to_unit}")

    # Simplified conversion factors (meters to other units)
    conversions = {
        ("meters", "feet"): 3.28084,
        ("feet", "meters"): 0.3048,
        ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("kg", "pounds"): 2.20462,
        ("pounds", "kg"): 0.453592,
    }

    key = (from_unit.lower(), to_unit.lower())
    conversion = conversions.get(key)

    if conversion is None:
        return {"error": f"Conversion from {from_unit} to {to_unit} not supported"}

    if callable(conversion):
        result = conversion(value)
    else:
        result = value * conversion

    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": round(result, 2),
        "converted_unit": to_unit
    }


# ============================================================================
# STEP 2: TOOL REGISTRY
# ============================================================================

# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs basic mathematical operations: add, subtract, multiply, divide",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
                    },
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Searches knowledge base for information about MLflow, tracing, evaluation, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or topic to look up"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Returns current date, time, and day of week",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Converts between units (meters/feet, celsius/fahrenheit, kg/pounds)",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"}
                },
                "required": ["value", "from_unit", "to_unit"]
            }
        }
    }
]

# Map function names to actual Python functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
    "get_current_date": get_current_date,
    "convert_units": convert_units
}


# ============================================================================
# STEP 3: TOOL EXECUTION
# ============================================================================

@mlflow.trace
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Execute a tool by name with given arguments.

    This function is traced so you can see:
    - Which tool was called
    - What arguments were passed
    - What result was returned
    - How long execution took

    Args:
        tool_name: Name of tool to execute
        arguments: Arguments to pass to tool

    Returns:
        Tool execution result
    """
    tool_func = TOOL_FUNCTIONS.get(tool_name)

    if tool_func is None:
        return {"error": f"Tool '{tool_name}' not found"}

    try:
        result = tool_func(**arguments)
        return result
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# STEP 4: MULTI-TOOL AGENT ORCHESTRATION
# ============================================================================

@mlflow.trace
def multitool_agent(question: str, model: str = "gpt-5.4", max_iterations: int = 5) -> Dict:
    """
    Multi-Tool Agent with complex decision making.

    This agent can:
    1. Analyze the question
    2. Decide which tool(s) to use
    3. Execute tools in sequence (if needed)
    4. Synthesize final answer from tool results

    The agent may call multiple tools in a chain:
    Example: "What's 25 * 4 in feet?" ->
        1. Calculator (25 * 4 = 100)
        2. Convert units (100 meters to feet)
        3. Synthesize final answer

    Args:
        question: User's question
        model: OpenAI model to use
        max_iterations: Max tool calls to prevent infinite loops

    Returns:
        Dict with answer, tools used, and execution trace
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"\nMulti-Tool Agent - Question: {question}")

    messages = [{"role": "user", "content": question}]
    tools_used = []
    iteration = 0

    # Agent loop: may use multiple tools
    while iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}:")

        # Ask agent what to do
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_completion_tokens=500
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # If no tools needed, agent has final answer
        if not tool_calls:
            print("  Agent has final answer (no more tools needed)")
            final_answer = response_message.content
            break

        # Execute each tool call
        messages.append(response_message)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  Tool selected: {tool_name}")
            print(f"  Arguments: {tool_args}")

            # Execute tool
            tool_result = execute_tool(tool_name, tool_args)

            # Track what tools were used
            tools_used.append({
                "tool": tool_name,
                "arguments": tool_args,
                "result": tool_result
            })

            # Add tool result to conversation
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result)
            })

    else:
        # Max iterations reached
        print(f"  Max iterations ({max_iterations}) reached")
        final_answer = "I apologize, but I couldn't complete this task within the allowed steps."

    print(f"\nFinal Answer: {final_answer[:100]}...")

    return {
        "question": question,
        "answer": final_answer,
        "tools_used": tools_used,
        "num_tools_called": len(tools_used),
        "iterations": iteration,
        "model": model
    }


# ============================================================================
# STEP 5: TESTING WITH DIVERSE QUESTIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LEVEL 3: MULTI-TOOL AGENT DEMONSTRATION")
    print("=" * 80)

    # Test questions requiring different tool combinations
    questions = [
        # Single tool (calculator)
        "What is 156 multiplied by 23?",

        # Single tool (knowledge search)
        "What is MLflow Tracing?",

        # Single tool (date)
        "What day of the week is it today?",

        # Single tool (converter)
        "Convert 100 meters to feet",

        # Multiple tools (calculator + knowledge)
        "If I run 3 experiments and each tracks 150 metrics, how many total metrics is that? Also tell me what MLflow tracking is.",

        # Multiple tools (date + calculator)
        "What's today's date? Also calculate 45 + 67.",

        # Complex multi-tool
        "Convert 25 celsius to fahrenheit, then multiply by 2"
    ]

    print("\nRunning Multi-Tool Agent on diverse questions...\n")

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {question}")
        print(f"{'='*80}")

        result = multitool_agent(question)

        print(f"\n  TOOLS USED ({result['num_tools_called']}):")
        for j, tool_use in enumerate(result['tools_used'], 1):
            print(f"    {j}. {tool_use['tool']}")
            print(f"       Args: {tool_use['arguments']}")
            print(f"       Result: {tool_use['result']}")

        print(f"\n  FINAL ANSWER:")
        print(f"  {result['answer']}")

    print("\n" + "=" * 80)
    print("Multi-Tool Agent Demonstration Complete!")
    print("=" * 80)
    print("\nView traces in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")
    print("\nLook for:")
    print("   - multitool_agent span (parent)")
    print("   - execute_tool spans (children - one per tool call)")
    print("   - Individual tool function spans (calculator, search_knowledge_base, etc.)")
    print("   - Multiple iterations if agent used tools sequentially")
    print("\nKey observations:")
    print("   - Agent decides which tool(s) to use based on question")
    print("   - Some questions require multiple tools in sequence")
    print("   - Total latency = sum of all tool executions + LLM reasoning")
    print("   - More complex = more powerful but slower and more expensive")
