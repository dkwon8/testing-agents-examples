"""
Phase 2: Simple Agent (No Tools)

Goal: Build a basic agent with OpenAI API and trace it.
This agent just answers questions without using any tools.

Setup:
    1. Copy .env.example to .env
    2. Add your OPENAI_API_KEY to .env
    3. Run: python examples/02_simple_agent.py

To run:
    python examples/02_simple_agent.py
"""

import os
from openai import OpenAI
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@mlflow.trace
def call_openai(prompt: str, model: str = "gpt-5.4") -> str:
    """Call OpenAI API and trace the interaction"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # GPT-5.4 uses max_completion_tokens instead of max_tokens
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1024
    )

    return response.choices[0].message.content


@mlflow.trace
def simple_agent(question: str, model: str = "gpt-5.4") -> dict:
    """A simple agent that answers questions"""
    print(f"\nQuestion: {question}")

    # Call OpenAI (pass model parameter for flexibility)
    answer = call_openai(question, model=model)

    print(f"Answer: {answer}")

    return {
        "question": question,
        "answer": answer,
        "model": model
    }


if __name__ == "__main__":
    # Test the agent
    questions = [
        "What is MLflow?",
        "Explain what agent tracing means in 2 sentences.",
    ]

    for q in questions:
        result = simple_agent(q)
        print(f"\n{'='*60}\n")

    print("\nAgent traces created! View them in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")
    print("\nLook for:")
    print("   - simple_agent span containing call_openai span")
    print("   - Input question and output answer")
    print("   - Timing information")
