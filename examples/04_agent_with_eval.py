"""
Phase 3: Agent with MLflow Built-in Eval Scorers

Goal: Learn how to evaluate agent responses using MLflow's built-in scorers.
This is the key part - understanding what makes a good agent response!

MLflow Built-in Scorers include:
- mlflow.metrics.genai.answer_relevance() - Is the answer relevant to the question?
- mlflow.metrics.genai.answer_correctness() - Is the answer correct?
- mlflow.metrics.genai.faithfulness() - Is the answer faithful to the context?
- mlflow.metrics.latency() - How long did it take?

To run:
    python examples/04_agent_with_eval.py
"""

import os
from openai import OpenAI
import mlflow
import pandas as pd
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
def agent_with_context(question: str, context: str = None, model: str = "gpt-5.4") -> dict:
    """An agent that answers questions, optionally with context"""
    # Build prompt with context if provided
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nPlease answer based on the context provided."
    else:
        prompt = question

    answer = call_openai(prompt, model=model)

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "model": model
    }


def create_test_dataset():
    """Create a test dataset for evaluation"""
    return pd.DataFrame({
        "question": [
            "What is MLflow?",
            "How does MLflow tracing work?",
            "What are the benefits of using MLflow for ML projects?"
        ],
        "context": [
            "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It includes experiment tracking, model packaging, and deployment.",
            "MLflow tracing captures the execution of your code, recording inputs, outputs, and metadata for each function call. This helps debug and understand agent behavior.",
            "MLflow provides experiment tracking, reproducibility, model versioning, and deployment capabilities for ML projects."
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing machine learning lifecycles, including experiment tracking and model deployment.",
            "MLflow tracing records function calls with their inputs and outputs to help understand agent execution.",
            "MLflow helps with tracking experiments, ensuring reproducibility, versioning models, and deploying them."
        ]
    })


def run_evaluation(model_name: str = "gpt-5.4"):
    """Run agent evaluation with MLflow built-in scorers

    Args:
        model_name: OpenAI model to use. Options:
            - "gpt-4o-mini" (cheap, fast, good for testing)
            - "gpt-5.4" (latest model, recommended)
            - "gpt-4o" (good performance)
    """
    print(f"Starting Agent Evaluation with {model_name}\n")

    # Create test dataset
    test_data = create_test_dataset()

    # Define the model function for MLflow to evaluate
    def model(inputs):
        """Wrapper function that MLflow will call for each test case"""
        results = []
        for _, row in inputs.iterrows():
            response = agent_with_context(
                question=row["question"],
                context=row["context"],
                model=model_name
            )
            results.append(response["answer"])
        return results

    # Define evaluation metrics
    # Note: Some metrics like answer_relevance require an LLM judge
    metrics = [
        mlflow.metrics.latency(),
        # You can add these if you want LLM-based evaluation:
        # mlflow.metrics.genai.answer_relevance(),
        # mlflow.metrics.genai.faithfulness(),
    ]

    print("Evaluating agent responses...\n")

    # Run evaluation
    with mlflow.start_run():
        results = mlflow.evaluate(
            model=model,
            data=test_data,
            targets="ground_truth",  # What we expect
            model_type="question-answering",
            evaluators="default",
        )

    print("\nEvaluation complete!\n")
    print("Results:")
    print(results.metrics)
    print("\nView detailed results in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")
    print("\nWhat to look for:")
    print("   - Latency metrics (how fast)")
    print("   - Token usage")
    print("   - Per-question traces")
    print("   - Comparison of answers vs ground truth")

    return results


if __name__ == "__main__":
    # Run with gpt-5.4 by default
    # To use cheaper model for testing, pass: run_evaluation("gpt-4o-mini")
    results = run_evaluation("gpt-5.4")
