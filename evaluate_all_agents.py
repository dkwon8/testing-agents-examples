"""
Evaluate All Agent Levels with MLflow Scorers

This script runs evaluation on all 3 agent complexity levels:
- Level 1: Simple Agent
- Level 2: RAG Agent
- Level 3: Multi-Tool Agent

Each is evaluated with:
- Readability metrics (Flesch-Kincaid, ARI)
- Performance metrics (latency, token count)
- Quality metrics (exact match, variance)

To run:
    python evaluate_all_agents.py
"""

import mlflow
import pandas as pd
from dotenv import load_dotenv
import importlib.util
import sys

load_dotenv()


# Import modules with numeric prefixes
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import all three agents
simple_agent_module = import_from_file("simple_agent", "examples/02_simple_agent.py")
rag_agent_module = import_from_file("rag_agent", "examples/05_rag_agent.py")
multitool_agent_module = import_from_file("multitool_agent", "examples/06_multitool_agent.py")

simple_agent = simple_agent_module.simple_agent
rag_agent = rag_agent_module.rag_agent
KnowledgeBase = rag_agent_module.KnowledgeBase
multitool_agent = multitool_agent_module.multitool_agent


def create_evaluation_dataset():
    """
    Create test dataset for evaluating all agents.

    Questions cover:
    - Simple facts (all agents should handle)
    - Knowledge retrieval (RAG excels)
    - Calculations (Multi-Tool excels)
    """
    return pd.DataFrame({
        "question": [
            "What is MLflow?",
            "How does MLflow Tracing work?",
            "What is 25 multiplied by 4?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing machine learning lifecycles.",
            "MLflow Tracing captures execution of code, recording inputs, outputs, and metadata.",
            "100",
        ]
    })


def evaluate_simple_agent(test_data: pd.DataFrame, model: str = "gpt-5.4"):
    """Evaluate Level 1: Simple Agent"""
    print("\n" + "="*100)
    print("EVALUATING LEVEL 1: SIMPLE AGENT")
    print("="*100)

    def model_fn(inputs):
        results = []
        for _, row in inputs.iterrows():
            response = simple_agent(question=row["question"], model=model)
            results.append(response["answer"])
        return results

    with mlflow.start_run(run_name="Level_1_Simple_Agent_Evaluation"):
        mlflow.set_tag("agent_level", "1_simple")
        mlflow.set_tag("agent_type", "simple_qa")

        results = mlflow.evaluate(
            model=model_fn,
            data=test_data,
            targets="ground_truth",
            model_type="question-answering",
            evaluators="default",
        )

        print(f"\nSimple Agent Evaluation Complete!")
        print(f"Average Metrics:")
        for metric, value in results.metrics.items():
            print(f"  {metric}: {value}")

    return results


def evaluate_rag_agent(test_data: pd.DataFrame, model: str = "gpt-5.4"):
    """Evaluate Level 2: RAG Agent"""
    print("\n" + "="*100)
    print("EVALUATING LEVEL 2: RAG AGENT")
    print("="*100)

    kb = KnowledgeBase()

    def model_fn(inputs):
        results = []
        for _, row in inputs.iterrows():
            response = rag_agent(
                question=row["question"],
                kb=kb,
                top_k=3,
                model=model
            )
            results.append(response["answer"])
        return results

    with mlflow.start_run(run_name="Level_2_RAG_Agent_Evaluation"):
        mlflow.set_tag("agent_level", "2_rag")
        mlflow.set_tag("agent_type", "retrieval_augmented")

        results = mlflow.evaluate(
            model=model_fn,
            data=test_data,
            targets="ground_truth",
            model_type="question-answering",
            evaluators="default",
        )

        print(f"\nRAG Agent Evaluation Complete!")
        print(f"Average Metrics:")
        for metric, value in results.metrics.items():
            print(f"  {metric}: {value}")

    return results


def evaluate_multitool_agent(test_data: pd.DataFrame, model: str = "gpt-5.4"):
    """Evaluate Level 3: Multi-Tool Agent"""
    print("\n" + "="*100)
    print("EVALUATING LEVEL 3: MULTI-TOOL AGENT")
    print("="*100)

    def model_fn(inputs):
        results = []
        for _, row in inputs.iterrows():
            response = multitool_agent(
                question=row["question"],
                model=model,
                max_iterations=5
            )
            results.append(response["answer"])
        return results

    with mlflow.start_run(run_name="Level_3_MultiTool_Agent_Evaluation"):
        mlflow.set_tag("agent_level", "3_multitool")
        mlflow.set_tag("agent_type", "multi_tool")

        results = mlflow.evaluate(
            model=model_fn,
            data=test_data,
            targets="ground_truth",
            model_type="question-answering",
            evaluators="default",
        )

        print(f"\nMulti-Tool Agent Evaluation Complete!")
        print(f"Average Metrics:")
        for metric, value in results.metrics.items():
            print(f"  {metric}: {value}")

    return results


def run_all_evaluations():
    """Run evaluation on all 3 agent levels"""
    print("="*100)
    print("EVALUATING ALL AGENT COMPLEXITY LEVELS")
    print("="*100)
    print("\nThis will evaluate:")
    print("  Level 1: Simple Agent")
    print("  Level 2: RAG Agent")
    print("  Level 3: Multi-Tool Agent")
    print("\nWith metrics:")
    print("  - Readability (Flesch-Kincaid, ARI)")
    print("  - Performance (latency, token count)")
    print("  - Quality (exact match, variance)")

    # Create test dataset
    test_data = create_evaluation_dataset()
    print(f"\nTest dataset: {len(test_data)} questions")

    # Run evaluations
    simple_results = evaluate_simple_agent(test_data)
    rag_results = evaluate_rag_agent(test_data)
    multitool_results = evaluate_multitool_agent(test_data)

    # Summary comparison
    print("\n" + "="*100)
    print("EVALUATION SUMMARY - ALL AGENTS")
    print("="*100)

    print("\nKey Metrics Comparison:")
    print("-"*100)

    # Compare latency
    if 'latency/mean' in simple_results.metrics:
        print(f"\nLatency (average):")
        print(f"  Simple Agent:     {simple_results.metrics.get('latency/mean', 'N/A'):.2f}s")
        print(f"  RAG Agent:        {rag_results.metrics.get('latency/mean', 'N/A'):.2f}s")
        print(f"  Multi-Tool Agent: {multitool_results.metrics.get('latency/mean', 'N/A'):.2f}s")

    # Compare token usage
    if 'token_count/mean' in simple_results.metrics:
        print(f"\nToken Count (average):")
        print(f"  Simple Agent:     {simple_results.metrics.get('token_count/mean', 'N/A'):.0f}")
        print(f"  RAG Agent:        {rag_results.metrics.get('token_count/mean', 'N/A'):.0f}")
        print(f"  Multi-Tool Agent: {multitool_results.metrics.get('token_count/mean', 'N/A'):.0f}")

    # Compare readability
    if 'flesch_kincaid_grade_level/mean' in simple_results.metrics:
        print(f"\nFlesch-Kincaid Grade Level (average):")
        print(f"  Simple Agent:     {simple_results.metrics.get('flesch_kincaid_grade_level/mean', 'N/A'):.2f}")
        print(f"  RAG Agent:        {rag_results.metrics.get('flesch_kincaid_grade_level/mean', 'N/A'):.2f}")
        print(f"  Multi-Tool Agent: {multitool_results.metrics.get('flesch_kincaid_grade_level/mean', 'N/A'):.2f}")

    print("\n" + "="*100)
    print("All evaluations complete! View in MLflow UI:")
    print("  mlflow ui")
    print("  http://localhost:5000")
    print("\nWhat to look for:")
    print("  1. Click 'Experiments' tab")
    print("  2. Look for runs: 'Level_1_Simple_Agent_Evaluation', etc.")
    print("  3. Compare metrics across all 3 agent levels")
    print("  4. Click on each run to see detailed per-question results")
    print("  5. Check 'Traces' tab to see execution paths")
    print("="*100)

    return {
        "simple": simple_results,
        "rag": rag_results,
        "multitool": multitool_results
    }


if __name__ == "__main__":
    results = run_all_evaluations()
