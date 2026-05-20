"""
Agent Complexity Comparison Script

This script runs all 3 agent complexity levels and compares their performance.

PURPOSE:
Demonstrate the trade-offs between simple, medium, and complex agents.

WHAT IT MEASURES:
- Latency (speed)
- Token usage (cost)
- Accuracy/Quality
- Complexity (lines of code, # of function calls)

AGENTS COMPARED:
1. Simple Q&A Agent (Level 1)
2. RAG Agent (Level 2)
3. Multi-Tool Agent (Level 3)

To run:
    python compare_agents.py
"""

import time
import mlflow
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
import importlib.util
import sys

# Import modules with numeric prefixes using importlib
def import_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import all three agent types
simple_agent_module = import_from_file("simple_agent", "examples/02_simple_agent.py")
rag_agent_module = import_from_file("rag_agent", "examples/05_rag_agent.py")
multitool_agent_module = import_from_file("multitool_agent", "examples/06_multitool_agent.py")

simple_agent = simple_agent_module.simple_agent
rag_agent = rag_agent_module.rag_agent
KnowledgeBase = rag_agent_module.KnowledgeBase
multitool_agent = multitool_agent_module.multitool_agent

load_dotenv()


# ============================================================================
# TEST QUESTIONS
# ============================================================================

def get_comparison_questions() -> List[Dict]:
    """
    Questions designed to test different capabilities.

    Each question is categorized by what it requires:
    - Simple facts (all agents should handle)
    - Knowledge retrieval (RAG and multi-tool should excel)
    - Calculations (multi-tool should excel)
    - Complex multi-step (multi-tool only)
    """
    return [
        {
            "question": "What is MLflow?",
            "category": "simple_fact",
            "ground_truth": "MLflow is an open-source platform for managing machine learning lifecycles.",
            "context": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle."
        },
        {
            "question": "How does MLflow Tracing work?",
            "category": "knowledge_retrieval",
            "ground_truth": "MLflow Tracing captures execution of code, recording inputs, outputs, and metadata.",
            "context": "MLflow tracing captures the execution of your code, recording inputs, outputs, and metadata for each function call."
        },
        {
            "question": "What is 25 multiplied by 4?",
            "category": "calculation",
            "ground_truth": "100",
            "context": "This is a mathematical question."
        },
        {
            "question": "Convert 100 meters to feet",
            "category": "unit_conversion",
            "ground_truth": "328.08 feet",
            "context": "This requires unit conversion."
        },
    ]


# ============================================================================
# AGENT WRAPPERS
# ============================================================================

@mlflow.trace
def run_simple_agent(question: str, context: str) -> Dict:
    """Wrapper for simple agent"""
    start_time = time.time()

    result = simple_agent(
        question=question,
        model="gpt-5.4"
    )

    latency = time.time() - start_time

    return {
        "agent_type": "simple",
        "answer": result["answer"],
        "latency": latency,
        "complexity": "low"
    }


@mlflow.trace
def run_rag_agent(question: str, kb: KnowledgeBase) -> Dict:
    """Wrapper for RAG agent"""
    start_time = time.time()

    result = rag_agent(
        question=question,
        kb=kb,
        top_k=3,
        model="gpt-5.4"
    )

    latency = time.time() - start_time

    return {
        "agent_type": "rag",
        "answer": result["answer"],
        "latency": latency,
        "complexity": "medium",
        "sources": result["sources"]
    }


@mlflow.trace
def run_multitool_agent(question: str) -> Dict:
    """Wrapper for multi-tool agent"""
    start_time = time.time()

    result = multitool_agent(
        question=question,
        model="gpt-5.4"
    )

    latency = time.time() - start_time

    return {
        "agent_type": "multitool",
        "answer": result["answer"],
        "latency": latency,
        "complexity": "high",
        "tools_used": result["num_tools_called"]
    }


# ============================================================================
# COMPARISON EXECUTION
# ============================================================================

def run_comparison():
    """
    Run all three agents on test questions and compare results.
    """
    print("=" * 100)
    print("AGENT COMPLEXITY COMPARISON")
    print("=" * 100)

    questions = get_comparison_questions()
    kb = KnowledgeBase()  # For RAG agent

    all_results = []

    for i, test_case in enumerate(questions, 1):
        question = test_case["question"]
        category = test_case["category"]
        context = test_case["context"]

        print(f"\n{'='*100}")
        print(f"TEST CASE {i}: {question}")
        print(f"Category: {category}")
        print(f"{'='*100}\n")

        results_for_question = []

        # ======================================
        # Run Simple Agent
        # ======================================
        print("[1/3] Running Simple Agent...")
        try:
            simple_result = run_simple_agent(question, context)
            print(f"      Latency: {simple_result['latency']:.2f}s")
            print(f"      Answer: {simple_result['answer'][:80]}...")
            results_for_question.append(simple_result)
        except Exception as e:
            print(f"      ERROR: {e}")

        # ======================================
        # Run RAG Agent
        # ======================================
        print("\n[2/3] Running RAG Agent...")
        try:
            rag_result = run_rag_agent(question, kb)
            print(f"      Latency: {rag_result['latency']:.2f}s")
            print(f"      Answer: {rag_result['answer'][:80]}...")
            print(f"      Sources: {', '.join(rag_result.get('sources', []))}")
            results_for_question.append(rag_result)
        except Exception as e:
            print(f"      ERROR: {e}")

        # ======================================
        # Run Multi-Tool Agent
        # ======================================
        print("\n[3/3] Running Multi-Tool Agent...")
        try:
            multitool_result = run_multitool_agent(question)
            print(f"      Latency: {multitool_result['latency']:.2f}s")
            print(f"      Answer: {multitool_result['answer'][:80]}...")
            print(f"      Tools used: {multitool_result.get('tools_used', 0)}")
            results_for_question.append(multitool_result)
        except Exception as e:
            print(f"      ERROR: {e}")

        # Store results
        for result in results_for_question:
            all_results.append({
                "question": question,
                "category": category,
                **result
            })

    # ======================================
    # Generate Comparison Summary
    # ======================================
    print("\n\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)

    df = pd.DataFrame(all_results)

    # Average latency by agent type
    print("\nAVERAGE LATENCY BY AGENT TYPE:")
    print("-" * 100)
    latency_by_agent = df.groupby('agent_type')['latency'].mean()
    for agent_type, avg_latency in latency_by_agent.items():
        print(f"  {agent_type.upper():15s}: {avg_latency:6.2f}s")

    # Latency by category
    print("\nLATENCY BY QUESTION CATEGORY:")
    print("-" * 100)
    for category in df['category'].unique():
        print(f"\n  {category.upper()}:")
        category_df = df[df['category'] == category]
        for agent_type in ['simple', 'rag', 'multitool']:
            agent_data = category_df[category_df['agent_type'] == agent_type]
            if not agent_data.empty:
                avg = agent_data['latency'].mean()
                print(f"    {agent_type:10s}: {avg:6.2f}s")

    # Overall statistics
    print("\nOVERALL STATISTICS:")
    print("-" * 100)
    print(f"  Total test cases: {len(questions)}")
    print(f"  Total evaluations: {len(all_results)}")
    print(f"  Success rate: {len(all_results) / (len(questions) * 3) * 100:.1f}%")

    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 100)
    simple_avg = latency_by_agent.get('simple', 0)
    rag_avg = latency_by_agent.get('rag', 0)
    multitool_avg = latency_by_agent.get('multitool', 0)

    print(f"  1. Simple agent is FASTEST: {simple_avg:.2f}s average")
    print(f"  2. RAG agent is {(rag_avg/simple_avg - 1)*100:.1f}% slower (retrieval overhead)")
    print(f"  3. Multi-tool agent is {(multitool_avg/simple_avg - 1)*100:.1f}% slower (tool execution overhead)")
    print(f"  4. RAG agent provides SOURCE CITATIONS (better for factual questions)")
    print(f"  5. Multi-tool agent can PERFORM CALCULATIONS (simple/RAG cannot)")
    print(f"  6. Choose based on use case:")
    print(f"     - Simple questions? Use SIMPLE agent")
    print(f"     - Knowledge retrieval? Use RAG agent")
    print(f"     - Complex tasks? Use MULTI-TOOL agent")

    print("\n" + "=" * 100)
    print("Comparison complete! View detailed traces in MLflow UI:")
    print("  mlflow ui")
    print("  http://localhost:5000")
    print("=" * 100)

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results_df = run_comparison()

    # Save results to CSV for further analysis
    results_df.to_csv("agent_comparison_results.csv", index=False)
    print("\nResults saved to: agent_comparison_results.csv")
