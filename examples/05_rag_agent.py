"""
Level 2: RAG (Retrieval-Augmented Generation) Agent

WHAT THIS IS:
A RAG agent retrieves relevant information from a knowledge base before answering.
This is the MOST COMMON pattern in production AI agents (80% of use cases).

HOW IT WORKS:
1. User asks a question
2. Agent searches knowledge base for relevant documents
3. Agent ranks/filters results by relevance
4. Agent synthesizes answer from retrieved documents
5. Agent cites sources used

WHY THIS MATTERS:
- More accurate than simple agents (grounded in real data)
- Can answer questions about specific documents/knowledge
- Traceable (can see what documents were used)
- Updatable (update knowledge base without retraining)

COMPLEXITY COMPARED TO SIMPLE AGENT:
- Simple: Question -> LLM -> Answer
- RAG: Question -> Retrieve -> Rank -> LLM with context -> Answer

To run:
    python examples/05_rag_agent.py
"""

import os
from openai import OpenAI
import mlflow
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()


# ============================================================================
# STEP 1: KNOWLEDGE BASE
# ============================================================================

class KnowledgeBase:
    """
    A simple in-memory knowledge base for demonstration.

    In production, this would be:
    - Vector database (Pinecone, Weaviate, Chroma)
    - Document store (Elasticsearch)
    - SQL database with full-text search

    For this demo, we use a simple list of documents.
    """

    def __init__(self):
        """Initialize with sample MLflow documentation"""
        self.documents = [
            {
                "id": "doc_001",
                "content": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It tackles three primary functions: tracking experiments, packaging code into reproducible runs, and sharing/deploying models.",
                "source": "mlflow_overview.md",
                "category": "introduction"
            },
            {
                "id": "doc_002",
                "content": "MLflow Tracking provides APIs and UI for logging parameters, code versions, metrics, and output files when running machine learning code. You can use Python, R, Java, and REST APIs to log runs.",
                "source": "mlflow_tracking.md",
                "category": "tracking"
            },
            {
                "id": "doc_003",
                "content": "MLflow Projects are a standard format for packaging reusable data science code. Each project is simply a directory with code or a Git repository, and uses a descriptor file to specify dependencies.",
                "source": "mlflow_projects.md",
                "category": "projects"
            },
            {
                "id": "doc_004",
                "content": "MLflow Models is a convention for packaging machine learning models in multiple formats. Each model is saved as a directory containing arbitrary files and a descriptor file that lists the available flavors.",
                "source": "mlflow_models.md",
                "category": "models"
            },
            {
                "id": "doc_005",
                "content": "The Model Registry component provides a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models. It provides model lineage, model versioning, stage transitions, and annotations.",
                "source": "mlflow_registry.md",
                "category": "registry"
            },
            {
                "id": "doc_006",
                "content": "MLflow Tracing provides observability into the execution of AI systems. It captures the inputs, outputs, and metadata of each step in a multi-step workflow, making it easier to debug and understand agent behavior.",
                "source": "mlflow_tracing.md",
                "category": "tracing"
            },
            {
                "id": "doc_007",
                "content": "To install MLflow, run: pip install mlflow. MLflow requires Python 3.8 or higher. For full functionality, install additional dependencies like scikit-learn, tensorflow, or pytorch.",
                "source": "mlflow_installation.md",
                "category": "setup"
            },
            {
                "id": "doc_008",
                "content": "MLflow evaluation provides tools to validate model quality, including metrics calculation, model comparison, and performance tracking. It supports both traditional ML metrics and GenAI-specific scorers.",
                "source": "mlflow_evaluation.md",
                "category": "evaluation"
            }
        ]

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple keyword-based search.

        In production, this would use:
        - Semantic search (embedding-based)
        - Vector similarity (cosine, dot product)
        - Hybrid search (keyword + semantic)

        For this demo, we use simple keyword matching.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of documents sorted by relevance
        """
        query_lower = query.lower()

        # Calculate relevance scores
        scored_docs = []
        for doc in self.documents:
            # Simple scoring: count keyword matches
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            scored_docs.append((score, doc))

        # Sort by score (highest first)
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Return top_k documents
        return [doc for score, doc in scored_docs[:top_k] if score > 0]


# ============================================================================
# STEP 2: RETRIEVAL FUNCTION
# ============================================================================

@mlflow.trace
def retrieve_documents(question: str, kb: KnowledgeBase, top_k: int = 3) -> List[Dict]:
    """
    Retrieve relevant documents from knowledge base.

    This function is traced so you can see:
    - What question was asked
    - What documents were retrieved
    - How long retrieval took

    Args:
        question: User's question
        kb: Knowledge base instance
        top_k: Number of documents to retrieve

    Returns:
        List of relevant documents
    """
    print(f"  Retrieving documents for: {question[:50]}...")

    # Perform search
    results = kb.search(question, top_k=top_k)

    print(f"  Retrieved {len(results)} documents")

    return results


# ============================================================================
# STEP 3: CONTEXT FORMATTING
# ============================================================================

@mlflow.trace
def format_context(documents: List[Dict]) -> str:
    """
    Format retrieved documents into context for the LLM.

    This function structures the context so the LLM knows:
    - What documents are available
    - Where each piece of information comes from
    - How to cite sources

    Args:
        documents: Retrieved documents

    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"[Document {i}] (Source: {doc['source']})\n{doc['content']}"
        )

    return "\n\n".join(context_parts)


# ============================================================================
# STEP 4: ANSWER GENERATION
# ============================================================================

@mlflow.trace
def generate_answer(question: str, context: str, model: str = "gpt-5.4") -> str:
    """
    Generate answer using LLM with retrieved context.

    This is similar to simple agent, but with structured context.

    Args:
        question: User's question
        context: Formatted context from retrieved documents
        model: OpenAI model to use

    Returns:
        Generated answer
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Structured prompt for RAG
    prompt = f"""You are a helpful assistant answering questions about MLflow.

Use the following documents to answer the question. If the answer is not in the documents, say so.
Always cite which document(s) you used (e.g., "According to Document 1...").

DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
        temperature=0.3  # Lower temperature for factual accuracy
    )

    return response.choices[0].message.content


# ============================================================================
# STEP 5: RAG AGENT ORCHESTRATION
# ============================================================================

@mlflow.trace
def rag_agent(question: str, kb: KnowledgeBase, top_k: int = 3, model: str = "gpt-5.4") -> Dict:
    """
    RAG Agent: Retrieval-Augmented Generation

    This is the main orchestration function that:
    1. Retrieves documents
    2. Formats context
    3. Generates answer
    4. Returns result with metadata

    Everything is traced so you can see the full pipeline.

    Args:
        question: User's question
        kb: Knowledge base instance
        top_k: Number of documents to retrieve
        model: OpenAI model to use

    Returns:
        Dict with answer, sources, and metadata
    """
    print(f"\nRAG Agent - Question: {question}")

    # Step 1: Retrieve
    documents = retrieve_documents(question, kb, top_k=top_k)

    # Step 2: Format context
    context = format_context(documents)

    # Step 3: Generate answer
    answer = generate_answer(question, context, model=model)

    print(f"Answer: {answer[:100]}...")

    # Return structured response
    return {
        "question": question,
        "answer": answer,
        "sources": [doc["source"] for doc in documents],
        "num_sources": len(documents),
        "model": model
    }


# ============================================================================
# STEP 6: TESTING & COMPARISON
# ============================================================================

if __name__ == "__main__":
    # Initialize knowledge base
    kb = KnowledgeBase()

    print("=" * 80)
    print("LEVEL 2: RAG AGENT DEMONSTRATION")
    print("=" * 80)

    # Test questions
    questions = [
        "What is MLflow?",
        "How do I install MLflow?",
        "What is MLflow Tracing used for?",
        "Tell me about MLflow Models and the Model Registry",
    ]

    print("\nRunning RAG Agent on test questions...\n")

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}: {question}")
        print(f"{'='*80}")

        result = rag_agent(question, kb, top_k=3)

        print(f"\nSOURCES USED:")
        for source in result["sources"]:
            print(f"  - {source}")

        print(f"\nFULL ANSWER:")
        print(result["answer"])

    print("\n" + "=" * 80)
    print("RAG Agent Demonstration Complete!")
    print("=" * 80)
    print("\nView traces in MLflow UI:")
    print("   Run: mlflow ui")
    print("   Open: http://localhost:5000")
    print("\nLook for:")
    print("   - rag_agent span (parent)")
    print("   - retrieve_documents span (child 1)")
    print("   - format_context span (child 2)")
    print("   - generate_answer span (child 3)")
    print("   - Full execution timeline")
    print("\nKey observations:")
    print("   - Total latency = retrieval + formatting + LLM")
    print("   - Context size affects LLM cost/latency")
    print("   - More documents = more accurate but slower")
