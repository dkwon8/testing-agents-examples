# Simple Agent with MLflow Tracing and Evaluation

A quick and comprehensive project demonstrating agent complexity levels, MLflow tracing, and evaluation for production AI systems.

## Project Overview

This quick little project builds three levels of AI agent complexity to understand trade-offs between speed, cost, and capability. Each level is fully traced with MLflow and evaluated with built-in scorers.

**Ultimate Goal:** Foundation for building continuous monitoring systems that detect quality drift, identify failure patterns, and enable self-healing for production agents.

## Agent Complexity Levels

### Level 1: Simple Q&A Agent
- **File:** `examples/02_simple_agent.py`
- **Architecture:** Question → LLM → Answer
- **Performance:** 3-4s latency, ~$0.002/query
- **Use Case:** Basic factual questions with provided context

### Level 2: RAG Agent (Retrieval-Augmented Generation)
- **File:** `examples/05_rag_agent.py`
- **Architecture:** Question → Retrieve docs → Rank → LLM with context → Answer
- **Performance:** 5-7s latency, ~$0.004/query
- **Use Case:** Knowledge base Q&A (most common in production)
- **Features:** Source citations, document search, grounded answers

### Level 3: Multi-Tool Agent
- **File:** `examples/06_multitool_agent.py`
- **Architecture:** Question → Analyze → Choose tools → Execute → Synthesize
- **Performance:** 8-12s latency, ~$0.008/query
- **Use Case:** Complex multi-step tasks requiring calculations, conversions, external data
- **Tools:** Calculator, Knowledge Search, DateTime, Unit Converter

## Project Structure

```
simple-agent-example/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .env                            # OpenAI API key (create from .env.example)
├── compare_agents.py               # Side-by-side agent comparison
├── AGENT_LEVELS_EXPLAINED.md       # Detailed documentation
├── TESTING_GUIDE.md                # How to run and test
└── examples/
    ├── 01_hello_trace.py           # MLflow tracing basics
    ├── 02_simple_agent.py          # Level 1: Simple Q&A
    ├── 03_agent_with_tools.py      # Tool use example
    ├── 04_agent_with_eval.py       # Evaluation with scorers
    ├── 05_rag_agent.py             # Level 2: RAG
    └── 06_multitool_agent.py       # Level 3: Multi-Tool
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run Examples

```bash
# Test Simple Agent (Level 1)
python examples/02_simple_agent.py

# Test RAG Agent (Level 2)
python examples/05_rag_agent.py

# Test Multi-Tool Agent (Level 3)
python examples/06_multitool_agent.py

# Compare all three
python compare_agents.py
```

### 3. View Traces in MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

## Key Features

- **MLflow Tracing:** Every agent execution is traced with inputs, outputs, latency, and execution flow
- **Built-in Evaluation:** Readability metrics (Flesch-Kincaid, ARI), token usage, latency tracking
- **Comparison Framework:** Side-by-side performance analysis across complexity levels
- **Production-Ready Patterns:** RAG and multi-tool patterns used in 80%+ of production agents

## Performance Comparison

| Agent Type | Latency | Cost/Query | Capability |
|------------|---------|------------|------------|
| Simple     | 3-4s    | $0.002     | Basic Q&A |
| RAG        | 5-7s    | $0.004     | + Knowledge retrieval, citations |
| Multi-Tool | 8-12s   | $0.008     | + Calculations, multi-step tasks |

## What's Covered

1. **MLflow Tracing:** How to capture agent execution for debugging and analysis
2. **Agent Patterns:** Simple Q&A, RAG, and multi-tool architectures
3. **Evaluation:** How to measure agent quality with built-in scorers
4. **Trade-offs:** Speed vs capability vs cost in production systems
5. **Production Readiness:** Patterns for continuous monitoring and quality assurance

## Next Steps: Continuous Monitoring

This project is Phase 1 of building a continuous monitoring system.

## Documentation

- **AGENT_LEVELS_EXPLAINED.md** - Detailed explanation of each complexity level
- **TESTING_GUIDE.md** - Step-by-step testing instructions
- **MLflow UI** - View traces and evaluation results at http://localhost:5000

## Requirements

- Python 3.8+
- OpenAI API key
- MLflow 2.15.0+
- See `requirements.txt` for full dependencies

## License

MIT
