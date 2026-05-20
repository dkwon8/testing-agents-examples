# Comprehensive Guide: Understanding Everything

This guide explains every concept, every trace, and every metric in complete detail.

---

## Part 1: Understanding MLflow Tracing

### What is Tracing?

**Tracing** is like recording a video of your code execution. Instead of just seeing the final result, you see EVERY step that happened along the way.

**Analogy:** Think of it like a security camera recording everything that happens in a building. Later, if something goes wrong, you can review the footage to see exactly what happened and when.

### Why Do We Need Tracing for AI Agents?

**Problem without tracing:**
```
User: "What is MLflow?"
Agent: "MLflow is a database."  (WRONG!)

Why did it give a wrong answer?
- Was the prompt bad?
- Did retrieval find the wrong documents?
- Did the LLM hallucinate?
- How long did each step take?

WITHOUT TRACING: You have no idea. It's a black box.
```

**With tracing:**
```
User: "What is MLflow?"

TRACE SHOWS:
1. simple_agent() started at 14:03:45.123
   - Input: question = "What is MLflow?"
   
2. call_openai() started at 14:03:45.125 (2ms later)
   - Input: prompt = "What is MLflow?"
   - Model: gpt-5.4
   - Max tokens: 1024
   
3. OpenAI API call: 3.340 seconds
   - Tokens used: 186
   - Cost: $0.002165
   
4. call_openai() returned at 14:03:48.465
   - Output: "MLflow is an open-source platform..."
   
5. simple_agent() returned at 14:03:48.467
   - Total duration: 3.344 seconds
   - Answer: "MLflow is an open-source platform..."

NOW YOU KNOW EXACTLY:
- What function was called
- What inputs it received
- How long each step took
- What outputs it produced
- Where 99.9% of time was spent (the API call)
```

### The @mlflow.trace Decorator

**What it does:**

```python
@mlflow.trace
def my_function(x, y):
    result = x + y
    return result
```

When you add `@mlflow.trace`, MLflow automatically:
1. Records when the function starts
2. Captures all input parameters (x, y)
3. Times how long the function runs
4. Captures the output/return value
5. Handles errors (records exceptions if they happen)
6. Creates a "span" (a unit of work in the trace)

**Spans Explained:**

A **span** represents one unit of work. Think of it like a task.

```
Parent Span: simple_agent()
  └─ Child Span: call_openai()
      └─ Grandchild Span: OpenAI API network call
```

**Nested Spans:**

When one traced function calls another traced function, you get a hierarchy:

```python
@mlflow.trace
def parent():
    child()  # This creates a child span
    return "done"

@mlflow.trace
def child():
    # Some work
    return "child done"
```

Result in MLflow UI:
```
parent (total: 5s)
  └─ child (total: 3s)

Interpretation: Parent took 5s total, but 3s of that was spent in child,
so parent's own work was only 2s.
```

---

## Part 2: Understanding Each Agent Level

### Level 1: Simple Agent - Line by Line Explanation

**File:** `examples/02_simple_agent.py`

```python
@mlflow.trace
def call_openai(prompt: str, model: str = "gpt-5.4") -> str:
```
**What this does:**
- `@mlflow.trace` = Record this function's execution
- `prompt: str` = The question we're asking (typed as string)
- `model: str = "gpt-5.4"` = Which OpenAI model to use (default gpt-5.4)
- `-> str` = This function returns a string

```python
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```
**What this does:**
- Creates an OpenAI API client
- Gets API key from environment variable (.env file)
- This client will make API calls to OpenAI

```python
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1024
    )
```
**What this does:**
- Calls OpenAI's chat completion API
- `model=model` = Use gpt-5.4
- `messages=[...]` = The conversation history (just our question for simple agent)
- `"role": "user"` = This message is from the user (not the assistant)
- `"content": prompt` = The actual question text
- `max_completion_tokens=1024` = Don't generate more than 1024 tokens (GPT-5.4 requires this parameter instead of max_tokens)

**This is the slowest part** - making the actual API call. Usually takes 3-4 seconds.

```python
    return response.choices[0].message.content
```
**What this does:**
- OpenAI returns multiple possible completions (called "choices")
- We take the first one: `choices[0]`
- Extract the message content (the actual answer text)

**Why "choices[0]"?**
- OpenAI can generate multiple responses (if you set n=2, you get 2 choices)
- We always request just 1, so we take choices[0]

```python
@mlflow.trace
def simple_agent(question: str, model: str = "gpt-5.4") -> dict:
```
**What this does:**
- Main orchestration function
- Takes a question, returns a dictionary with results
- Also traced by MLflow

```python
    print(f"\nQuestion: {question}")
    answer = call_openai(question, model=model)
    print(f"Answer: {answer}")
```
**What this does:**
- Prints the question (for console output)
- Calls our traced `call_openai` function
  - This creates a NESTED span in MLflow
  - Parent: simple_agent
  - Child: call_openai
- Prints the answer

```python
    return {
        "question": question,
        "answer": answer,
        "model": model
    }
```
**What this does:**
- Returns a dictionary (Python dict) with structured data
- Makes it easy to access: `result["question"]`, `result["answer"]`, etc.

**When this runs, MLflow creates this trace:**
```
simple_agent (3.34s)
  ├─ Input: {"question": "What is MLflow?", "model": "gpt-5.4"}
  ├─ call_openai (3.33s)
  │   ├─ Input: {"prompt": "What is MLflow?", "model": "gpt-5.4"}
  │   ├─ API Call: 3.33s, 186 tokens, $0.002
  │   └─ Output: "MLflow is an open-source platform..."
  └─ Output: {"question": "...", "answer": "...", "model": "gpt-5.4"}
```

---

### Level 2: RAG Agent - Detailed Walkthrough

**File:** `examples/05_rag_agent.py`

**What makes RAG different:**

Instead of just asking the LLM directly, we:
1. Search our knowledge base for relevant documents
2. Give those documents to the LLM as context
3. LLM answers based on the documents

**Why this is better:**
- More accurate (grounded in real documents)
- Can cite sources (shows which document the answer came from)
- Can handle large knowledge bases (millions of docs)
- Updatable (add new docs without retraining)

**The Knowledge Base:**

```python
class KnowledgeBase:
    def __init__(self):
        self.documents = [
            {
                "id": "doc_001",
                "content": "MLflow is an open-source platform...",
                "source": "mlflow_overview.md",
                "category": "introduction"
            },
            # ... more documents
        ]
```

**What this is:**
- A simple in-memory list of documents
- Each document has:
  - `id`: Unique identifier
  - `content`: The actual text
  - `source`: Where it came from (filename)
  - `category`: Type of document

**In production, this would be:**
- Vector database (Pinecone, Weaviate, Chroma)
- Elasticsearch
- SQL database with full-text search

**The Search Function:**

```python
@mlflow.trace
def retrieve_documents(question: str, kb: KnowledgeBase, top_k: int = 3) -> List[Dict]:
```

**What top_k means:**
- top_k = 3 means "return the 3 most relevant documents"
- If top_k = 5, return 5 documents
- More documents = more context but slower + more expensive

**How search works (simplified):**

```python
def search(self, query: str, top_k: int = 3) -> List[Dict]:
    query_lower = query.lower()
    
    # Score each document
    scored_docs = []
    for doc in self.documents:
        content_lower = doc["content"].lower()
        # Count how many query words appear in document
        score = sum(1 for word in query_lower.split() if word in content_lower)
        scored_docs.append((score, doc))
    
    # Sort by score (highest first)
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    
    # Return top 3
    return [doc for score, doc in scored_docs[:top_k]]
```

**Example:**

Query: "How do I install MLflow?"

Scoring:
```
doc_007 (installation.md): score=3 (matches: "install", "mlflow", "how")
doc_001 (overview.md): score=1 (matches: "mlflow")
doc_002 (tracking.md): score=1 (matches: "mlflow")

Top 3 returned: [doc_007, doc_001, doc_002]
```

**Format Context:**

```python
@mlflow.trace
def format_context(documents: List[Dict]) -> str:
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(
            f"[Document {i}] (Source: {doc['source']})\n{doc['content']}"
        )
    return "\n\n".join(context_parts)
```

**What this produces:**

```
[Document 1] (Source: mlflow_installation.md)
To install MLflow, run: pip install mlflow. Requires Python 3.8+.

[Document 2] (Source: mlflow_overview.md)
MLflow is an open-source platform for managing ML lifecycles.

[Document 3] (Source: mlflow_tracking.md)
MLflow Tracking provides APIs for logging experiments.
```

**Why this format?**
- Clear document boundaries
- Shows source for each document
- LLM can cite "According to Document 1..."

**Generate Answer:**

```python
prompt = f"""You are a helpful assistant answering questions about MLflow.

Use the following documents to answer the question. If the answer is not in the documents, say so.
Always cite which document(s) you used (e.g., "According to Document 1...").

DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:"""
```

**Why this prompt structure?**
- Clear instructions for the LLM
- Tells it to cite sources
- Tells it to admit if answer isn't in documents (reduces hallucination)
- Provides documents first, then question

**The Full RAG Flow:**

```
User: "How do I install MLflow?"

Step 1: retrieve_documents()
  - Searches knowledge base
  - Finds 3 most relevant docs
  - Returns: [doc_007, doc_001, doc_002]
  - Time: ~0.1s (in-memory search is fast)

Step 2: format_context()
  - Structures documents for LLM
  - Creates formatted string
  - Time: ~0.001s (string formatting is instant)

Step 3: generate_answer()
  - Builds prompt with context
  - Calls OpenAI API
  - LLM reads documents and synthesizes answer
  - Time: ~4-5s (LLM call)

Step 4: Return result
  - Answer: "According to Document 1, run: pip install mlflow"
  - Sources: ["mlflow_installation.md", "mlflow_overview.md", ...]
  - Total time: ~5-7s
```

**MLflow Trace for RAG:**

```
rag_agent (5.34s)
  ├─ retrieve_documents (0.12s)
  │   ├─ Searched 8 documents
  │   └─ Returned 3 matches
  ├─ format_context (0.001s)
  │   └─ Formatted 3 documents
  └─ generate_answer (5.21s)
      ├─ Built prompt with context (512 tokens)
      ├─ API call (5.20s)
      └─ Returned answer (with citation)
```

---

### Level 3: Multi-Tool Agent - Deep Dive

**File:** `examples/06_multitool_agent.py`

**What makes this different:**

The agent has access to MULTIPLE tools and must DECIDE which ones to use.

**The Tools:**

1. **Calculator** - Math operations
2. **Knowledge Search** - Look up information
3. **DateTime** - Get current date/time
4. **Unit Converter** - Convert measurements

**Tool Definitions (Schema):**

```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Performs basic mathematical operations...",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", ...]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    # ... more tools
]
```

**What this schema does:**
- Tells OpenAI what tools exist
- Describes what each tool does
- Specifies what parameters each tool needs
- OpenAI reads this and decides which tool to call

**The Agent Loop:**

```python
while iteration < max_iterations:
    iteration += 1
    
    # Ask LLM what to do
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto"
    )
    
    tool_calls = response.choices[0].message.tool_calls
    
    # If no tools needed, we're done
    if not tool_calls:
        final_answer = response.message.content
        break
    
    # Execute each tool
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_result = execute_tool(tool_name, tool_args)
        
        # Add result to conversation
        messages.append(tool_result)
```

**How the Agent Decides:**

**Example: "What is 25 × 4?"**

Iteration 1:
```
LLM thinks: "I need to multiply two numbers. I have a calculator tool."
LLM decides: tool_call = {
    "function": "calculator",
    "arguments": {"operation": "multiply", "a": 25, "b": 4}
}

We execute: calculator(multiply, 25, 4) → 100

We add to conversation: "The calculator returned 100"
```

Iteration 2:
```
LLM thinks: "I have the answer now: 100"
LLM decides: No more tools needed
LLM returns: "25 multiplied by 4 equals 100"

We break out of loop and return final answer.
```

**Complex Example: "Convert 100 meters to feet, then multiply by 2"**

Iteration 1:
```
LLM thinks: "I need to convert units first"
LLM decides: convert_units(value=100, from_unit="meters", to_unit="feet")
Result: 328.08 feet
```

Iteration 2:
```
LLM thinks: "Now I have 328.08, need to multiply by 2"
LLM decides: calculator(operation="multiply", a=328.08, b=2)
Result: 656.16
```

Iteration 3:
```
LLM thinks: "I have the final answer"
LLM returns: "100 meters equals 328.08 feet. Multiplied by 2 gives 656.16 feet."
```

**MLflow Trace for Multi-Tool:**

```
multitool_agent (9.87s)
  ├─ Iteration 1 (4.23s)
  │   ├─ LLM decides to use converter (2.34s)
  │   └─ execute_tool: convert_units (0.001s)
  ├─ Iteration 2 (3.45s)
  │   ├─ LLM decides to use calculator (2.21s)
  │   └─ execute_tool: calculator (0.001s)
  └─ Iteration 3 (2.19s)
      └─ LLM synthesizes final answer (2.19s)
```

**Why it's slower:**
- Each iteration requires an LLM call (2-3s each)
- Tool execution is fast (<0.01s)
- But we need multiple LLM calls for multi-step reasoning

---

## Part 3: Understanding Evaluation Metrics

**File:** `examples/04_agent_with_eval.py`

### What is Evaluation?

**Evaluation** = Measuring how GOOD your agent's responses are.

**Analogy:** Like grading a student's test answers.
- You have the correct answer (ground truth)
- You have the student's answer
- You measure how close they are

### The Evaluation Dataset

```python
test_data = pd.DataFrame({
    "question": [
        "What is MLflow?",
        "How does MLflow tracing work?",
    ],
    "context": [
        "MLflow is an open-source platform...",
        "MLflow tracing captures execution...",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing ML lifecycles.",
        "MLflow tracing records function calls with inputs and outputs.",
    ]
})
```

**What each column means:**

1. **question** - What we ask the agent
2. **context** - Background information provided
3. **ground_truth** - The "correct" answer we expect

### Running Evaluation

```python
mlflow.evaluate(
    model=model,
    data=test_data,
    targets="ground_truth",
    model_type="question-answering",
    evaluators="default"
)
```

**What happens:**
1. MLflow runs your agent on each question
2. Compares agent's answer to ground_truth
3. Calculates metrics
4. Saves results

### Metric 1: Flesch-Kincaid Grade Level

**What it measures:** Reading difficulty / complexity of text

**Formula (simplified):**
```
FK = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
```

**Scale:**
- 6 = 6th grade reading level (11-12 year olds)
- 12 = High school senior
- 16 = College senior
- 18+ = Graduate school

**Your score: 16.45**

**Interpretation:**
- Your agent writes at college senior level
- Appropriate for technical ML audience
- Not too simple (would be <10)
- Not overly academic (would be >20)

**Why it matters:**
- Too simple = oversimplified, not detailed enough
- Too complex = hard to understand, loses users
- Want to match your target audience

**Example:**

Grade 6 text:
```
"MLflow helps you track ML work. It's easy to use."
```

Grade 16 text (your agent):
```
"MLflow is an open-source platform for managing the end-to-end machine
learning lifecycle, encompassing experiment tracking, model packaging,
and deployment capabilities."
```

Graduate level (20+):
```
"MLflow constitutes a comprehensive framework facilitating the orchestration
and operationalization of machine learning pipelines through systematic
experiment tracking, reproducible model artifacts, and scalable deployment
infrastructures."
```

### Metric 2: ARI (Automated Readability Index)

**What it measures:** Text complexity (similar to FK but different formula)

**Formula:**
```
ARI = 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
```

**Your score: 18.28**

**Why it's higher than FK:**
- Uses character count instead of syllables
- Different weighting
- Both together give complete picture

**Interpretation:**
- Confirms FK score (both say "college level")
- If FK and ARI disagree significantly, investigate why

### Metric 3: Exact Match

**What it measures:** Does answer EXACTLY match ground truth word-for-word?

**Your score: 0.0 (0%)**

**Why this is OK:**

Ground truth:
```
"MLflow is an open-source platform for managing machine learning lifecycles,
including experiment tracking and model deployment."
```

Agent answer:
```
"MLflow is an open-source platform for managing the end-to-end machine
learning lifecycle. It focuses on three main functions: tracking experiments,
packaging code into reproducible runs, and sharing/deploying models."
```

**Exact match: 0%**
(Different words, different structure)

**Semantic match: ~90%**
(Same meaning, same information)

**Why exact match doesn't matter for Q&A:**
- We WANT the agent to use its own words
- Parroting = bad (memorization, not understanding)
- Semantic similarity = what actually matters

**When exact match DOES matter:**
- Code generation (syntax must be exact)
- Legal/compliance text (exact wording required)
- Form filling (specific format needed)

### Metric 4: Token Count (not shown but tracked)

**What it measures:** How many tokens (word pieces) were used

**Example breakdown:**

Question: "What is MLflow?" = ~5 tokens
Context: "MLflow is an open-source platform..." = ~30 tokens  
Answer: "MLflow is an open-source..." = ~40 tokens

Total: ~75 tokens

**Your typical usage: 186 tokens**

**Why it matters:**
- Cost = tokens × price per token
- GPT-5.4 pricing: ~$0.01 per 1000 tokens
- 186 tokens = $0.002 (very cheap!)
- But at scale: 1M queries = $2,000

**Optimization:**
- Shorter prompts = fewer tokens = lower cost
- But don't sacrifice quality for cost

### Metric 5: Latency

**What it measures:** How long the agent took to respond

**Your trace showed: 3.34s**

**Breakdown:**
- Your code overhead: 0.01s
- OpenAI API call: 3.33s

**Why it matters:**
- User experience: >5s feels slow
- Production SLA: Often <3s required
- Cost: Some APIs charge per second

**Acceptable latency:**
- Chatbot: <3s ideal, <5s acceptable
- Background task: <30s acceptable
- Batch processing: Minutes OK

### Metric 6: Variance (Consistency)

**What it measures:** How much variation between responses

**Your scores:**
- FK variance: 0.13
- ARI variance: 0.06

**Low variance (like yours) = Very consistent**

All 3 answers had similar complexity:
- Answer 1: FK = 16.3
- Answer 2: FK = 16.5
- Answer 3: FK = 16.6

Range: 16.3 to 16.6 (very tight!)

**Why consistency matters:**
- Predictable user experience
- Stable quality
- If variance is high (e.g., 5.0):
  - Some answers grade 11, others grade 21
  - Inconsistent behavior
  - Hard to trust

---

## Part 4: Viewing in MLflow UI - Complete Guide

### Starting MLflow UI

```bash
mlflow ui
# Opens server at http://localhost:5000
```

### Main Page: Experiments

**What you see:**
- List of all evaluation runs
- Each run has a unique ID
- Shows when it ran, what model, etc.

### Traces Tab

**Click "Traces" in top navigation**

**What you see:**
- Every traced function execution
- List view showing:
  - Trace name (e.g., "simple_agent")
  - Status (OK, Error)
  - Duration
  - Timestamp

**Click on a trace to see details:**

### Trace Detail View

**Left side: Execution Timeline**

```
simple_agent (3.34s)
  └─ call_openai (3.33s)
      └─ API request (3.33s)
```

**What the timeline shows:**
- Bars representing time
- Longer bar = took more time
- Nested bars = function calls within functions

**Click on a span:**

### Span Details

**Top section: Metadata**
- Status: OK
- Duration: 3.34s
- Token count: 186
- Cost: $0.002

**Inputs tab:**
```json
{
  "question": "What is MLflow?",
  "model": "gpt-5.4"
}
```
Shows exactly what was passed in

**Outputs tab:**
```json
{
  "answer": "MLflow is an open-source platform...",
  "model": "gpt-5.4"
}
```
Shows exactly what was returned

**Attributes tab:**
- Additional metadata
- Model used
- Temperature
- Max tokens

**Events tab:**
- Log messages
- Warnings
- Errors (if any)

### Comparing Traces

**Use case:** Compare simple vs RAG vs multi-tool

**How:**
1. Click on trace for simple agent
2. Note duration: 3.34s
3. Note structure: single → LLM → done
4. Click on trace for RAG agent
5. Note duration: 5.67s
6. Note structure: retrieve → format → LLM
7. Compare: RAG is 70% slower but has retrieval

### Evaluation Runs Tab

**Shows evaluation results:**
- Metrics table
- Charts comparing runs
- Per-question breakdown

**Click on an evaluation run:**

**Metrics:**
```
flesch_kincaid_grade_level/v1/mean: 16.45
ari_grade_level/v1/mean: 18.28
exact_match/v1: 0.0
```

**Per-question results:**
- Question 1: FK=16.3, ARI=18.1
- Question 2: FK=16.5, ARI=18.4
- Question 3: FK=16.6, ARI=18.3

### Using MLflow for Debugging

**Scenario: Agent gives wrong answer**

**Steps:**
1. Find the trace for that question
2. Open trace details
3. Check inputs: Was question correct?
4. Check RAG retrieval: Did it find right docs?
5. Check LLM prompt: Was context provided?
6. Check output: What did LLM actually say?
7. Check latency: Did it timeout/fail?

**Example investigation:**

```
Question: "How do I install MLflow?"
Answer: "MLflow is a database." (WRONG!)

1. Open trace
2. Check retrieve_documents span
   - Retrieved: [database_doc.md, cloud_doc.md]
   - PROBLEM: Didn't retrieve installation.md!
   
3. Root cause: Search ranked database higher
   
4. Fix: Improve search scoring
```

---

## Part 5: What It All Means for Production

### Why We Built 3 Levels

**Real-world decision tree:**

"I need an agent. Which level do I build?"

**If your use case is:**
- Simple Q&A with context provided → Level 1
- Search documents and answer → Level 2 (RAG)
- Multi-step tasks, calculations → Level 3

**Trade-off matrix:**

```
         Speed    Cost    Capability    Complexity
Level 1   Best    Best      Limited        Low
Level 2   Good    Good        Good        Medium  
Level 3   Worst   Worst       Best         High
```

### What We Learned from Metrics

**FK = 16.45, ARI = 18.28:**
- Agent writes at college level
- Good for technical audience (ML engineers, data scientists)
- If building for general public, would need to simplify

**Variance = 0.13:**
- Very consistent behavior
- Users get predictable experience
- Ready for production (from consistency standpoint)

**Latency = 3.34s:**
- Acceptable for chatbot (<5s target)
- Could be faster with caching
- Most time is LLM (can't optimize much)

**Cost = $0.002/query:**
- Very cheap
- At 1M queries/month: $2,000
- Scales linearly
- Could reduce with cheaper model (gpt-4o-mini)

### Next Steps: Continuous Monitoring

**Why we need it:**

Day 1: Everything works great
```
FK: 16.45
Latency: 3.34s
Correctness: 95%
```

Day 100: Quality has degraded
```
FK: 14.2 (simpler answers)
Latency: 8.5s (slower)
Correctness: 72% (less accurate)
```

**Without monitoring:**
- You don't notice until users complain
- No idea when degradation started
- Hard to find root cause
- Reactive firefighting

**With monitoring (what we're building):**
- Detect on Day 45 when drift starts
- Alert: "FK dropped 10%"
- Triage: "Retrieval returning wrong docs"
- Auto-fix: "Re-index knowledge base"
- Validate: "FK back to 16.2"

**This is the ultimate goal of your project.**

---

## Summary

You now understand:

**Tracing:**
- What it records (inputs, outputs, timing, structure)
- Why it matters (debugging, monitoring, optimization)
- How to read traces in MLflow UI

**Agents:**
- Level 1: Simple Q&A (fast, cheap, limited)
- Level 2: RAG (most common, retrieval + LLM)
- Level 3: Multi-tool (complex, powerful, slow)

**Evaluation:**
- FK: Reading difficulty (~16 = college)
- ARI: Confirms FK with different formula
- Exact match: Word-for-word (doesn't matter for Q&A)
- Token count: Cost driver
- Latency: Speed
- Variance: Consistency

**Production:**
- Trade-offs between speed/cost/capability
- Metrics tell you if agent is production-ready
- Monitoring needed to maintain quality over time

You're ready to present this work and move to Phase 2: continuous monitoring!
