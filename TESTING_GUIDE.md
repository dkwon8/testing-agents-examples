# Testing Guide: Running All Agent Levels

Quick guide to test all agents and see the comparisons.

## Step 1: Test Simple Agent (Level 1)

```bash
python examples/02_simple_agent.py
```

**What you'll see:**
- 2 questions answered
- Clean, fast responses
- ~3-4 seconds each

**Check in MLflow UI:**
- `simple_agent` span
- `call_openai` child span
- Simple execution path

---

## Step 2: Test RAG Agent (Level 2)

```bash
python examples/05_rag_agent.py
```

**What you'll see:**
- 4 questions answered
- Source citations shown
- Slightly slower (~5-7 seconds each)

**Check in MLflow UI:**
- `rag_agent` parent span
- `retrieve_documents` child span
- `format_context` child span
- `generate_answer` child span
- More complex execution path

---

## Step 3: Test Multi-Tool Agent (Level 3)

```bash
python examples/06_multitool_agent.py
```

**What you'll see:**
- 7 diverse questions
- Tool usage logged for each
- Multiple iterations for complex questions
- Slowest (~8-12 seconds each)

**Check in MLflow UI:**
- `multitool_agent` parent span
- Multiple `execute_tool` child spans
- Individual tool function spans
- Complex execution tree

---

## Step 4: Compare All Agents

```bash
python compare_agents.py
```

**What you'll see:**
- All 3 agents run on same questions
- Side-by-side latency comparison
- Summary statistics
- Key insights printed

**Output includes:**
- Average latency by agent type
- Latency by question category
- Performance trade-off analysis
- Results saved to CSV

---

## Step 5: Run Evaluation

```bash
python examples/04_agent_with_eval.py
```

**What you'll see:**
- Evaluation on 3 test questions
- Readability metrics (FK, ARI)
- Results in MLflow UI

---

## Expected Results Summary

### Latency Comparison
```
Simple Agent:     3-4s   (baseline)
RAG Agent:        5-7s   (~50% slower)
Multi-Tool Agent: 8-12s  (~150% slower)
```

### Cost Comparison
```
Simple Agent:     $0.002/query
RAG Agent:        $0.004/query  (2x more expensive)
Multi-Tool Agent: $0.008/query  (4x more expensive)
```

### Capability Comparison
```
Simple Agent:     Basic Q&A only
RAG Agent:        + Knowledge retrieval, source citation
Multi-Tool Agent: + Calculations, conversions, multi-step tasks
```

---

## Viewing in MLflow UI

1. **Start MLflow UI** (if not running):
   ```bash
   mlflow ui
   ```

2. **Open browser:**
   ```
   http://localhost:5000
   ```

3. **Navigate to Traces:**
   - Click "Traces" tab
   - You'll see traces from all agents
   - Filter by agent name to compare

4. **Compare execution paths:**
   - Simple: Single chain (agent → LLM)
   - RAG: Multi-step (agent → retrieve → format → LLM)
   - Multi-Tool: Complex tree (agent → tool → tool → LLM → tool)

---

## Troubleshooting

### Error: "No module named 'examples'"
```bash
# Make sure you're in the project root
cd simple-agent-example
python -m examples.05_rag_agent
```

### Error: "OPENAI_API_KEY not found"
```bash
# Check .env file exists
ls -la .env

# Make sure it contains your key
cat .env
```

### Slow response times
- Normal! Multi-tool agent can take 10-15 seconds for complex questions
- GPT-5.4 is slower but higher quality
- Use `gpt-4o-mini` for faster testing (edit model parameter)

---

## Next Steps After Testing

1. Review traces in MLflow UI
2. Compare metrics across agent types
3. Read `AGENT_LEVELS_EXPLAINED.md` for full understanding
4. Prepare presentation based on results
5. Move to Phase 2 (baseline comparison over time)

---

## Quick Test Checklist

- [ ] Simple agent runs successfully
- [ ] RAG agent runs and shows sources
- [ ] Multi-tool agent uses tools correctly
- [ ] Comparison script runs all three
- [ ] Evaluation script produces metrics
- [ ] All traces visible in MLflow UI
- [ ] No emojis in output (all removed)
- [ ] Ready to explain each level
