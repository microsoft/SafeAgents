# Running Datasets

Learn how to run benchmarks and datasets with SafeAgents.

---

## Overview

SafeAgents makes it easy to run agent benchmarks like AgentHarm and ASB with:
- ✅ Automatic checkpointing and resume
- ✅ Progress tracking
- ✅ Built-in safety assessment (ARIA, DHARMA)
- ✅ Summary statistics with score distributions
- ✅ Support for all frameworks and architectures

---

## Quick Start

```python
from safeagents import Dataset
from dotenv import load_dotenv

load_dotenv()

# Load AgentHarm dataset
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    split="test_public",
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2]  # Run first 3 tasks
).load()

# Run with automatic assessment
results = dataset.run(
    assessment=["aria", "dharma"],
    progress_bar=True,
    verbose=False
)

# Print summary with score distributions
dataset.print_summary()
```

**Output:**
```
[SafeAgents] Running 3 tasks
Processing tasks: 100%|████████| 3/3 [00:45<00:00, 15.2s/it]

================================================================================
DATASET RUN SUMMARY
================================================================================
Total tasks: 3
Successful: 3
Errors: 0

   ARIA Score Distribution
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Score ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│   1   │     2 │      66.7% │
│   4   │     1 │      33.3% │
└───────┴───────┴────────────┘
```

---

## Supported Datasets

### 1. AgentHarm

arxiv link: https://arxiv.org/abs/2410.09024

AI safety benchmark with harmful and benign tasks.

```python
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",  # or "harmless_benign", "chat"
    split="test_public",  # or "validation"
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2, 3, 4]  # Optional: specific tasks
).load()
```

**Config Options:**
- `"harmful"` - Tasks with harmful objectives (default)
- `"harmless_benign"` - Benign/safe tasks
- `"chat"` - Chat-based tasks

**Split Options:**
- `"test_public"` - Public test set
- `"validation"` - Validation set

### 2. ASB (Agent Security Benchmark)

arxiv link: https://arxiv.org/abs/2410.02644


Comprehensive agent safety evaluation.

```python
dataset = Dataset(
    name="asb",  # or "agent-safety-benchmark"
    config="financial_analyst_agent",  # Optional: agent name filter
    framework="langgraph",
    architecture="decentralized",
    indices=[0, 1, 2]
).load()
```

**Config Options:**
- Specific agent name (e.g., `"financial_analyst_agent"`)
- `None` - Run all agents

---

## Basic Usage

### Step 1: Create Dataset

```python
from safeagents import Dataset

dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    split="test_public",
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2, 3, 4],  # Run 5 tasks
    log_dir="./logs/my_experiment"  # Optional: custom log directory
)
```

### Step 2: Load Dataset

```python
dataset.load()
```

This loads the dataset and prepares tasks. Dataset handlers are auto-imported!

### Step 3: Run Tasks

```python
results = dataset.run(
    assessment=["aria", "dharma"],  # Optional: safety assessment
    progress_bar=True,  # Show progress
    verbose=False,  # Don't print logs during execution
    resume=True  # Resume from checkpoint if interrupted
)
```

### Step 4: View Results

```python
# Print formatted summary
dataset.print_summary()

# Or get programmatic access
summary = dataset.get_summary()
print(f"Total tasks: {summary['total_tasks']}")
print(f"Errors: {summary['errors']}")
```

---

## Advanced Features

### Checkpoint and Resume

Long-running experiments can be interrupted and resumed:

```python
# First run (may be interrupted)
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    indices=range(100),  # 100 tasks
    log_dir="./logs/big_experiment"
).load()

results = dataset.run(
    resume=True,  # Enable resume
    save_checkpoints=True  # Save after each task
)

# ... interrupted at task 47 ...

# Later: Resume from where it left off
dataset2 = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    indices=range(100),
    log_dir="./logs/big_experiment"  # Same log directory!
).load()

results = dataset2.run(
    resume=True  # Continues from task 47
)
```

**How it works:**
- After each task, a checkpoint is saved to `log_dir/checkpoint.json`
- On resume, completed task indices are loaded
- Only remaining tasks are executed

### Progress Tracking

```python
results = dataset.run(
    progress_bar=True  # Show tqdm progress bar
)
```

**Output:**
```
[SafeAgents] Running 100 tasks
Processing tasks: 47%|████▋     | 47/100 [12:34<13:22, 15.1s/it]
```

### Custom Log Directory

```python
from datetime import datetime

# Timestamped log directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    indices=[0, 1, 2],
    log_dir=f"./logs/agentharm_{timestamp}"
).load()
```

If not specified, SafeAgents creates a timestamped directory automatically.

### Specify Indices

Run specific tasks:

```python
# Run first 10 tasks
indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Or use range
indices=range(10)

# Run specific tasks
indices=[0, 5, 10, 15, 20]

# Run all tasks
indices=None  # or omit the parameter
```

---

## Results and Logging

### Result Structure

Each task result contains:

```python
{
    'task_index': 0,
    'task_prompt': 'Task description',
    'logs': 'Full execution logs',
    'messages': [...],  # All messages
    'stop_reason': 'max_turns',
    'execution_trace': {
        'tool_calls': [...],
        'bash_commands': [...]
    },
    'assessment': {
        'aria': '1',  # If assessment requested
        'dharma': 'A'
    },
    'attack_detected': False,  # If detector used
    'framework': 'openai-agents',
    'architecture': 'centralized'
}
```

### Log Files

Results are saved in the log directory:

```
./logs/AgentHarm_20250102_143022/
├── task_0.json          # Individual task results
├── task_1.json
├── task_2.json
├── results.json         # All results combined
├── results.jsonl        # Results in JSONL format
└── checkpoint.json      # Checkpoint data
```

### Accessing Results

```python
# During execution
results = dataset.run(...)
for result in results:
    print(f"Task {result['task_index']}: {result['assessment']}")

# After execution from memory
summary = dataset.get_summary()
print(summary['assessment']['aria']['scores'])

# After execution from disk
import json
with open('./logs/my_experiment/results.json') as f:
    results = json.load(f)
```

---

## Summary Statistics

### Print Summary

```python
dataset.print_summary()
```

**Output with `rich` library installed:**
```
================================================================================
DATASET RUN SUMMARY
================================================================================
Total tasks: 10
Successful: 9
Errors: 1
Framework: openai-agents
Architecture: centralized

Assessment Score Distributions:
   ARIA Score Distribution
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Score ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│   1   │     5 │      55.6% │
│   2   │     2 │      22.2% │
│   4   │     2 │      22.2% │
└───────┴───────┴────────────┘

  DHARMA Score Distribution
┏━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Score ┃ Count ┃ Percentage ┃
┡━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│   A   │     4 │      44.4% │
│   B   │     3 │      33.3% │
│   D   │     2 │      22.2% │
└───────┴───────┴────────────┘

Results saved to: ./logs/AgentHarm_20250102_143022
================================================================================
```

### Get Summary Programmatically

```python
summary = dataset.get_summary()

print(f"Total: {summary['total_tasks']}")
print(f"Successful: {summary['successful']}")
print(f"Errors: {summary['errors']}")

# Access assessment scores
if 'assessment' in summary:
    aria_scores = summary['assessment']['aria']['scores']
    print(f"ARIA scores: {aria_scores}")
```

### Loading Summary from Checkpoint

Even without running the dataset, you can load previous results:

```python
# Point to a previous run
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    framework="openai-agents",
    architecture="centralized",
    log_dir="./logs/AgentHarm_20250102_143022"  # Previous run
)

# No need to load() or run()!
dataset.print_summary()  # Loads results from disk
```

---

## Complete Example

```python
from safeagents import Dataset
from dotenv import load_dotenv

load_dotenv()

# Configuration
FRAMEWORK = "openai-agents"
ARCHITECTURE = "centralized"
NUM_TASKS = 10

# Create dataset
dataset = Dataset(
    name="ai-safety-institute/AgentHarm",
    config="harmful",
    split="test_public",
    framework=FRAMEWORK,
    architecture=ARCHITECTURE,
    indices=range(NUM_TASKS),
    log_dir=f"./logs/experiment_{FRAMEWORK}"
).load()

print(f"[Experiment] Running {NUM_TASKS} tasks with {FRAMEWORK}")

# Run with all features
results = dataset.run(
    assessment=["aria", "dharma"],
    progress_bar=True,
    verbose=False,
    resume=True,
    save_checkpoints=True
)

# Print summary
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
dataset.print_summary()

# Save additional analysis
summary = dataset.get_summary()
print(f"\n[Analysis] Success rate: {summary['successful'] / summary['total_tasks'] * 100:.1f}%")

if 'assessment' in summary:
    aria_scores = summary['assessment']['aria']['scores']
    harmful_count = aria_scores.count('4')
    print(f"[Analysis] Harmful responses (ARIA=4): {harmful_count}/{len(aria_scores)}")
```

---

## Best Practices

### 1. Use Checkpointing for Long Runs

```python
results = dataset.run(
    resume=True,  # Always enable for long experiments
    save_checkpoints=True
)
```

### 2. Organize Experiments with Log Directories

```python
# Descriptive log directories
log_dir=f"./logs/{framework}_{architecture}_{config}_{timestamp}"
```

### 3. Run Subsets First

Test with small subsets before running full datasets:

```python
# Test with 5 tasks first
indices=[0, 1, 2, 3, 4]

# Then run full dataset
indices=None
```

### 4. Monitor Progress

```python
results = dataset.run(
    progress_bar=True,  # See progress
    verbose=False  # But don't spam logs
)
```

### 5. Save Results Programmatically

```python
import json

results = dataset.run(...)

# Save custom analysis
analysis = {
    'framework': FRAMEWORK,
    'total_tasks': len(results),
    'aria_scores': [r['assessment']['aria'] for r in results if 'assessment' in r]
}

with open(f'{dataset.log_dir}/analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)
```

---

## Comparing Frameworks

Run the same dataset with different frameworks:

```python
from safeagents import Dataset

FRAMEWORKS = ["autogen", "langgraph", "openai-agents"]
INDICES = range(10)

for framework in FRAMEWORKS:
    print(f"\n{'='*80}")
    print(f"Running with {framework}")
    print(f"{'='*80}")

    dataset = Dataset(
        name="ai-safety-institute/AgentHarm",
        config="harmful",
        framework=framework,
        architecture="centralized",
        indices=INDICES,
        log_dir=f"./logs/comparison_{framework}"
    ).load()

    results = dataset.run(
        assessment=["aria"],
        progress_bar=True
    )

    dataset.print_summary()
```

Then compare results across `./logs/comparison_*` directories.

---

## Troubleshooting

### Issue: "No handler registered"

**Solution:** Dataset handlers are auto-imported. Make sure you're using the correct dataset name:

```python
# ✅ Correct
name="ai-safety-institute/AgentHarm"
name="asb"

# ❌ Wrong
name="AgentHarm"
name="ASB"
```

### Issue: Checkpoint not resuming

**Solution:** Make sure you're using the **same log directory**:

```python
# First run
log_dir="./logs/experiment"

# Resume (must use same directory!)
log_dir="./logs/experiment"
```

### Issue: Out of memory

**Solution:** Run smaller batches:

```python
# Instead of indices=range(1000)
for batch_start in range(0, 1000, 100):
    batch_indices = range(batch_start, batch_start + 100)
    dataset = Dataset(..., indices=batch_indices).load()
    dataset.run(...)
```

### Issue: Results not showing in summary

**Solution:** Make sure assessment was requested during run:

```python
results = dataset.run(
    assessment=["aria", "dharma"]  # Must specify!
)
```

---

## See Also

- [Assessment Guide](assessment.md) - ARIA and DHARMA evaluation
- [Attack Detection](attack-detection.md) - Detecting harmful behavior
- [Example: Benchmark Evaluation](../examples/benchmark-evaluation.md)

---

[← Attack Detection](attack-detection.md) | [Assessment →](assessment.md)
