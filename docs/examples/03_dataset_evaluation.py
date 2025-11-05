"""
Dataset Evaluation Example
===========================

This example demonstrates how to evaluate agents on safety benchmarks:
- Loading datasets (AgentHarm, ASB, etc.)
- Running on full datasets or subsets
- Checkpoint and resume capability
- Progress tracking with progress bars
- Assessment with ARIA and DHARMA
- Analyzing and comparing results

Perfect for research, benchmarking, and safety evaluation.
"""

import asyncio
from dotenv import load_dotenv

from safeagents import Dataset, DatasetRegistry

load_dotenv()


# ============================================================================
# Example 1: Basic Dataset Run
# ============================================================================

def example_basic_dataset():
    """
    Run a small subset of AgentHarm with all features enabled.

    This demonstrates the basic workflow for dataset evaluation.
    """
    print("\n" + "="*80)
    print("Example 1: Basic Dataset Run (AgentHarm)")
    print("="*80 + "\n")

    # Load dataset
    dataset = Dataset(
        name="ai-safety-institute/AgentHarm",
        config="harmful",              # Which configuration to use
        split="test_public",           # Dataset split
        indices=[0, 1, 2],             # Run only first 3 tasks
        framework="openai-agents",     # Framework to use
        architecture="centralized",     # Architecture type
    ).load()

    print(f"Loaded {len(dataset.dataset)} tasks from AgentHarm\n")

    # Run with all features
    results = dataset.run(
        parallel=False,                 # Sequential execution
        resume=True,                    # Resume from checkpoint if exists
        progress_bar=True,              # Show progress bar
        save_checkpoints=True,          # Save after each task
        assessment=["aria", "dharma"],  # Run both assessments
        verbose=False                   # Don't print logs during execution
    )

    # Print summary with distributions
    dataset.print_summary()

    return results


# ============================================================================
# Example 2: Resume from Checkpoint
# ============================================================================

def example_resume_checkpoint():
    """
    Demonstrate checkpoint/resume capability.

    Run this example, interrupt it (Ctrl+C), then run again to see
    it resume from where it left off.
    """
    print("\n" + "="*80)
    print("Example 2: Checkpoint and Resume")
    print("="*80 + "\n")

    dataset = Dataset(
        name="ai-safety-institute/AgentHarm",
        config="harmful",
        split="test_public",
        indices=range(10),              # Run 10 tasks
        framework="openai-agents",
        architecture="centralized",
        log_dir="./logs/checkpoint_example"  # Fixed directory for resume
    ).load()

    print("Running dataset (interrupt with Ctrl+C to test resume)...\n")

    try:
        results = dataset.run(
            resume=True,                # Enable resume
            progress_bar=True,
            save_checkpoints=True,      # Save after each task
            assessment=["aria"]
        )

        dataset.print_summary()

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Progress saved.")
        print("Run again to resume from checkpoint.\n")


# ============================================================================
# Example 3: Framework Comparison
# ============================================================================

def example_framework_comparison():
    """
    Compare multiple frameworks on the same tasks.

    This is useful for benchmarking and understanding framework differences
    in safety behavior.
    """
    print("\n" + "="*80)
    print("Example 3: Framework Comparison")
    print("="*80 + "\n")

    # Test on small subset
    test_indices = [0, 1, 2]
    frameworks = ["openai-agents", "autogen", "langgraph"]

    all_results = {}

    for framework in frameworks:
        print(f"\n--- Running {framework} ---")

        dataset = Dataset(
            name="ai-safety-institute/AgentHarm",
            config="harmful",
            split="test_public",
            indices=test_indices,
            framework=framework,
            architecture="centralized",
            log_dir=f"./logs/comparison_{framework}"
        ).load()

        results = dataset.run(
            progress_bar=True,
            assessment=["aria", "dharma"],
            verbose=False
        )

        # Store summary
        all_results[framework] = dataset.get_summary()

    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    for framework, summary in all_results.items():
        print(f"\n{framework}:")
        print(f"  Total Tasks: {summary.get('total_tasks', 0)}")
        print(f"  Successful: {summary.get('successful', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")

        if 'assessment' in summary:
            # ARIA distribution
            if 'aria' in summary['assessment']:
                aria_scores = summary['assessment']['aria']['scores']
                critical = aria_scores.count('4')
                safe = aria_scores.count('1')
                print(f"  ARIA: {safe} safe, {critical} critical")

            # DHARMA distribution
            if 'dharma' in summary['assessment']:
                dharma_scores = summary['assessment']['dharma']['scores']
                from collections import Counter
                dharma_dist = Counter(dharma_scores)
                print(f"  DHARMA: {dict(dharma_dist)}")


# ============================================================================
# Example 4: Custom Subset Selection
# ============================================================================

def example_custom_subset():
    """
    Run on a custom subset of tasks.

    Useful when you want to focus on specific types of tasks or
    test on a representative sample.
    """
    print("\n" + "="*80)
    print("Example 4: Custom Subset Selection")
    print("="*80 + "\n")

    # Load dataset with custom indices
    # You can select specific task IDs based on your needs
    custom_indices = [0, 5, 10, 15, 20]  # Select specific tasks

    dataset = Dataset(
        name="ai-safety-institute/AgentHarm",
        config="harmful",
        split="test_public",
        indices=custom_indices,
        framework="openai-agents",
        architecture="centralized",
    ).load()

    print(f"Running on custom subset: tasks {custom_indices}\n")

    results = dataset.run(
        progress_bar=True,
        assessment=["aria", "dharma"],
        verbose=False
    )

    dataset.print_summary()


# ============================================================================
# Example 5: Analyzing Results Programmatically
# ============================================================================

def example_analyze_results():
    """
    Programmatically analyze dataset results.

    Shows how to extract and analyze assessment data for further processing.
    """
    print("\n" + "="*80)
    print("Example 5: Programmatic Result Analysis")
    print("="*80 + "\n")

    dataset = Dataset(
        name="ai-safety-institute/AgentHarm",
        config="harmful",
        split="test_public",
        indices=range(5),
        framework="openai-agents",
        architecture="centralized",
    ).load()

    results = dataset.run(
        progress_bar=True,
        assessment=["aria", "dharma"],
        verbose=False
    )

    # Get summary
    summary = dataset.get_summary()

    # Analyze ARIA scores
    if 'assessment' in summary and 'aria' in summary['assessment']:
        aria_scores = summary['assessment']['aria']['scores']

        print("\n--- ARIA Analysis ---")
        print(f"Total tasks: {len(aria_scores)}")

        for level in ['1', '2', '3', '4']:
            count = aria_scores.count(level)
            percentage = (count / len(aria_scores)) * 100
            print(f"  Level {level}: {count} ({percentage:.1f}%)")

        # Identify high-risk tasks
        high_risk = [i for i, score in enumerate(aria_scores) if score == '4']
        if high_risk:
            print(f"\n⚠️  High-risk tasks (ARIA-4): {high_risk}")

    # Analyze DHARMA classifications
    if 'assessment' in summary and 'dharma' in summary['assessment']:
        dharma_scores = summary['assessment']['dharma']['scores']

        print("\n--- DHARMA Analysis ---")
        from collections import Counter
        dharma_dist = Counter(dharma_scores)

        for classification, count in dharma_dist.most_common():
            percentage = (count / len(dharma_scores)) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")

        # Count safety interventions
        interventions = [s for s in dharma_scores if 'stop' in s]
        print(f"\n✓ Safety interventions: {len(interventions)}/{len(dharma_scores)}")

    # Access individual task results
    print("\n--- Individual Task Details ---")
    for i, result in enumerate(results[:3]):  # Show first 3 tasks
        print(f"\nTask {i}:")
        print(f"  Stop reason: {result.get('stop_reason', 'N/A')}")
        print(f"  ARIA: {result.get('assessment', {}).get('aria', 'N/A')}")
        print(f"  DHARMA: {result.get('assessment', {}).get('dharma', 'N/A')}")
        print(f"  Tools called: {len(result.get('execution_trace', {}).get('tool_calls', []))}")


# ============================================================================
# Example 6: Working with ASB Dataset
# ============================================================================

def example_asb_dataset():
    """
    Run the ASB (Agent Safety Benchmark) dataset.

    ASB is structured by agent types with attack and normal tools.
    """
    print("\n" + "="*80)
    print("Example 6: ASB Dataset Evaluation")
    print("="*80 + "\n")

    # Run ASB for a specific agent
    dataset = Dataset(
        name="asb",
        config="financial_analyst_agent",  # Specific agent
        split="test",
        indices=[0, 1, 2],
        framework="openai-agents",
        architecture="centralized",
    ).load()

    print(f"Running ASB for financial_analyst_agent\n")

    results = dataset.run(
        progress_bar=True,
        assessment=["aria", "dharma"],
        verbose=False
    )

    dataset.print_summary()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main entry point with example selection."""
    import sys

    examples = {
        "1": ("Basic dataset run", example_basic_dataset),
        "2": ("Checkpoint and resume", example_resume_checkpoint),
        "3": ("Framework comparison", example_framework_comparison),
        "4": ("Custom subset", example_custom_subset),
        "5": ("Analyze results", example_analyze_results),
        "6": ("ASB dataset", example_asb_dataset),
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        name, func = examples[sys.argv[1]]
        print(f"\nRunning: {name}")
        func()
    else:
        print("\nAvailable Dataset Evaluation Examples:")
        print("="*80)
        for key, (name, _) in examples.items():
            print(f"  {key}: {name}")
        print("\nUsage: python 03_dataset_evaluation.py <example_number>")
        print("\nExamples:")
        print("  python 03_dataset_evaluation.py 1    # Run basic example")
        print("  python 03_dataset_evaluation.py 3    # Compare frameworks")


if __name__ == "__main__":
    main()
