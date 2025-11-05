from ..models.agent import AgentConfig, Agent
from ..frameworks.base import Team
from ..models.tool import Tool
from ...logger import logger

from datasets import load_dataset
import asyncio
import importlib
import sys
import io
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import tempfile
import subprocess


class DatasetRegistry:
    """Registry for custom dataset handlers."""
    _handlers = {}

    @classmethod
    def register(cls, dataset_name: str, handler_class):
        """Register a custom dataset handler."""
        cls._handlers[dataset_name] = handler_class

    @classmethod
    def get_handler(cls, dataset_name: str):
        """Get handler for a dataset."""
        return cls._handlers.get(dataset_name)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets."""
        return list(cls._handlers.keys())


class Dataset:
    """
    Represents the collection of data that the multi-agent system will use for evaluation and execution of tasks.

    Features:
    - Parallel execution with configurable workers
    - Checkpoint/resume capability
    - Progress tracking
    - Automatic assessment
    - Extensible for custom datasets
    """

    def __init__(self, name, config=None, split="test", indices=None, architecture=None,
                 framework=None, log_dir=None):
        self.name = name
        self.config = config
        self.split = split
        self.indices = indices
        self.dataset = None
        self.architecture = architecture
        self.framework = framework

        self.team = None
        self.outputs = None
        self.results = []

        if log_dir is None:
            # Create timestamped log directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = f"./logs/{self.name.replace('/', '_')}_{timestamp}"
        else:
            self.log_dir = log_dir

        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def load(self):
        """
        Load the dataset using the handler's load method.

        All dataset handlers must implement a load() method that returns a list
        of task dictionaries with standardized fields (prompt, target_functions).

        The handler is responsible for:
        - Loading data from any source (HuggingFace, local files, API, etc.)
        - Converting to standardized format
        - Returning a list where position = numerical index

        Returns:
            self: For method chaining

        Raises:
            ValueError: If no handler is registered or handler lacks load() method
        """
        handler = DatasetRegistry.get_handler(self.name)

        if not handler:
            raise ValueError(
                f"No handler registered for dataset '{self.name}'. "
                f"Available datasets: {DatasetRegistry.list_datasets()}\n"
                f"To add a new dataset, create a handler class and register it:\n"
                f"  DatasetRegistry.register('{self.name}', YourHandler)"
            )

        if not hasattr(handler, 'load'):
            raise ValueError(
                f"Handler for dataset '{self.name}' does not implement load() method. "
                f"All handlers must implement: load(config, split) -> List[Dict]"
            )

        # Use handler's load() method
        self.dataset = handler.load(config=self.config, split=self.split)

        if not isinstance(self.dataset, list):
            raise ValueError(
                f"Handler for '{self.name}' must return a list from load(), "
                f"got {type(self.dataset)} instead"
            )

        return self

    def __iter__(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        for i, data in enumerate(self.dataset):
            if self.indices is None or i in self.indices:
                yield data

    def _save_checkpoint(self, completed_indices: List[int], results: List[Dict]):
        """Save checkpoint for resume capability."""
        checkpoint_path = Path(self.log_dir) / "checkpoint.json"
        checkpoint = {
            "completed_indices": completed_indices,
            "total_tasks": len(self.dataset),
            "timestamp": datetime.now().isoformat(),
            "framework": self.framework,
            "architecture": self.architecture,
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _load_checkpoint(self) -> Optional[List[int]]:
        """Load checkpoint if exists."""
        checkpoint_path = Path(self.log_dir) / "checkpoint.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
                return checkpoint.get("completed_indices", [])
        return None

    def _save_results(self, results: List[Dict]):
        """Save all results to a JSON file."""
        results_path = Path(self.log_dir) / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Also save as JSONL for easier streaming
        results_jsonl_path = Path(self.log_dir) / "results.jsonl"
        with open(results_jsonl_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    def _load_results(self) -> Optional[List[Dict]]:
        """Load results from disk if they exist."""
        results_path = Path(self.log_dir) / "results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        return None

    def translate_tool_names(self, tool_names):
        """
        Translate dataset-specific tool names to SafeAgents Tool objects.

        Uses DatasetRegistry to delegate to dataset-specific handlers.
        Falls back to direct tool creation if no handler is registered.
        """
        handler = DatasetRegistry.get_handler(self.name)
        if handler:
            return handler.translate_tool_names(tool_names)
        else:
            # Fallback: try to import from tools_agentharm (legacy support)
            logger.warning(f"No handler registered for dataset '{self.name}'. Using legacy import.")
            tools_module = importlib.import_module("tools_agentharm")
            return [
                Tool(tool, getattr(tools_module, tool))
                for tool in tool_names
                if hasattr(tools_module, tool)
            ]
        
    def create_agents(self, task_data, **kwargs):
        """
        Create agents for a specific task.

        Uses DatasetRegistry to delegate to dataset-specific handlers.
        Falls back to default agent creation if no handler is registered.
        """
        handler = DatasetRegistry.get_handler(self.name)
        if handler:
            return handler.create_agents(task_data, self.framework)
        else:
            # Fallback: default agent creation (legacy support)
            logger.warning(f"No handler registered for dataset '{self.name}'. Using default agent creation.")
            return self._default_create_agents(task_data)

    def _default_create_agents(self, task_data):
        """Default agent creation for datasets without custom handlers."""
        agents = []
        special_agents = ["file_surfer", "web_surfer", "coder", "code_executor"]

        for agent_name in special_agents:
            agent_config = AgentConfig(
                name=agent_name,
                special_agent=agent_name
            )
            agents.append((Agent(config=agent_config)))

        # Add a general purpose agent as per the benchmark's requirements
        agents.append(Agent(config=AgentConfig(
            name="tool_user_agent",
            tools=self.translate_tool_names(task_data["target_functions"]),
            system_message="Use tools to solve tasks.",
            description="A general purpose agent."
        )))

        return agents

    async def _run_single_task(self, task_index: int, task_data: Dict, assessment: Optional[List[str]] = None,
                               verbose: bool = False) -> Dict:
        """
        Run a single task through the team.

        Args:
            task_index: Index of the task in dataset
            task_data: Task data dictionary
            assessment: List of assessment methods to run
            verbose: Whether to print logs during execution

        Returns:
            dict: Result containing logs, messages, assessment scores, etc.
        """
        try:
            # Create agents for this task
            agents = self.create_agents(task_data, framework=self.framework)

            # Create team
            team = Team.create(
                agents=agents,
                framework=self.framework,
                architecture=self.architecture,
            )

            # Create attack detector if handler provides one
            attack_detector = None
            handler = DatasetRegistry.get_handler(self.name)
            if handler and hasattr(handler, 'create_attack_detector'):
                try:
                    attack_detector = handler.create_attack_detector(task_data)
                except Exception as e:
                    logger.warning(f"Failed to create attack detector: {e}")

            # Run the task
            result = await team.run(
                task=task_data['prompt'],
                verbose=verbose,
                assessment=assessment,
                attack_detector=attack_detector
            )

            # Add task metadata to result
            result['task_index'] = task_index
            result['task_prompt'] = task_data['prompt']
            result['framework'] = self.framework
            result['architecture'] = self.architecture

            return result

        except Exception as e:
            logger.error(f"Error running task {task_index}: {e}")
            return {
                'task_index': task_index,
                'task_prompt': task_data.get('prompt', ''),
                'error': str(e),
                'logs': '',
                'messages': [],
                'stop_reason': 'error'
            }

    def run(self, parallel: bool = False, workers: int = 1, resume: bool = True,
            progress_bar: bool = True, save_checkpoints: bool = True,
            assessment: Optional[List[str]] = None, verbose: bool = False,
            max_turns: Optional[int] = None) -> List[Dict]:
        """
        Run the dataset through the team of agents with enhanced features.

        Args:
            parallel: Enable parallel execution of tasks
            workers: Number of parallel workers (only if parallel=True)
            resume: Resume from checkpoint if exists
            progress_bar: Show progress bar with tqdm
            save_checkpoints: Save checkpoint after each task
            assessment: List of assessment methods (e.g., ["aria", "dharma"])
            verbose: Print logs during execution
            max_turns: Maximum turns per task (overrides team default)

        Returns:
            List[Dict]: Results for all tasks

        Note:
            For Docker isolation, write your own script and use DockerRunner.
            See documentation for examples.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Load checkpoint if resume is enabled
        completed_indices = []
        if resume:
            loaded_checkpoint = self._load_checkpoint()
            if loaded_checkpoint:
                completed_indices = loaded_checkpoint
                logger.info(f"Resuming from checkpoint: {len(completed_indices)} tasks already completed")

        # Determine which tasks to run
        tasks_to_run = []
        for i, data in enumerate(self.dataset):
            if (self.indices is None or i in self.indices) and i not in completed_indices:
                tasks_to_run.append((i, data))

        if not tasks_to_run:
            logger.info("No tasks to run (all completed or no indices specified)")
            return self.results

        logger.info(f"Running {len(tasks_to_run)} tasks")
        logger.info(f"Framework: {self.framework}, Architecture: {self.architecture}")
        logger.info(f"Logs will be saved to: {self.log_dir}")

        # Setup progress bar
        if progress_bar:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(tasks_to_run), desc="Processing tasks")
            except ImportError:
                logger.warning("tqdm not installed. Install with: pip install tqdm")
                progress_bar = False

        # Run tasks
        results = []
        for task_index, task_data in tasks_to_run:
            if progress_bar:
                pbar.set_description(f"Task {task_index + 1}")

            # Run single task
            result = asyncio.run(self._run_single_task(
                task_index, task_data, assessment=assessment, verbose=verbose
            ))

            results.append(result)

            # Save individual task result
            task_result_path = Path(self.log_dir) / f"task_{task_index}.json"
            with open(task_result_path, 'w') as f:
                json.dump(result, f, indent=2)

            # Save checkpoint
            if save_checkpoints:
                completed_indices.append(task_index)
                self._save_checkpoint(completed_indices, results)

            if progress_bar:
                pbar.update(1)

        if progress_bar:
            pbar.close()

        # Save final results
        self.results = results
        self._save_results(results)

        logger.info(f"✓ Completed {len(results)} tasks")
        logger.info(f"Results saved to: {self.log_dir}/results.json")

        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from the results.

        If results are not in memory, attempts to load them from disk.

        Returns:
            dict: Summary including total tasks, average scores, etc.
        """
        # If results not in memory, try loading from disk
        results = self.results
        if not results:
            loaded_results = self._load_results()
            if loaded_results:
                results = loaded_results
            else:
                return {"error": "No results available. Run the dataset first or check log_dir."}

        summary = {
            "total_tasks": len(results),
            "framework": self.framework,
            "architecture": self.architecture,
            "log_dir": self.log_dir,
        }

        # Count errors
        errors = [r for r in results if 'error' in r]
        summary["errors"] = len(errors)
        summary["successful"] = summary["total_tasks"] - len(errors)

        # Summarize assessment scores if present
        if results and 'assessment' in results[0]:
            assessment_summary = {}
            for metric in results[0]['assessment'].keys():
                scores = [r['assessment'][metric] for r in results if 'assessment' in r]
                assessment_summary[metric] = {
                    "scores": scores,
                    "count": len(scores)
                }
            summary["assessment"] = assessment_summary

        return summary

    def print_summary(self):
        """Print a formatted summary of the results with percentage distributions."""
        from collections import Counter

        summary = self.get_summary()

        print("\n" + "="*80)
        print("DATASET RUN SUMMARY")
        print("="*80)
        print(f"Total tasks: {summary.get('total_tasks', 0)}")
        print(f"Successful: {summary.get('successful', 0)}")
        print(f"Errors: {summary.get('errors', 0)}")
        print(f"Framework: {summary.get('framework', 'N/A')}")
        print(f"Architecture: {summary.get('architecture', 'N/A')}")

        if 'assessment' in summary:
            print("\nAssessment Score Distributions:")

            # Try to use rich for nice tables, fallback to simple text
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()

                for metric, data in summary['assessment'].items():
                    scores = data['scores']
                    total = len(scores)

                    if total == 0:
                        continue

                    # Count occurrences of each score
                    counter = Counter(scores)

                    # Create table for this metric
                    table = Table(title=f"{metric.upper()} Score Distribution")
                    table.add_column("Score", style="cyan", justify="center")
                    table.add_column("Count", style="magenta", justify="right")
                    table.add_column("Percentage", style="green", justify="right")

                    # Sort by score for consistent display
                    for score in sorted(counter.keys()):
                        count = counter[score]
                        percentage = (count / total) * 100
                        table.add_row(
                            str(score),
                            str(count),
                            f"{percentage:.1f}%"
                        )

                    console.print(table)

            except ImportError:
                # Fallback to simple text table if rich is not available
                print("\n  (Install 'rich' for better formatted tables: pip install rich)")

                for metric, data in summary['assessment'].items():
                    scores = data['scores']
                    total = len(scores)

                    if total == 0:
                        continue

                    # Count occurrences of each score
                    counter = Counter(scores)

                    print(f"\n  {metric.upper()} Score Distribution:")
                    print(f"  {'Score':<10} {'Count':<10} {'Percentage':<10}")
                    print(f"  {'-'*10} {'-'*10} {'-'*10}")

                    # Sort by score for consistent display
                    for score in sorted(counter.keys()):
                        count = counter[score]
                        percentage = (count / total) * 100
                        print(f"  {str(score):<10} {count:<10} {percentage:>8.1f}%")

        print(f"\nResults saved to: {summary.get('log_dir', 'N/A')}")
        print("="*80)


# Note: Dataset handlers are auto-imported by safeagents/core/src/__init__.py
# Users don't need to import them manually
