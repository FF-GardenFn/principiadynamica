"""
Visualizer - Rich/ASCII visualization components

This module provides visualization tools for alignment trajectories,
metrics, and analysis results.
"""

import logging
from typing import Dict, List, Any, Optional, Union

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rich_print

    USE_RICH = True
    console = Console()
except ImportError:
    USE_RICH = False
    console = None
    logging.warning("Rich library not available. Using plain text visualization.")

logger = logging.getLogger("constitutional_dynamics.vis.visualizer")


class TrajectoryVisualizer:
    """
    Visualizes alignment trajectories and metrics in the console.

    This class provides methods for creating sparklines, tables, and
    other visualizations of alignment data.
    """

    def __init__(self, use_rich: bool = None, width: int = 60):
        """
        Initialize the trajectory visualizer.

        Args:
            use_rich: Whether to use rich formatting (None = auto-detect)
            width: Default width for visualizations
        """
        self.use_rich = use_rich if use_rich is not None else USE_RICH
        self.width = width

        if self.use_rich and not USE_RICH:
            logger.warning("Rich library requested but not available. Using plain text.")
            self.use_rich = False

    def visualize_alignment_history(self,
                                   alignment_scores: List[float],
                                   width: Optional[int] = None,
                                   threshold: float = 0.7) -> None:
        """
        Visualize alignment history as a sparkline.

        Args:
            alignment_scores: List of alignment scores
            width: Width of the visualization (defaults to self.width)
            threshold: Alignment threshold to highlight
        """
        if not alignment_scores:
            logger.warning("No alignment data to visualize")
            return

        width = width or self.width

        # Scale to fit width
        indices = [int(i * (width - 1) / max(1, len(alignment_scores) - 1)) for i in range(len(alignment_scores))]

        # Create a sparse array for the sparkline
        sparkline = [None] * width
        for i, score in zip(indices, alignment_scores):
            sparkline[i] = score

        # Fill in gaps
        for i in range(width):
            if sparkline[i] is None:
                # Find nearest non-None values
                left = right = i
                while left >= 0 and sparkline[left] is None:
                    left -= 1
                while right < width and sparkline[right] is None:
                    right += 1

                if left >= 0 and right < width:
                    # Interpolate
                    left_val = sparkline[left]
                    right_val = sparkline[right]
                    weight = (i - left) / (right - left)
                    sparkline[i] = left_val + weight * (right_val - left_val)
                elif left >= 0:
                    sparkline[i] = sparkline[left]
                elif right < width:
                    sparkline[i] = sparkline[right]
                else:
                    sparkline[i] = 0.5  # Fallback

        # Create visualization
        if self.use_rich:
            console.print("\n[bold]Alignment Trajectory:[/bold]")

            # Create colored sparkline
            chars = []
            for score in sparkline:
                if score >= threshold:
                    chars.append(
                        f"[green]{'█' if score > 0.9 else '▇' if score > 0.8 else '▆' if score > 0.7 else '▅'}[/]")
                else:
                    chars.append(
                        f"[red]{'▄' if score > 0.6 else '▃' if score > 0.5 else '▂' if score > 0.4 else '▁'}[/]")

            console.print("".join(chars))
            console.print(f"Start: [cyan]{alignment_scores[0]:.2f}[/] → End: [cyan]{alignment_scores[-1]:.2f}[/]")

        else:
            print("\nAlignment Trajectory:")

            # Create ASCII sparkline
            chars = []
            for score in sparkline:
                if score >= threshold:
                    chars.append('#')
                else:
                    chars.append('.')

            print("".join(chars))
            print(f"Start: {alignment_scores[0]:.2f} → End: {alignment_scores[-1]:.2f}")

    def print_trajectory_analysis(self,
                                 analysis: Dict[str, Any],
                                 predictions: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Print trajectory analysis results.

        Args:
            analysis: Stability metrics dictionary
            predictions: Optional trajectory predictions
        """
        if "error" in analysis:
            logger.error(f"Error in trajectory analysis: {analysis['error']}")
            if self.use_rich:
                console.print(f"[bold red]Error:[/] {analysis['error']}")
            else:
                print(f"Error: {analysis['error']}")
            return

        if self.use_rich:
            console.print("\n[bold cyan]Trajectory Analysis[/bold cyan]")

            # Create metrics table
            table = Table(title="Alignment Stability Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            # Add rows for key metrics
            table.add_row("Average Alignment", f"{analysis.get('avg_alignment', 0):.3f}")
            table.add_row("Alignment Trend", f"{analysis.get('alignment_trend', 0):.3f}")
            table.add_row("Alignment Volatility", f"{analysis.get('alignment_volatility', 0):.3f}")
            table.add_row("Stability Score", f"{analysis.get('stability_score', 0):.3f}")

            console.print(table)

            # Print summary judgment
            stability_score = analysis.get('stability_score', 0)
            if stability_score > 0.8:
                console.print("[bold green]✓ The system exhibits strong alignment stability[/]")
            elif stability_score > 0.5:
                console.print("[bold yellow]⚠ The system has moderate alignment stability[/]")
            else:
                console.print("[bold red]✗ The system shows poor alignment stability[/]")

            # Print predictions if available
            if predictions:
                console.print("\n[bold cyan]Trajectory Predictions[/bold cyan]")

                pred_table = Table(title="Predicted Future States")
                pred_table.add_column("Step", style="dim")
                pred_table.add_column("Alignment", style="green")
                pred_table.add_column("Trend", style="cyan")

                prev_alignment = None
                for pred in predictions:
                    alignment = pred.get("predicted_alignment", 0)

                    if prev_alignment is not None:
                        trend = alignment - prev_alignment
                        trend_str = f"{'↑' if trend > 0 else '↓' if trend < 0 else '→'} {abs(trend):.3f}"
                    else:
                        trend_str = "—"

                    pred_table.add_row(
                        str(pred.get("step", 0)),
                        f"{alignment:.3f}",
                        trend_str
                    )

                    prev_alignment = alignment

                console.print(pred_table)

                # Final prediction
                final_alignment = predictions[-1].get("predicted_alignment", 0)
                if final_alignment >= 0.7:
                    console.print("[bold green]✓ The system is predicted to remain aligned[/]")
                else:
                    console.print("[bold red]✗ The system is predicted to become misaligned[/]")

        else:
            print("\nTrajectory Analysis")
            print("-------------------")
            print(f"Average Alignment: {analysis.get('avg_alignment', 0):.3f}")
            print(f"Alignment Trend: {analysis.get('alignment_trend', 0):.3f}")
            print(f"Alignment Volatility: {analysis.get('alignment_volatility', 0):.3f}")
            print(f"Stability Score: {analysis.get('stability_score', 0):.3f}")

            # Print summary judgment
            stability_score = analysis.get('stability_score', 0)
            if stability_score > 0.8:
                print("✓ The system exhibits strong alignment stability")
            elif stability_score > 0.5:
                print("⚠ The system has moderate alignment stability")
            else:
                print("✗ The system shows poor alignment stability")

            # Print predictions if available
            if predictions:
                print("\nTrajectory Predictions")
                print("---------------------")

                prev_alignment = None
                for pred in predictions:
                    alignment = pred.get("predicted_alignment", 0)

                    if prev_alignment is not None:
                        trend = alignment - prev_alignment
                        trend_str = f"{'↑' if trend > 0 else '↓' if trend < 0 else '→'} {abs(trend):.3f}"
                    else:
                        trend_str = "—"

                    print(f"Step {pred.get('step', 0)}: Alignment = {alignment:.3f} ({trend_str})")

                    prev_alignment = alignment

                # Final prediction
                final_alignment = predictions[-1].get("predicted_alignment", 0)
                if final_alignment >= 0.7:
                    print("✓ The system is predicted to remain aligned")
                else:
                    print("✗ The system is predicted to become misaligned")

    def print_robustness_analysis(self, robustness: Dict[str, Any]) -> None:
        """
        Print robustness analysis results.

        Args:
            robustness: Robustness metrics dictionary
        """
        if "error" in robustness:
            logger.error(f"Error in robustness analysis: {robustness['error']}")
            if self.use_rich:
                console.print(f"[bold red]Error:[/] {robustness['error']}")
            else:
                print(f"Error: {robustness['error']}")
            return

        if self.use_rich:
            console.print("\n[bold cyan]Robustness Analysis[/bold cyan]")

            # Create metrics table
            table = Table(title="Alignment Robustness Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            # Add rows for key metrics
            table.add_row("Base Alignment", f"{robustness.get('base_alignment', 0):.3f}")
            table.add_row("Average Change", f"{robustness.get('avg_change', 0):.3f}")
            table.add_row("Max Negative Change", f"{robustness.get('max_negative_change', 0):.3f}")
            table.add_row("Max Positive Change", f"{robustness.get('max_positive_change', 0):.3f}")
            table.add_row("Robustness Score", f"{robustness.get('robustness_score', 0):.3f}")
            table.add_row("PSD Distance", f"{robustness.get('psd_distance', 0):.3f}")

            console.print(table)

            # Print summary judgment
            robustness_score = robustness.get('robustness_score', 0)
            if robustness_score > 0.8:
                console.print("[bold green]✓ The system exhibits strong alignment robustness[/]")
            elif robustness_score > 0.5:
                console.print("[bold yellow]⚠ The system has moderate alignment robustness[/]")
            else:
                console.print("[bold red]✗ The system shows poor alignment robustness[/]")

        else:
            print("\nRobustness Analysis")
            print("-------------------")
            print(f"Base Alignment: {robustness.get('base_alignment', 0):.3f}")
            print(f"Average Change: {robustness.get('avg_change', 0):.3f}")
            print(f"Max Negative Change: {robustness.get('max_negative_change', 0):.3f}")
            print(f"Max Positive Change: {robustness.get('max_positive_change', 0):.3f}")
            print(f"Robustness Score: {robustness.get('robustness_score', 0):.3f}")
            print(f"PSD Distance: {robustness.get('psd_distance', 0):.3f}")

            # Print summary judgment
            robustness_score = robustness.get('robustness_score', 0)
            if robustness_score > 0.8:
                print("✓ The system exhibits strong alignment robustness")
            elif robustness_score > 0.5:
                print("⚠ The system has moderate alignment robustness")
            else:
                print("✗ The system shows poor alignment robustness")

    def print_optimization_results(self, results: Dict[str, Any]) -> None:
        """
        Print optimization results.

        Args:
            results: Optimization results dictionary
        """
        if not results or "path" not in results:
            logger.error("No valid optimization results to display")
            return

        if self.use_rich:
            console.print("\n[bold cyan]Optimization Results[/bold cyan]")

            # Create path table
            table = Table(title=f"Optimized Path (Energy: {results.get('energy', 0):.4f})")
            table.add_column("Step", style="dim")
            table.add_column("State", style="cyan")
            table.add_column("Metadata", style="green")

            # Add rows for path
            path_info = results.get("path_info", [])
            for i, state in enumerate(path_info):
                metadata_str = ", ".join(f"{k}: {v}" for k, v in state.get("metadata", {}).items())
                table.add_row(
                    str(i + 1),
                    state.get("state_id", "unknown"),
                    metadata_str
                )

            console.print(table)
            console.print(f"Solver: [cyan]{results.get('solver', 'unknown')}[/]")

        else:
            print("\nOptimization Results")
            print("-------------------")
            print(f"Energy: {results.get('energy', 0):.4f}")
            print(f"Solver: {results.get('solver', 'unknown')}")
            print("\nOptimized Path:")

            path_info = results.get("path_info", [])
            for i, state in enumerate(path_info):
                metadata_str = ", ".join(f"{k}: {v}" for k, v in state.get("metadata", {}).items())
                print(f"{i+1}. {state.get('state_id', 'unknown')} ({metadata_str})")


def create_visualizer(use_rich: bool = None, width: int = 60) -> TrajectoryVisualizer:
    """
    Create a new TrajectoryVisualizer instance.

    Args:
        use_rich: Whether to use rich formatting (None = auto-detect)
        width: Default width for visualizations

    Returns:
        TrajectoryVisualizer instance
    """
    return TrajectoryVisualizer(use_rich=use_rich, width=width)
