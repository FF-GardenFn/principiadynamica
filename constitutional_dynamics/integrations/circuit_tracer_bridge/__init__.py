"""
Circuit Tracer Bridge - Integration between Constitutional Dynamics and Circuit Tracer

This module provides a bridge between Constitutional Dynamics' alignment monitoring capabilities
and Anthropic's Circuit Tracer mechanistic interpretability tools, creating a feedback loop
that leverages both behavioral monitoring and mechanistic insights to improve alignment interventions.

Components:
- AlignmentThermostat: The main class that orchestrates the feedback loop
- Examples: Practical demonstrations of the bridge in action
"""

from .feedback_loop import AlignmentThermostat

__all__ = ["AlignmentThermostat"]
