"""
Constitutional Dynamics Integrations Module

This module provides integrations with external systems for the Constitutional Dynamics package,
including Neo4j graph database, D-Wave quantum annealer, LLM strategist, and Circuit Tracer.
"""

from .graph import create_graph_manager, GraphManager
from .quantum import create_annealer, QuantumAnnealer
from .strategist import create_strategist, MetaStrategist
from .circuit_tracer_bridge import AlignmentThermostat

__all__ = [
    "create_graph_manager",
    "GraphManager",
    "create_annealer",
    "QuantumAnnealer",
    "create_strategist",
    "MetaStrategist",
    "AlignmentThermostat"
]
