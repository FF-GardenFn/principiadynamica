"""
Graph Integration - Neo4j consequence-graph connector

This module provides a graph manager for storing and querying alignment states
and transitions in a Neo4j graph database.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    from neo4j import GraphDatabase, Driver, Session
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Graph integration will be disabled.")

logger = logging.getLogger("constitutional_dynamics.integrations.graph")


class GraphManager:
    """
    Manager for storing and querying alignment states and transitions
    in a Neo4j graph database.
    """

    def __init__(self, uri: str, auth: Optional[Tuple[str, str]] = None, database: str = "neo4j"):
        """
        Initialize the graph manager.

        Args:
            uri: Neo4j connection URI (e.g., bolt://localhost:7687)
            auth: Optional tuple of (username, password)
            database: Neo4j database name
        """
        self.uri = uri
        self.auth = auth
        self.database = database
        self.driver = None

        if not NEO4J_AVAILABLE:
            logger.error("Neo4j driver not available. Graph integration is disabled.")
            return

        try:
            if auth:
                self.driver = GraphDatabase.driver(uri, auth=auth)
            else:
                self.driver = GraphDatabase.driver(uri)

            # Test connection
            with self.driver.session(database=database) as session:
                result = session.run("RETURN 1 AS test")
                result.single()

            logger.info(f"Connected to Neo4j at {uri}")

            # Create constraints and indexes
            self._initialize_schema()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def _initialize_schema(self):
        """Initialize the graph schema with constraints and indexes"""
        if not self.driver:
            return

        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints
                session.run("""
                    CREATE CONSTRAINT state_id IF NOT EXISTS
                    FOR (s:State) REQUIRE s.id IS UNIQUE
                """)

                # Create indexes
                session.run("""
                    CREATE INDEX state_timestamp IF NOT EXISTS
                    FOR (s:State) ON (s.timestamp)
                """)

                session.run("""
                    CREATE INDEX state_alignment IF NOT EXISTS
                    FOR (s:State) ON (s.alignment_score)
                """)

                logger.info("Neo4j schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")

    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None

    def add_state(self, state_id: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a state node to the graph.

        Args:
            state_id: Unique identifier for the state
            embedding: Vector embedding of the state
            metadata: Optional metadata for the state

        Returns:
            Success status
        """
        if not self.driver:
            return False

        metadata = metadata or {}
        timestamp = metadata.get("timestamp", time.time())
        alignment_score = metadata.get("alignment_score", 0.5)

        # Prepare properties
        properties = {
            "id": state_id,
            "timestamp": timestamp,
            "alignment_score": alignment_score,
            "embedding_dimension": len(embedding)
        }

        # Add metadata properties
        for key, value in metadata.items():
            if key not in properties and isinstance(value, (str, int, float, bool)):
                properties[key] = value

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MERGE (s:State {id: $id})
                    SET s += $properties
                    RETURN s.id
                """, id=state_id, properties=properties)

                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to add state to graph: {e}")
            return False

    def add_transition(self, from_state_id: str, to_state_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a transition edge between two states.

        Args:
            from_state_id: ID of the source state
            to_state_id: ID of the target state
            metadata: Optional metadata for the transition

        Returns:
            Success status
        """
        if not self.driver:
            return False

        metadata = metadata or {}
        timestamp = metadata.get("timestamp", time.time())
        alignment_change = metadata.get("alignment_change", 0.0)

        # Prepare properties
        properties = {
            "timestamp": timestamp,
            "alignment_change": alignment_change
        }

        # Add metadata properties
        for key, value in metadata.items():
            if key not in properties and isinstance(value, (str, int, float, bool)):
                properties[key] = value

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (from:State {id: $from_id})
                    MATCH (to:State {id: $to_id})
                    MERGE (from)-[t:TRANSITION]->(to)
                    SET t += $properties
                    RETURN t
                """, from_id=from_state_id, to_id=to_state_id, properties=properties)

                return result.single() is not None
        except Exception as e:
            logger.error(f"Failed to add transition to graph: {e}")
            return False

    def get_state(self, state_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a state from the graph.

        Args:
            state_id: ID of the state to retrieve

        Returns:
            State data or None if not found
        """
        if not self.driver:
            return None

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (s:State {id: $id})
                    RETURN s
                """, id=state_id)

                record = result.single()
                if record:
                    return dict(record["s"])
                return None
        except Exception as e:
            logger.error(f"Failed to get state from graph: {e}")
            return None

    def get_transitions(self, state_id: str, direction: str = "outgoing") -> List[Dict[str, Any]]:
        """
        Get transitions for a state.

        Args:
            state_id: ID of the state
            direction: 'outgoing', 'incoming', or 'both'

        Returns:
            List of transitions
        """
        if not self.driver:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                if direction == "outgoing":
                    query = """
                        MATCH (s:State {id: $id})-[t:TRANSITION]->(target:State)
                        RETURN t, target.id AS target_id
                    """
                elif direction == "incoming":
                    query = """
                        MATCH (source:State)-[t:TRANSITION]->(s:State {id: $id})
                        RETURN t, source.id AS source_id
                    """
                else:  # both
                    query = """
                        MATCH (s:State {id: $id})-[t:TRANSITION]->(target:State)
                        RETURN t, target.id AS target_id, null AS source_id
                        UNION
                        MATCH (source:State)-[t:TRANSITION]->(s:State {id: $id})
                        RETURN t, null AS target_id, source.id AS source_id
                    """

                result = session.run(query, id=state_id)

                transitions = []
                for record in result:
                    transition = dict(record["t"])
                    if "target_id" in record and record["target_id"]:
                        transition["target_id"] = record["target_id"]
                    if "source_id" in record and record["source_id"]:
                        transition["source_id"] = record["source_id"]
                    transitions.append(transition)

                return transitions
        except Exception as e:
            logger.error(f"Failed to get transitions from graph: {e}")
            return []

    def get_alignment_impact(self, state_id: str) -> float:
        """
        Calculate the alignment impact of a state based on its transitions.

        Args:
            state_id: ID of the state

        Returns:
            Alignment impact score
        """
        if not self.driver:
            return 0.0

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (s:State {id: $id})-[t:TRANSITION]->(target:State)
                    RETURN avg(t.alignment_change) AS avg_change,
                           count(t) AS num_transitions
                """, id=state_id)

                record = result.single()
                if record and record["num_transitions"] > 0:
                    return record["avg_change"]
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get alignment impact: {e}")
            return 0.0

    def get_transition_strength(self, from_state_id: str, to_state_id: str) -> float:
        """
        Get the strength of a transition between two states.

        Args:
            from_state_id: ID of the source state
            to_state_id: ID of the target state

        Returns:
            Transition strength
        """
        if not self.driver:
            return 0.0

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (from:State {id: $from_id})-[t:TRANSITION]->(to:State {id: $to_id})
                    RETURN t.alignment_change
                """, from_id=from_state_id, to_id=to_state_id)

                record = result.single()
                if record and record["t.alignment_change"] is not None:
                    # Positive alignment change is good
                    return record["t.alignment_change"]
                return 0.0
        except Exception as e:
            logger.error(f"Failed to get transition strength: {e}")
            return 0.0

    def get_aligned_states(self, threshold: float = 0.7, limit: int = 100) -> List[str]:
        """
        Get states with alignment score above threshold.

        Args:
            threshold: Alignment score threshold
            limit: Maximum number of states to return

        Returns:
            List of state IDs
        """
        if not self.driver:
            return []

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    MATCH (s:State)
                    WHERE s.alignment_score >= $threshold
                    RETURN s.id AS id
                    ORDER BY s.alignment_score DESC
                    LIMIT $limit
                """, threshold=threshold, limit=limit)

                return [record["id"] for record in result]
        except Exception as e:
            logger.error(f"Failed to get aligned states: {e}")
            return []


def create_graph_manager(uri: str, auth: Optional[Tuple[str, str]] = None, database: str = "neo4j") -> Optional[GraphManager]:
    """
    Create a new GraphManager instance.

    Args:
        uri: Neo4j connection URI
        auth: Optional tuple of (username, password)
        database: Neo4j database name

    Returns:
        GraphManager instance or None if Neo4j is not available
    """
    if not NEO4J_AVAILABLE:
        logger.error("Neo4j driver not available. Cannot create GraphManager.")
        return None

    return GraphManager(uri, auth, database)
