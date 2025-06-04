"""
Circuit Tracer Bridge - Interventions Module

This module provides intervention mechanisms that leverage Circuit Tracer's mechanistic insights
to apply targeted modifications to language model behavior. These interventions are triggered
by the feedback loop when alignment issues are detected.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
# Import strategist for future integration
from constitutional_dynamics.integrations.strategist import create_strategist, MetaStrategist

logger = logging.getLogger("constitutional_dynamics.integrations.circuit_tracer_bridge.interventions")


class CircuitTracerIntervention:
    """
    Base class for interventions that leverage Circuit Tracer's mechanistic insights.

    This class provides the foundation for implementing various intervention strategies
    that can be applied when alignment issues are detected. Interventions use the
    attribution graph from Circuit Tracer to identify and modify specific components
    of the model's internal circuitry.
    """

    def __init__(
            self,
            model_interface: Any,
            circuit_tracer_instance: Any,
            intervention_strength: float = 0.5,
            max_features_to_modify: int = 5,
            stability_aware: bool = True,
            enable_strategist_preparation: bool = True
    ):
        """
        Initialize the intervention with necessary components.

        Args:
            model_interface: Interface to the model for applying interventions
            circuit_tracer_instance: Instance of Circuit Tracer for mechanistic analysis
            intervention_strength: Strength of the intervention (0.0 to 1.0)
            max_features_to_modify: Maximum number of features to modify in a single intervention
            stability_aware: Whether to take stability metrics into account when applying interventions
            enable_strategist_preparation: Whether to prepare context for MetaStrategist integration

        This intervention is prepared for integration with MetaStrategist, which can:
        - Generate strategic recommendations based on intervention context
        - Refine intervention approaches based on feedback
        - Enable human-in-the-loop guidance for complex alignment issues

        To enable strategist integration, uncomment the strategist code block
        and ensure the constitutional_dynamics.integrations.strategist module is available.
        """
        self.model = model_interface
        self.tracer = circuit_tracer_instance
        self.strength = intervention_strength
        self.max_features = max_features_to_modify
        self.stability_aware = stability_aware
        self.enable_strategist_preparation = enable_strategist_preparation
        self.intervention_history = []

        logger.info(
            "CircuitTracerIntervention initialized with strength=%.2f, max_features=%d, stability_aware=%s, enable_strategist=%s",
            intervention_strength, max_features_to_modify, stability_aware, enable_strategist_preparation
        )

    def apply(
            self,
            circuit_analysis_result: Dict[str, Any],
            issue_type: Optional[str] = None,
            stability_metrics: Optional[Dict[str, Any]] = None,
            modulated_probabilities: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Apply an intervention based on circuit analysis results.

        This is the main method that should be implemented by subclasses to apply
        specific types of interventions. When stability_aware is True, the method
        should take stability_metrics and modulated_probabilities into account.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected (optional)
            stability_metrics: Stability metrics from the AlignmentThermostat (optional)
            modulated_probabilities: Activation probabilities modulated by stability (optional)

        Returns:
            Dictionary with intervention results
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def _prepare_strategist_context(
            self, 
            intervention_type: str, 
            targets: Any, 
            circuit_analysis_result: Dict[str, Any], 
            issue_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Prepare context for MetaStrategist integration.

        Args:
            intervention_type: Type of intervention applied
            targets: Targets of the intervention
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected (optional)

        Returns:
            Dictionary with context for MetaStrategist or None if strategist preparation is disabled
        """
        if not self.enable_strategist_preparation:
            return None

        return {
            "intervention_type": intervention_type,
            "targets": targets,
            "issue_type": issue_type,
            "circuit_analysis_summary": circuit_analysis_result.get("summary", "No summary available"),
            "intervention_strength": self.strength
        }

    def _record_intervention(
            self,
            intervention_type: str,
            targets: Any,
            metadata: Optional[Dict[str, Any]] = None,
            stability_metrics: Optional[Dict[str, Any]] = None,
            modulated_probabilities: Optional[List[float]] = None
    ) -> None:
        """
        Record an intervention for later analysis.

        Args:
            intervention_type: Type of intervention applied
            targets: Targets of the intervention
            metadata: Additional metadata about the intervention
            stability_metrics: Stability metrics that influenced the intervention
            modulated_probabilities: Activation probabilities modulated by stability
        """
        import time

        record = {
            "type": intervention_type,
            "targets": targets,
            "strength": self.strength,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        # Add stability information if available
        if self.stability_aware and stability_metrics:
            record["stability_metrics"] = stability_metrics

        if self.stability_aware and modulated_probabilities:
            record["modulated_probabilities"] = modulated_probabilities

            # Calculate average modulation effect
            if modulated_probabilities:
                avg_modulation = sum(modulated_probabilities) / len(modulated_probabilities)
                record["average_modulation"] = avg_modulation

                logger.debug(
                    "Stability-modulated intervention: type=%s, avg_modulation=%.3f",
                    intervention_type, avg_modulation
                )

        self.intervention_history.append(record)
        logger.debug("Recorded intervention: %s", record)

    def get_intervention_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of interventions applied.

        Returns:
            List of intervention records
        """
        return self.intervention_history


class FeatureSuppressionIntervention(CircuitTracerIntervention):
    """
    Intervention that suppresses specific features identified in the circuit analysis.

    This intervention targets specific features in the model's internal circuitry
    that are contributing to alignment issues, and suppresses their activation.
    """

    def apply(
            self,
            circuit_analysis_result: Dict[str, Any],
            issue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply feature suppression based on circuit analysis results.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected (optional)

        Returns:
            Dictionary with intervention results
        """
        if not circuit_analysis_result or "critical_features" not in circuit_analysis_result:
            logger.warning("Cannot apply feature suppression: no critical features in analysis result")
            return {"success": False, "reason": "no_critical_features"}

        # Extract critical features to suppress
        features = circuit_analysis_result["critical_features"]
        if not features:
            logger.warning("Empty critical features list")
            return {"success": False, "reason": "empty_features_list"}

        # Limit the number of features to modify
        features_to_suppress = features[:min(len(features), self.max_features)]
        logger.info("Suppressing %d features: %s", len(features_to_suppress), features_to_suppress)

        # In a real implementation, this would call the model interface to suppress features
        # For this POC, we simulate the action

        # Prepare data for strategist integration
        strategist_context = self._prepare_strategist_context(
            intervention_type="feature_suppression",
            targets=features_to_suppress,
            circuit_analysis_result=circuit_analysis_result,
            issue_type=issue_type
        )

        # Skip strategist integration if disabled
        if strategist_context is None:
            logger.debug("Strategist preparation is disabled, skipping context creation")
            strategist_context = {}

        # This data can be used to feed into the strategist in a future implementation
        # Example usage (commented out for now):
        # strategist = create_strategist()
        # strategy_result = strategist.generate_strategy(
        #     context=strategist_context,
        #     metrics={"alignment_score": 0.7},  # Would come from actual monitoring
        #     constraints={"max_steps": 5}
        # )
        # The strategy_result could then be used to guide further interventions
        # human in the loop type of feedback to then regulate the intervention

        # Record the intervention
        self._record_intervention(
            intervention_type="feature_suppression",
            targets=features_to_suppress,
            metadata={
                "issue_type": issue_type,
                "strategist_context": strategist_context,
                "for_strategist": True  # Flag to indicate this data is prepared for strategist
            }
        )

        return {
            "success": True,
            "intervention_type": "feature_suppression",
            "features_suppressed": features_to_suppress,
            "strength": self.strength,
            "strategist_context": strategist_context  # Include strategist context in the return value
        }


class FeatureAmplificationIntervention(CircuitTracerIntervention):
    """
    Intervention that amplifies specific features to counteract alignment issues.

    This intervention identifies features associated with aligned behavior and
    amplifies their activation to counteract unaligned features.
    """

    def __init__(
            self,
            model_interface: Any,
            circuit_tracer_instance: Any,
            aligned_feature_map: Dict[str, List[str]],
            **kwargs
    ):
        """
        Initialize with a mapping of alignment issues to aligned features.

        Args:
            model_interface: Interface to the model for applying interventions
            circuit_tracer_instance: Instance of Circuit Tracer for mechanistic analysis
            aligned_feature_map: Mapping from issue types to lists of aligned features
            **kwargs: Additional arguments for the base class
        """
        super().__init__(model_interface, circuit_tracer_instance, **kwargs)
        self.aligned_feature_map = aligned_feature_map

        logger.info(
            "FeatureAmplificationIntervention initialized with %d issue types",
            len(aligned_feature_map)
        )

    def apply(
            self,
            circuit_analysis_result: Dict[str, Any],
            issue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply feature amplification based on circuit analysis and issue type.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected

        Returns:
            Dictionary with intervention results
        """
        if not issue_type or issue_type not in self.aligned_feature_map:
            logger.warning(
                "Cannot apply feature amplification: unknown issue type '%s'",
                issue_type
            )
            return {"success": False, "reason": "unknown_issue_type"}

        # Get aligned features for this issue type
        aligned_features = self.aligned_feature_map[issue_type]
        if not aligned_features:
            logger.warning("Empty aligned features list for issue type '%s'", issue_type)
            return {"success": False, "reason": "no_aligned_features"}

        # Limit the number of features to modify
        features_to_amplify = aligned_features[:min(len(aligned_features), self.max_features)]
        logger.info(
            "Amplifying %d features for issue type '%s': %s",
            len(features_to_amplify), issue_type, features_to_amplify
        )

        # In a real implementation, this would call the model interface to amplify features
        # For this POC, we simulate the action

        # Record the intervention
        self._record_intervention(
            intervention_type="feature_amplification",
            targets=features_to_amplify,
            metadata={"issue_type": issue_type}
        )

        return {
            "success": True,
            "intervention_type": "feature_amplification",
            "features_amplified": features_to_amplify,
            "strength": self.strength,
            "issue_type": issue_type
        }


class CircuitReroutingIntervention(CircuitTracerIntervention):
    """
    Intervention that modifies the attribution graph to reroute activations.

    This advanced intervention modifies the flow of information through the model's
    internal circuitry by rerouting activations from problematic paths to more
    aligned alternatives.
    """

    def apply(
            self,
            circuit_analysis_result: Dict[str, Any],
            issue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply circuit rerouting based on attribution graph analysis.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected (optional)

        Returns:
            Dictionary with intervention results
        """
        if not circuit_analysis_result or "attribution_graph" not in circuit_analysis_result:
            logger.warning("Cannot apply circuit rerouting: no attribution graph in analysis result")
            return {"success": False, "reason": "no_attribution_graph"}

        # In a real implementation, this would analyze the attribution graph to identify
        # problematic paths and alternative routes
        # For this POC, we simulate the action

        # Simulate identifying problematic paths
        problematic_paths = [
            {"source": f"feature_{i}", "target": f"feature_{i + 5}", "weight": 0.8}
            for i in range(3)
        ]

        # Simulate identifying alternative routes
        alternative_routes = [
            {"source": path["source"], "target": f"feature_{int(path['target'].split('_')[1]) + 10}", "weight": 0.6}
            for path in problematic_paths
        ]

        logger.info(
            "Rerouting %d problematic paths to alternative routes",
            len(problematic_paths)
        )

        # Record the intervention
        self._record_intervention(
            intervention_type="circuit_rerouting",
            targets={"problematic_paths": problematic_paths, "alternative_routes": alternative_routes},
            metadata={"issue_type": issue_type}
        )

        return {
            "success": True,
            "intervention_type": "circuit_rerouting",
            "paths_modified": len(problematic_paths),
            "strength": self.strength,
            "problematic_paths": problematic_paths,
            "alternative_routes": alternative_routes
        }


class JailbreakMitigationIntervention(CircuitTracerIntervention):
    """
    Specialized intervention for mitigating jailbreak attempts.

    This intervention combines multiple strategies specifically designed to
    counteract jailbreak attempts, based on known jailbreak circuits.
    """

    def __init__(
            self,
            model_interface: Any,
            circuit_tracer_instance: Any,
            known_jailbreak_circuits: List[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        Initialize with known jailbreak circuits.

        Args:
            model_interface: Interface to the model for applying interventions
            circuit_tracer_instance: Instance of Circuit Tracer for mechanistic analysis
            known_jailbreak_circuits: List of known jailbreak circuit patterns
            **kwargs: Additional arguments for the base class
        """
        super().__init__(model_interface, circuit_tracer_instance, **kwargs)
        self.known_jailbreak_circuits = known_jailbreak_circuits or []

        # Create sub-interventions
        self.suppression = FeatureSuppressionIntervention(
            model_interface, circuit_tracer_instance, **kwargs
        )

        # Define aligned features for jailbreak mitigation
        aligned_feature_map = {
            "jailbreak": ["safety_feature_1", "safety_feature_2", "instruction_following_feature"],
            "harmful_content": ["safety_feature_3", "safety_feature_4", "ethical_reasoning_feature"]
        }

        self.amplification = FeatureAmplificationIntervention(
            model_interface, circuit_tracer_instance, aligned_feature_map, **kwargs
        )

        logger.info(
            "JailbreakMitigationIntervention initialized with %d known circuits",
            len(self.known_jailbreak_circuits)
        )

    def apply(
            self,
            circuit_analysis_result: Dict[str, Any],
            issue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply jailbreak mitigation strategies based on circuit analysis.

        This method combines multiple intervention strategies:
        1. Suppress features associated with jailbreak attempts
        2. Amplify features associated with instruction following and safety
        3. Apply pattern-based interventions for known jailbreak circuits

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected (should be "jailbreak")

        Returns:
            Dictionary with intervention results
        """
        if issue_type != "jailbreak" and issue_type != "harmful_content":
            logger.warning(
                "JailbreakMitigationIntervention applied to non-jailbreak issue: %s",
                issue_type
            )

        # Apply feature suppression
        suppression_result = self.suppression.apply(circuit_analysis_result, issue_type)

        # Apply feature amplification
        amplification_result = self.amplification.apply(circuit_analysis_result, issue_type or "jailbreak")

        # Check for matches with known jailbreak circuits
        known_circuit_matches = self._match_known_circuits(circuit_analysis_result)

        # Apply pattern-based interventions for known matches
        pattern_interventions = []
        if known_circuit_matches:
            for match in known_circuit_matches:
                logger.info("Applying pattern-based intervention for known jailbreak circuit: %s", match["pattern_id"])
                # In a real implementation, this would apply specific countermeasures
                pattern_interventions.append({
                    "pattern_id": match["pattern_id"],
                    "confidence": match["confidence"],
                    "applied": True
                })

        # Record the combined intervention
        self._record_intervention(
            intervention_type="jailbreak_mitigation",
            targets={
                "suppressed_features": suppression_result.get("features_suppressed", []),
                "amplified_features": amplification_result.get("features_amplified", []),
                "pattern_matches": [m["pattern_id"] for m in known_circuit_matches]
            },
            metadata={"issue_type": issue_type}
        )

        return {
            "success": suppression_result.get("success", False) or amplification_result.get("success", False),
            "intervention_type": "jailbreak_mitigation",
            "suppression_result": suppression_result,
            "amplification_result": amplification_result,
            "known_circuit_matches": known_circuit_matches,
            "pattern_interventions": pattern_interventions,
            "strength": self.strength
        }

    def _match_known_circuits(self, circuit_analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match the current circuit analysis against known jailbreak circuits.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis

        Returns:
            List of matches with confidence scores
        """
        # In a real implementation, this would perform sophisticated pattern matching
        # For this POC, we simulate matches with random confidence

        import random

        matches = []
        if self.known_jailbreak_circuits and random.random() < 0.7:  # 70% chance of finding a match
            num_matches = random.randint(1, min(3, len(self.known_jailbreak_circuits)))
            for i in range(num_matches):
                pattern_id = f"jailbreak_pattern_{i + 1}"
                matches.append({
                    "pattern_id": pattern_id,
                    "confidence": random.uniform(0.6, 0.95),
                    "features": [f"feature_{j}" for j in range(random.randint(2, 5))]
                })

        return matches


class DriftCorrectionIntervention(CircuitTracerIntervention):
    """
    Specialized intervention for correcting alignment drift over time.

    This intervention targets circuit components that have changed over time
    and are contributing to alignment drift.
    """

    def __init__(
            self,
            model_interface: Any,
            circuit_tracer_instance: Any,
            baseline_circuits: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        """
        Initialize with baseline circuits representing aligned behavior.

        Args:
            model_interface: Interface to the model for applying interventions
            circuit_tracer_instance: Instance of Circuit Tracer for mechanistic analysis
            baseline_circuits: Baseline circuits representing aligned behavior
            **kwargs: Additional arguments for the base class
        """
        super().__init__(model_interface, circuit_tracer_instance, **kwargs)
        self.baseline_circuits = baseline_circuits or {}

        logger.info(
            "DriftCorrectionIntervention initialized with %d baseline circuits",
            len(self.baseline_circuits)
        )

    def apply(
            self,
            circuit_analysis_result: Dict[str, Any],
            issue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply drift correction based on comparison with baseline circuits.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis
            issue_type: Type of alignment issue detected (should be "drift")

        Returns:
            Dictionary with intervention results
        """
        if not circuit_analysis_result or "attribution_graph" not in circuit_analysis_result:
            logger.warning("Cannot apply drift correction: no attribution graph in analysis result")
            return {"success": False, "reason": "no_attribution_graph"}

        if not self.baseline_circuits:
            logger.warning("Cannot apply drift correction: no baseline circuits available")
            return {"success": False, "reason": "no_baseline_circuits"}

        # In a real implementation, this would compare the current attribution graph
        # with baseline circuits to identify components that have drifted
        # For this POC, we simulate the comparison

        # Simulate identifying drifted components
        drifted_components = [
            {
                "component_id": f"component_{i}",
                "drift_magnitude": np.random.uniform(0.1, 0.5),
                "baseline_value": np.random.uniform(0.3, 0.7),
                "current_value": np.random.uniform(0.3, 0.7)
            }
            for i in range(5)
        ]

        # Sort by drift magnitude (descending)
        drifted_components.sort(key=lambda x: x["drift_magnitude"], reverse=True)

        # Limit to max_features
        components_to_correct = drifted_components[:self.max_features]

        logger.info(
            "Correcting %d drifted components with average magnitude %.3f",
            len(components_to_correct),
            np.mean([c["drift_magnitude"] for c in components_to_correct])
        )

        # In a real implementation, this would apply corrections to the drifted components
        # For this POC, we simulate the corrections
        corrections = []
        for component in components_to_correct:
            # Calculate correction (move current value toward baseline)
            correction = self.strength * (component["baseline_value"] - component["current_value"])
            new_value = component["current_value"] + correction

            corrections.append({
                "component_id": component["component_id"],
                "original_value": component["current_value"],
                "correction": correction,
                "new_value": new_value
            })

        # Record the intervention
        self._record_intervention(
            intervention_type="drift_correction",
            targets=components_to_correct,
            metadata={"corrections": corrections}
        )

        return {
            "success": True,
            "intervention_type": "drift_correction",
            "components_corrected": len(components_to_correct),
            "average_drift_magnitude": np.mean([c["drift_magnitude"] for c in components_to_correct]),
            "corrections": corrections,
            "strength": self.strength
        }


class InterventionFactory:
    """
    Factory class for creating appropriate interventions based on issue type.

    This class simplifies the process of selecting and configuring the right
    intervention strategy for different types of alignment issues.
    """

    @staticmethod
    def create_intervention(
            issue_type: str,
            model_interface: Any,
            circuit_tracer_instance: Any,
            **kwargs
    ) -> CircuitTracerIntervention:
        """
        Create an appropriate intervention based on the issue type.

        Args:
            issue_type: Type of alignment issue detected
            model_interface: Interface to the model for applying interventions
            circuit_tracer_instance: Instance of Circuit Tracer for mechanistic analysis
            **kwargs: Additional configuration for the intervention

        Returns:
            Configured intervention instance
        """
        if issue_type == "jailbreak" or issue_type == "harmful_content":
            return JailbreakMitigationIntervention(
                model_interface, circuit_tracer_instance, **kwargs
            )
        elif issue_type == "drift":
            return DriftCorrectionIntervention(
                model_interface, circuit_tracer_instance, **kwargs
            )
        elif issue_type == "bias":
            # For bias, we use feature suppression with specific configuration
            return FeatureSuppressionIntervention(
                model_interface, circuit_tracer_instance,
                intervention_strength=kwargs.get("intervention_strength", 0.7),
                **kwargs
            )
        elif issue_type == "hallucination":
            # For hallucination, we use feature amplification with specific configuration
            aligned_feature_map = {
                "hallucination": ["factual_feature_1", "factual_feature_2", "grounding_feature"]
            }
            return FeatureAmplificationIntervention(
                model_interface, circuit_tracer_instance,
                aligned_feature_map=aligned_feature_map,
                intervention_strength=kwargs.get("intervention_strength", 0.8),
                **kwargs
            )
        else:
            # Default to feature suppression for unknown issue types
            logger.warning(
                "Unknown issue type '%s', defaulting to FeatureSuppressionIntervention",
                issue_type
            )
            return FeatureSuppressionIntervention(
                model_interface, circuit_tracer_instance, **kwargs
            )
