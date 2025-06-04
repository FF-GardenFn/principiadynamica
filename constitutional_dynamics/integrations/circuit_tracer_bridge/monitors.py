"""
Circuit Tracer Bridge - Monitors Module

This module provides monitor adapters that connect Constitutional Dynamics' alignment monitoring
capabilities with the Circuit Tracer Bridge. These monitors detect potential alignment issues
and trigger the feedback loop when necessary.
"""

import logging
import numpy as np
import time
from typing import Any, Dict, List, Optional, Union, Tuple

# Import from constitutional_dynamics core modules
from constitutional_dynamics.core.space import AlignmentVectorSpace
from constitutional_dynamics.core.metrics import calculate_stability_metrics, evaluate_alignment_robustness
from constitutional_dynamics.core.transition import predict_trajectory, compute_residual_potentiality

logger = logging.getLogger("constitutional_dynamics.integrations.circuit_tracer_bridge.monitors")


class CircuitTracerMonitorAdapter:
    """
    Adapter class that connects Constitutional Dynamics' monitoring capabilities
    with the Circuit Tracer Bridge.

    This class wraps an existing Constitutional Dynamics monitor (such as AlignmentVectorSpace)
    and adds functionality specific to the Circuit Tracer integration, such as:

    1. Tracking historical alignment scores to detect trends
    2. Providing context-specific thresholds for different types of content
    3. Identifying specific types of alignment issues for targeted circuit analysis
    """

    def __init__(
            self,
            cd_monitor_instance: Any,
            threshold_config: Optional[Dict[str, float]] = None,
            history_window_size: int = 10
    ):
        """
        Initialize the monitor adapter with a Constitutional Dynamics monitor instance.

        Args:
            cd_monitor_instance: An instance of a Constitutional Dynamics monitor
                (typically AlignmentVectorSpace or similar)
            threshold_config: Dictionary mapping content types to threshold values
            history_window_size: Number of recent scores to keep for trend analysis
        """
        self.monitor = cd_monitor_instance
        self.threshold_config = threshold_config or {
            "default": 0.7,
            "sensitive_topics": 0.8,
            "code_generation": 0.6,
            "creative_content": 0.5
        }
        self.history_window_size = history_window_size
        self.score_history = []
        self.issue_types = {
            "jailbreak": [],
            "hallucination": [],
            "bias": [],
            "harmful_content": []
        }

        logger.info("CircuitTracerMonitorAdapter initialized with %d content types",
                    len(self.threshold_config))

    def compute_alignment_score(
            self,
            embedding: List[float],
            content_type: str = "default"
    ) -> float:
        """
        Compute alignment score for the given embedding using the wrapped monitor.

        Args:
            embedding: Vector representation of model output
            content_type: Type of content being evaluated, used for threshold selection

        Returns:
            Alignment score between 0 and 1
        """
        # Delegate to the wrapped monitor for actual score computation
        score = self.monitor.compute_alignment_score(embedding)

        # Update history
        self.score_history.append(score)
        if len(self.score_history) > self.history_window_size:
            self.score_history.pop(0)

        logger.debug("Computed alignment score: %.3f for content type: %s", score, content_type)
        return score

    def get_threshold(self, content_type: str = "default") -> float:
        """
        Get the alignment threshold for the specified content type.

        Args:
            content_type: Type of content being evaluated

        Returns:
            Threshold value between 0 and 1
        """
        return self.threshold_config.get(content_type, self.threshold_config["default"])

    def update_monitoring_focus(self, parameters: Dict[str, Any]) -> bool:
        """
        Update monitoring parameters based on strategy recommendations.

        This method allows the MetaStrategist to adaptively adjust the monitor's
        behavior based on higher-level strategic insights.

        Args:
            parameters: Dictionary of parameters to update, which may include:
                - thresholds: Dict mapping content types to new threshold values
                - history_window_size: New window size for trend analysis
                - sensitivity: Dict mapping issue types to sensitivity multipliers
                - focus_issue_types: List of issue types to prioritize
                - additional_patterns: Dict mapping issue types to new detection patterns

        Returns:
            Boolean indicating whether any parameters were updated
        """
        updated = False

        # Update thresholds if provided
        if "thresholds" in parameters and isinstance(parameters["thresholds"], dict):
            for content_type, threshold in parameters["thresholds"].items():
                if isinstance(threshold, (int, float)) and 0 <= threshold <= 1:
                    self.threshold_config[content_type] = threshold
                    updated = True

            logger.info("Updated threshold configuration: %s", self.threshold_config)

        # Update history window size if provided
        if "history_window_size" in parameters and isinstance(parameters["history_window_size"], int):
            new_size = parameters["history_window_size"]
            if new_size > 0:
                self.history_window_size = new_size
                # Trim history if needed
                if len(self.score_history) > self.history_window_size:
                    self.score_history = self.score_history[-self.history_window_size:]
                updated = True

                logger.info("Updated history window size to %d", self.history_window_size)

        # Store additional monitoring parameters that might be used by specialized monitors
        if not hasattr(self, 'adaptive_parameters'):
            self.adaptive_parameters = {}

        # Update sensitivity for different issue types
        if "sensitivity" in parameters and isinstance(parameters["sensitivity"], dict):
            self.adaptive_parameters["sensitivity"] = parameters["sensitivity"]
            updated = True

            logger.info("Updated sensitivity parameters: %s", parameters["sensitivity"])

        # Update focus issue types
        if "focus_issue_types" in parameters and isinstance(parameters["focus_issue_types"], list):
            self.adaptive_parameters["focus_issue_types"] = parameters["focus_issue_types"]
            updated = True

            logger.info("Updated focus issue types: %s", parameters["focus_issue_types"])

        # Update additional patterns for detection
        if "additional_patterns" in parameters and isinstance(parameters["additional_patterns"], dict):
            self.adaptive_parameters["additional_patterns"] = parameters["additional_patterns"]
            updated = True

            logger.info("Updated additional detection patterns: %s", parameters["additional_patterns"])

        return updated

    def detect_alignment_issues(
            self,
            embedding: List[float],
            content_type: str = "default",
            prompt: Optional[str] = None,
            response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect potential alignment issues in the model output using advanced metrics
        from Constitutional Dynamics core modules.

        This method goes beyond simple threshold checking to identify specific
        types of alignment issues that might require different circuit analysis
        approaches.

        Args:
            embedding: Vector representation of model output
            content_type: Type of content being evaluated
            prompt: Original prompt text (optional, for context-aware detection)
            response: Model response text (optional, for text-based heuristics)

        Returns:
            Dictionary containing detection results and issue types
        """
        score = self.compute_alignment_score(embedding, content_type)
        threshold = self.get_threshold(content_type)

        # Basic threshold check
        below_threshold = score < threshold

        # Check for sudden drops in alignment (potential drift)
        drift_detected = False
        if len(self.score_history) >= 3:
            recent_avg = sum(self.score_history[-3:]) / 3
            previous_avg = sum(self.score_history[:-3]) / max(1, len(self.score_history) - 3)
            drift_detected = recent_avg < previous_avg - 0.1

        # issue detection using Constitutional Dynamics core functionality
        advanced_metrics = {}

        # Compute residual potentiality to estimate potential for alignment issues
        try:
            potentiality = compute_residual_potentiality(embedding, perturbation_magnitude=0.1)
            advanced_metrics["residual_potentiality"] = potentiality

            # Higher potentiality indicates more potential for alignment issues
            high_potentiality = False
            if "potentiality_score" in potentiality:
                high_potentiality = potentiality["potentiality_score"] > 0.7
                advanced_metrics["high_potentiality"] = high_potentiality
        except Exception as e:
            logger.warning(f"Error computing residual potentiality: {e}")
            high_potentiality = False

        # Evaluate alignment robustness if we have access to the monitor's vector space
        robustness_issues = False
        if hasattr(self.monitor, 'aligned_centroid') and self.monitor.aligned_centroid is not None:
            try:
                # Create a temporary AlignmentVectorSpace with the monitor's configuration
                temp_space = AlignmentVectorSpace(dimension=len(embedding))

                # Set the aligned centroid to match the monitor's
                temp_space.aligned_centroid = self.monitor.aligned_centroid

                # Add the current embedding as a state
                temp_space.add_state(embedding)

                # Evaluate alignment robustness
                robustness = evaluate_alignment_robustness(temp_space, 
                                                          perturbation_magnitude=0.05,
                                                          num_perturbations=5)
                advanced_metrics["robustness"] = robustness

                # Lower robustness score indicates higher susceptibility to alignment issues
                if "robustness_score" in robustness:
                    robustness_issues = robustness["robustness_score"] < 0.6
                    advanced_metrics["robustness_issues"] = robustness_issues
            except Exception as e:
                logger.warning(f"Error evaluating alignment robustness: {e}")

        # Determine issue types based on basic and advanced metrics
        issue_types = []

        # Use both basic threshold and advanced metrics to identify specific issue types
        if below_threshold or high_potentiality or robustness_issues:
            # Determine the most likely issue type based on available evidence
            if score < threshold - 0.2 or (high_potentiality and "harmful_content_potential" in potentiality):
                issue_types.append("harmful_content")
            elif drift_detected:
                issue_types.append("drift")
            elif content_type == "sensitive_topics" or (robustness_issues and "demographic_sensitivity" in advanced_metrics.get("robustness", {})):
                issue_types.append("bias")
            elif high_potentiality:
                # If high potentiality but no specific issue identified, assume jailbreak risk
                issue_types.append("jailbreak")
            else:
                # Default case if below threshold but no specific issue identified
                issue_types.append("jailbreak")

        # Log detection results
        logger.info(
            "Alignment issue detection: score=%.3f, threshold=%.3f, below_threshold=%s, high_potentiality=%s, issues=%s",
            score, threshold, below_threshold, high_potentiality, issue_types
        )

        return {
            "score": score,
            "threshold": threshold,
            "below_threshold": below_threshold,
            "drift_detected": drift_detected,
            "issue_types": issue_types,
            "content_type": content_type,
            "advanced_metrics": advanced_metrics,
            "high_potentiality": high_potentiality,
            "robustness_issues": robustness_issues,
            "requires_intervention": below_threshold or drift_detected or high_potentiality or robustness_issues
        }

    def get_issue_specific_circuit_targets(
            self,
            issue_type: str,
            advanced_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get circuit analysis targets specific to the identified issue type,
        using advanced metrics from Constitutional Dynamics core modules if available.

        Different alignment issues may involve different circuits within the model.
        This method provides guidance to Circuit Tracer on where to focus its analysis.

        Args:
            issue_type: Type of alignment issue detected
            advanced_metrics: Optional dictionary of advanced metrics from detect_alignment_issues

        Returns:
            Dictionary with circuit analysis targeting information
        """
        # Base targets for each issue type
        base_targets = {
            "jailbreak": {
                "focus_layers": ["last_3"],
                "attention_heads": [0, 4, 7],
                "feature_types": ["inhibitory"]
            },
            "hallucination": {
                "focus_layers": ["middle"],
                "attention_heads": [2, 3, 5],
                "feature_types": ["factual"]
            },
            "bias": {
                "focus_layers": ["early"],
                "attention_heads": [1, 6],
                "feature_types": ["demographic"]
            },
            "harmful_content": {
                "focus_layers": ["all"],
                "attention_heads": [0, 2, 4, 6],
                "feature_types": ["safety"]
            },
            "drift": {
                "focus_layers": ["all"],
                "attention_heads": "all",
                "feature_types": ["stability"]
            }
        }

        # Get base targets for the issue type
        result = base_targets.get(issue_type, {"focus_layers": ["all"], "attention_heads": "all"})

        # If advanced metrics are available, enhance the targeting
        if advanced_metrics:
            # Add issue-specific enhancements based on advanced metrics
            if issue_type == "jailbreak":
                # For jailbreak, use potentiality information if available
                if "residual_potentiality" in advanced_metrics:
                    potentiality = advanced_metrics["residual_potentiality"]
                    if "high_potential_dimensions" in potentiality:
                        # Target specific dimensions with high potentiality
                        result["high_potential_dimensions"] = potentiality["high_potential_dimensions"]
                    if "potentiality_score" in potentiality and potentiality["potentiality_score"] > 0.8:
                        # For very high potentiality, focus on more layers
                        result["focus_layers"] = ["all"]

                # If robustness issues detected, add robustness to feature types
                if advanced_metrics.get("robustness_issues", False):
                    if "feature_types" in result:
                        result["feature_types"].append("robustness")
                    else:
                        result["feature_types"] = ["inhibitory", "robustness"]

            elif issue_type == "harmful_content":
                # For harmful content, use robustness information if available
                if "robustness" in advanced_metrics:
                    robustness = advanced_metrics["robustness"]
                    if "vulnerable_components" in robustness:
                        # Target specific vulnerable components
                        result["vulnerable_components"] = robustness["vulnerable_components"]

            elif issue_type == "bias":
                # For bias, use specific demographic sensitivity information if available
                if "robustness" in advanced_metrics and "demographic_sensitivity" in advanced_metrics["robustness"]:
                    demographic_sensitivity = advanced_metrics["robustness"]["demographic_sensitivity"]
                    if "sensitive_dimensions" in demographic_sensitivity:
                        # Target specific dimensions with high demographic sensitivity
                        result["sensitive_dimensions"] = demographic_sensitivity["sensitive_dimensions"]

            elif issue_type == "drift":
                # For drift, delegate to the specialized drift detection monitor
                if hasattr(self, "get_drift_specific_circuit_targets"):
                    drift_targets = self.get_drift_specific_circuit_targets(advanced_metrics)
                    # Merge drift-specific targets with base targets
                    for key, value in drift_targets.items():
                        result[key] = value

        logger.info("Circuit targets for issue type '%s': %s", issue_type, result)
        return result


class JailbreakDetectionMonitor(CircuitTracerMonitorAdapter):
    """
    Specialized monitor for detecting jailbreak attempts.

    This monitor extends the base adapter with jailbreak-specific detection logic
    and circuit targeting information.
    """

    def __init__(self, cd_monitor_instance: Any, **kwargs):
        """Initialize with jailbreak-specific configuration"""
        super().__init__(cd_monitor_instance, **kwargs)

        # Override with jailbreak-specific thresholds
        self.threshold_config.update({
            "default": 0.75,
            "instruction_following": 0.85,
            "safety_critical": 0.9
        })

        # Known jailbreak patterns (in a real implementation, these would be embeddings)
        self.jailbreak_patterns = [
            np.random.rand(768).tolist() for _ in range(5)  # Placeholder
        ]

        logger.info("JailbreakDetectionMonitor initialized with %d patterns",
                    len(self.jailbreak_patterns))

    def detect_jailbreak_attempt(
            self,
            embedding: List[float],
            prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Specialized method for detecting jailbreak attempts using advanced metrics
        from Constitutional Dynamics core modules.

        Args:
            embedding: Vector representation of model output
            prompt: Original prompt text (optional, for heuristic detection)

        Returns:
            Dictionary with comprehensive jailbreak detection results
        """
        # Get basic alignment check with advanced metrics
        basic_check = self.detect_alignment_issues(embedding, "instruction_following", prompt)

        # Extract advanced metrics from the basic check
        advanced_metrics = basic_check.get("advanced_metrics", {})
        high_potentiality = basic_check.get("high_potentiality", False)
        robustness_issues = basic_check.get("robustness_issues", False)

        # Additional jailbreak-specific checks
        jailbreak_similarity = self._compute_jailbreak_pattern_similarity(embedding)

        # Enhanced jailbreak detection using advanced metrics
        jailbreak_evidence = []

        # Basic evidence: low alignment score or high pattern similarity
        if basic_check["below_threshold"]:
            jailbreak_evidence.append("low_alignment_score")

        if jailbreak_similarity > 0.6:
            jailbreak_evidence.append("high_pattern_similarity")

        # Advanced evidence from Constitutional Dynamics core metrics
        if high_potentiality:
            jailbreak_evidence.append("high_residual_potentiality")

        if robustness_issues:
            jailbreak_evidence.append("low_robustness")

        # Check for specific jailbreak indicators in potentiality
        if "residual_potentiality" in advanced_metrics:
            potentiality = advanced_metrics["residual_potentiality"]
            if "jailbreak_indicators" in potentiality:
                jailbreak_evidence.append("specific_jailbreak_indicators")

        # Compute a more comprehensive jailbreak score
        # Weight factors: alignment (40%), pattern similarity (30%), potentiality (20%), robustness (10%)
        jailbreak_score = (
            0.4 * (1 - basic_check["score"]) + 
            0.3 * jailbreak_similarity +
            0.2 * (advanced_metrics.get("residual_potentiality", {}).get("potentiality_score", 0.0)) +
            0.1 * (1 - advanced_metrics.get("robustness", {}).get("robustness_score", 1.0))
        )

        # Determine if a jailbreak attempt is detected
        jailbreak_detected = (
            jailbreak_score > 0.6 or 
            basic_check["below_threshold"] or 
            len(jailbreak_evidence) >= 2  # At least two pieces of evidence
        )

        logger.info(
            "Enhanced jailbreak detection: score=%.3f, evidence=%s, detected=%s",
            jailbreak_score, jailbreak_evidence, jailbreak_detected
        )

        # Get circuit targets if jailbreak detected
        circuit_targets = None
        if jailbreak_detected:
            circuit_targets = self.get_issue_specific_circuit_targets("jailbreak", advanced_metrics)

        return {
            "jailbreak_detected": jailbreak_detected,
            "jailbreak_score": jailbreak_score,
            "jailbreak_evidence": jailbreak_evidence,
            "basic_alignment": basic_check,
            "pattern_similarity": jailbreak_similarity,
            "advanced_metrics": advanced_metrics,
            "circuit_targets": circuit_targets,
            "requires_intervention": jailbreak_detected
        }

    def _compute_jailbreak_pattern_similarity(self, embedding: List[float]) -> float:
        """
        Compute similarity to known jailbreak patterns.

        Args:
            embedding: Vector representation of model output

        Returns:
            Similarity score between 0 and 1
        """
        # Use the monitor's compute_similarity method to calculate similarity
        # to known jailbreak patterns
        # Trying to demonstrate a point here.

        # For demonstration purposes, we'll use a placeholder pattern
        # In a real implementation, this would use actual jailbreak patterns
        if not hasattr(self, '_jailbreak_patterns'):
            # Create placeholder patterns for demonstration
            self._jailbreak_patterns = []
            for _ in range(3):
                # Create random unit vectors as placeholder patterns
                pattern = np.random.normal(0, 1, size=len(embedding))
                pattern = pattern / np.linalg.norm(pattern)
                self._jailbreak_patterns.append(pattern.tolist())

        # Compute similarity to each pattern and take the maximum
        max_similarity = 0.0
        for pattern in self._jailbreak_patterns:
            # Use the monitor's compute_similarity method if available
            if hasattr(self.monitor, 'compute_similarity'):
                # Convert to 0-1 range (similarity is -1 to 1 for cosine)
                similarity = (self.monitor.compute_similarity(embedding, pattern) + 1) / 2
            else:
                # Fallback to AlignmentVectorSpace's compute_similarity method
                # Create a temporary AlignmentVectorSpace instance with the right dimension
                temp_space = AlignmentVectorSpace(dimension=len(embedding))
                # Compute similarity using the imported class
                similarity = (temp_space.compute_similarity(embedding, pattern) + 1) / 2

            max_similarity = max(max_similarity, similarity)

        return max_similarity


class DriftDetectionMonitor(CircuitTracerMonitorAdapter):
    """
    Specialized monitor for detecting alignment drift over time.

    This monitor extends the base adapter with drift-specific detection logic
    and trend analysis capabilities.
    """

    def __init__(
            self,
            cd_monitor_instance: Any,
            drift_threshold: float = 0.1,
            window_sizes: List[int] = [5, 10, 20],
            **kwargs
    ):
        """
        Initialize with drift-specific configuration.

        Args:
            cd_monitor_instance: An instance of a Constitutional Dynamics monitor
            drift_threshold: Threshold for considering a change as drift
            window_sizes: List of window sizes for multi-scale drift detection
            **kwargs: Additional arguments for the base adapter
        """
        super().__init__(cd_monitor_instance, **kwargs)

        self.drift_threshold = drift_threshold
        self.window_sizes = window_sizes
        self.score_history_extended = []

        logger.info("DriftDetectionMonitor initialized with drift threshold %.3f",
                    drift_threshold)

    def compute_alignment_score(
            self,
            embedding: List[float],
            content_type: str = "default",
            timestamp: Optional[float] = None
    ) -> float:
        """
        Compute alignment score and track with timestamp for drift analysis.

        Args:
            embedding: Vector representation of model output
            content_type: Type of content being evaluated
            timestamp: Optional timestamp for the measurement

        Returns:
            Alignment score between 0 and 1
        """
        score = super().compute_alignment_score(embedding, content_type)

        # Track with timestamp for drift analysis
        self.score_history_extended.append({
            "score": score,
            "timestamp": timestamp or time.time(),
            "content_type": content_type
        })

        # Limit history size
        max_history = max(self.window_sizes) if self.window_sizes else 100
        if len(self.score_history_extended) > max_history:
            self.score_history_extended = self.score_history_extended[-max_history:]

        return score

    def detect_drift(
            self,
            current_embedding: List[float],
            content_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Detect alignment drift across multiple time scales using advanced metrics
        from Constitutional Dynamics core modules.

        Args:
            current_embedding: Vector representation of current model output
            content_type: Type of content being evaluated

        Returns:
            Dictionary with comprehensive drift detection results
        """
        # Compute current score
        current_score = self.compute_alignment_score(current_embedding, content_type)

        # Not enough history for drift detection
        if len(self.score_history_extended) < min(self.window_sizes):
            return {
                "drift_detected": False,
                "current_score": current_score,
                "reason": "insufficient_history",
                "requires_intervention": False
            }

        # Compute drift at multiple time scales (basic approach)
        drift_results = {}
        for window in self.window_sizes:
            if len(self.score_history_extended) >= window:
                window_scores = [entry["score"] for entry in self.score_history_extended[-window:]]
                avg_score = sum(window_scores) / window
                drift_magnitude = abs(current_score - avg_score)
                drift_detected = drift_magnitude > self.drift_threshold

                drift_results[f"window_{window}"] = {
                    "average_score": avg_score,
                    "drift_magnitude": drift_magnitude,
                    "drift_detected": drift_detected
                }

        # Combine results from multiple windows
        any_drift_detected = any(result["drift_detected"] for result in drift_results.values())
        max_magnitude = max(result["drift_magnitude"] for result in drift_results.values())

        # Enhanced drift detection using Constitutional Dynamics core functionality
        advanced_metrics = {}

        # Create a temporary AlignmentVectorSpace with the monitor's configuration
        temp_space = AlignmentVectorSpace(dimension=len(current_embedding))

        # Add historical states to the space
        for i, entry in enumerate(self.score_history_extended):
            if "embedding" in entry:
                temp_space.add_state(entry["embedding"], entry["timestamp"])
            elif i > 0 and hasattr(self.monitor, "state_history") and i < len(self.monitor.state_history):
                # If we have access to the monitor's state history, use it
                temp_space.add_state(self.monitor.state_history[i], entry["timestamp"])

        # Add current state
        current_state_idx = temp_space.add_state(current_embedding, time.time())

        # If we have enough history, compute advanced metrics
        if len(temp_space.state_history) > 2:
            # Calculate stability metrics
            try:
                stability_metrics = calculate_stability_metrics(temp_space)
                advanced_metrics["stability"] = stability_metrics

                # Higher Lyapunov exponent indicates more instability/drift
                if "lyapunov_estimate" in stability_metrics:
                    advanced_metrics["drift_indicator"] = stability_metrics["lyapunov_estimate"] > 0.5
            except Exception as e:
                logger.warning(f"Error calculating stability metrics: {e}")

            # Evaluate alignment robustness
            try:
                robustness = evaluate_alignment_robustness(temp_space, 
                                                          perturbation_magnitude=0.05,
                                                          num_perturbations=5)
                advanced_metrics["robustness"] = robustness

                # Lower robustness score indicates higher susceptibility to drift
                if "robustness_score" in robustness:
                    advanced_metrics["drift_susceptibility"] = robustness["robustness_score"] < 0.6
            except Exception as e:
                logger.warning(f"Error evaluating alignment robustness: {e}")

            # Predict future trajectory
            try:
                if current_state_idx > 0:
                    trajectory = predict_trajectory(temp_space, current_state_idx - 1, steps=3)
                    advanced_metrics["predicted_trajectory"] = trajectory

                    # Check if predicted trajectory shows decreasing alignment
                    if "predicted_states" in trajectory and len(trajectory["predicted_states"]) > 1:
                        predicted_scores = [temp_space.compute_alignment_score(state) 
                                           for state in trajectory["predicted_states"]]
                        advanced_metrics["predicted_drift"] = predicted_scores[-1] < predicted_scores[0]
            except Exception as e:
                logger.warning(f"Error predicting trajectory: {e}")

            # Compute residual potentiality
            try:
                potentiality = compute_residual_potentiality(current_embedding, perturbation_magnitude=0.1)
                advanced_metrics["residual_potentiality"] = potentiality

                # Higher potentiality indicates more potential for drift
                if "potentiality_score" in potentiality:
                    advanced_metrics["drift_potential"] = potentiality["potentiality_score"] > 0.7
            except Exception as e:
                logger.warning(f"Error computing residual potentiality: {e}")

        # Combine basic and advanced detection results
        advanced_drift_detected = any([
            advanced_metrics.get("drift_indicator", False),
            advanced_metrics.get("drift_susceptibility", False),
            advanced_metrics.get("predicted_drift", False),
            advanced_metrics.get("drift_potential", False)
        ])

        # Final drift detection combines both approaches
        final_drift_detected = any_drift_detected or advanced_drift_detected

        logger.info(
            "Drift detection: current_score=%.3f, max_drift_magnitude=%.3f, basic_drift=%s, advanced_drift=%s",
            current_score, max_magnitude, any_drift_detected, advanced_drift_detected
        )

        return {
            "drift_detected": final_drift_detected,
            "current_score": current_score,
            "max_drift_magnitude": max_magnitude,
            "window_results": drift_results,
            "advanced_metrics": advanced_metrics,
            "basic_drift_detected": any_drift_detected,
            "advanced_drift_detected": advanced_drift_detected,
            "requires_intervention": final_drift_detected and (
                max_magnitude > self.drift_threshold * 1.5 or advanced_drift_detected
            )
        }

    def get_drift_specific_circuit_targets(self, advanced_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get circuit analysis targets specific to the detected drift pattern,
        using advanced metrics from Constitutional Dynamics core modules.

        Args:
            advanced_metrics: Optional dictionary of advanced metrics from detect_drift

        Returns:
            Dictionary with circuit analysis targeting information for drift
        """
        # Default targeting if no advanced metrics are available
        if not advanced_metrics:
            return {
                "focus_layers": ["all"],
                "attention_heads": "all",
                "feature_types": ["stability", "consistency"],
                "drift_specific": True,
                "prioritize_recent_changes": True
            }

        # Use advanced metrics to provide more targeted recommendations
        targets = {
            "drift_specific": True,
            "prioritize_recent_changes": True
        }

        # Determine which layers to focus on based on stability metrics
        if "stability" in advanced_metrics and "unstable_components" in advanced_metrics["stability"]:
            unstable_components = advanced_metrics["stability"]["unstable_components"]
            if "layers" in unstable_components:
                targets["focus_layers"] = unstable_components["layers"]
            else:
                targets["focus_layers"] = ["all"]

            if "attention_heads" in unstable_components:
                targets["attention_heads"] = unstable_components["attention_heads"]
            else:
                targets["attention_heads"] = "all"
        else:
            targets["focus_layers"] = ["all"]
            targets["attention_heads"] = "all"

        # Determine feature types to focus on based on robustness and trajectory
        feature_types = ["stability", "consistency"]

        if "robustness" in advanced_metrics and advanced_metrics.get("drift_susceptibility", False):
            feature_types.append("robustness")

        if "predicted_trajectory" in advanced_metrics and advanced_metrics.get("predicted_drift", False):
            feature_types.append("trajectory")

        if "residual_potentiality" in advanced_metrics and advanced_metrics.get("drift_potential", False):
            feature_types.append("potentiality")

        targets["feature_types"] = feature_types

        # Add specific targeting information based on advanced metrics
        if "predicted_trajectory" in advanced_metrics:
            trajectory = advanced_metrics["predicted_trajectory"]
            if "critical_transition_points" in trajectory:
                targets["critical_points"] = trajectory["critical_transition_points"]

        if "residual_potentiality" in advanced_metrics:
            potentiality = advanced_metrics["residual_potentiality"]
            if "high_potential_dimensions" in potentiality:
                targets["high_potential_dimensions"] = potentiality["high_potential_dimensions"]

        logger.info(
            "Drift-specific circuit targets: focus_layers=%s, feature_types=%s",
            targets.get("focus_layers", ["all"]), targets.get("feature_types", [])
        )

        return targets
