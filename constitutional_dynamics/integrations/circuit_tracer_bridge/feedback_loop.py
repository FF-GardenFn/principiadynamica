"""
Circuit Tracer Bridge - Feedback Loop Module

This module implements the core feedback loop between Constitutional Dynamics' alignment monitoring
and Circuit Tracer's mechanistic interpretability tools, creating an "Alignment Thermostat" that
can detect, analyze, and correct alignment issues in language models.

# Attribution:
# - Core feedback loop architecture inspired by Anthropic, circuit-tracer team, and recent interpretability research (Ameisen et al. 2025, Lindsey et al. 2025).
# - Code is a scaffold for both demonstration and real integration.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

# Import from constitutional_dynamics core modules
from constitutional_dynamics.core.metrics import calculate_stability_metrics, evaluate_alignment_robustness
from constitutional_dynamics.core.transition import predict_trajectory, compute_residual_potentiality, compute_activation_probability

# Import strategist for higher-level strategy recommendations
from constitutional_dynamics.integrations.strategist import create_strategist, MetaStrategist

logger = logging.getLogger("constitutional_dynamics.integrations.circuit_tracer_bridge")


class AlignmentThermostat:
    """
    A feedback loop system that combines Constitutional Dynamics' alignment monitoring with
    Circuit Tracer's mechanistic interpretability to detect and correct alignment issues.

    The AlignmentThermostat acts as a regulatory system that:
    1. Monitors model outputs using Constitutional Dynamics' alignment metrics
    2. Triggers Circuit Tracer analysis when alignment issues are detected
    3. Identifies specific circuit components to target for intervention
    4. Applies targeted interventions based on mechanistic insights
    5. Verifies if the intervention improved alignment

    This creates a closed-loop system that can continuously monitor and improve model alignment.
    """

    def __init__(
            self,
            cd_monitor_instance: Any,
            circuit_tracer_instance: Any,
            model_interface: Any,
            threshold: float = 0.7,
            stability_weight: float = 0.3,
            auto_stabilize: bool = True,
            enable_strategist: bool = True,
            strategist_instance: Optional[MetaStrategist] = None
    ):
        """
        Initialize the AlignmentThermostat with necessary components.

        Args:
            cd_monitor_instance: An instance of Constitutional Dynamics' monitoring system
                (typically an AlignmentVectorSpace or similar)
            circuit_tracer_instance: An instance of Circuit Tracer for mechanistic analysis
            model_interface: Interface to the model for applying interventions
            threshold: Alignment score threshold below which interventions are triggered
            stability_weight: Weight given to stability metrics in activation probability
            auto_stabilize: Whether to automatically stabilize the system based on drift detection
            enable_strategist: Whether to enable the MetaStrategist integration
            strategist_instance: Optional pre-configured MetaStrategist instance
        """
        self.monitor = cd_monitor_instance
        self.tracer = circuit_tracer_instance
        self.model = model_interface
        self.threshold = threshold
        self.stability_weight = stability_weight
        self.auto_stabilize = auto_stabilize
        self.enable_strategist = enable_strategist
        self.strategist = strategist_instance
        self.intervention_history = []
        self.stability_history = []
        self.lyapunov_estimate = 0.0
        logger.info("AlignmentThermostat initialized with threshold %.2f, stability_weight %.2f, auto_stabilize=%s, enable_strategist=%s", 
                   threshold, stability_weight, auto_stabilize, enable_strategist)

    def identify_intervention_targets(self, circuit_analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze circuit analysis results to identify specific components for intervention.

        This method translates Circuit Tracer's mechanistic insights into actionable
        intervention targets. It identifies the most influential circuit components
        that are likely contributing to alignment issues.

        Args:
            circuit_analysis_result: Results from Circuit Tracer analysis

        Returns:
            Dictionary of intervention targets or None if no suitable targets found
        """
        logger.info("Identifying intervention targets from circuit analysis")

        # Currently a placeholder, this would contain the appropriate logic to analyze
        # the circuit analysis results and determine which components to target.
        # For now, we implement a simple placeholder that looks for critical features
        # For poc and demonstration purposes.

        if not circuit_analysis_result:
            logger.warning("Empty circuit analysis result, cannot identify targets")
            return None

        if "critical_features" in circuit_analysis_result and circuit_analysis_result["critical_features"]:
            # Extract the most influential features for intervention
            targets = {
                "suppress_features": circuit_analysis_result["critical_features"][:3],
                "analysis_summary": circuit_analysis_result.get("summary", "No summary available")
            }
            logger.info("Identified intervention targets: %s", targets)
            return targets

        logger.info("No suitable intervention targets identified")
        return None

    def apply_intervention(
            self, 
            intervention_targets: Dict[str, Any],
            stability_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Apply targeted interventions to the model based on identified targets.

        This method takes the intervention targets identified from circuit analysis
        and applies appropriate modifications to the model's behavior, taking into
        account stability metrics if available.

        Args:
            intervention_targets: Dictionary containing intervention specifications
            stability_metrics: Optional stability metrics to guide the intervention

        Returns:
            Boolean indicating whether intervention was successfully applied
        """
        if not intervention_targets:
            logger.warning("No intervention targets provided")
            return False

        logger.info("Applying intervention: %s", intervention_targets)

        # Extract stability context and modulated probabilities if available
        stability_context = intervention_targets.get("stability_context", {})
        modulated_probabilities = intervention_targets.get("modulated_probabilities", [])

        # Log stability-aware intervention details if available
        if stability_context:
            logger.info(
                "Stability-aware intervention: lyapunov=%.3f, stability_score=%.3f, reason=%s",
                stability_context.get("lyapunov_estimate", 0.0),
                stability_context.get("stability_score", 1.0),
                stability_context.get("intervention_reason", "unknown")
            )

        # Record the intervention for later analysis
        record = {
            "action": "suppress" if "suppress_features" in intervention_targets else "other",
            "targets": intervention_targets,
            "timestamp": time.time()
        }

        # Add stability information if available
        if stability_metrics:
            record["stability_metrics"] = stability_metrics
        if modulated_probabilities:
            record["modulated_probabilities"] = modulated_probabilities

        self.intervention_history.append(record)

        # Currently a placeholder, this would contain the interaction with the model interface
        # to apply the specified interventions. For now, we log the action.
        # In a real implementation, we would use the modulated probabilities to determine
        # the strength of the intervention for each feature.
        if "suppress_features" in intervention_targets:
            features = intervention_targets["suppress_features"]

            if modulated_probabilities and len(modulated_probabilities) == len(features):
                # Use modulated probabilities to determine intervention strength for each feature
                logger.info(
                    "Suppressing features with modulated probabilities: %s",
                    list(zip(features, [f"{p:.3f}" for p in modulated_probabilities]))
                )

                # In a real implementation, we would apply different suppression strengths
                # based on the modulated probabilities
                # Example: self.model.suppress_features_with_strengths(features, modulated_probabilities)
            else:
                # Fall back to regular suppression if no modulated probabilities
                logger.info("Suppressing features: %s", features)
                # self.model.suppress_features(features)  # Actual implementation would call the model

            return True

        return False

    def calculate_modulated_activation_probability(
            self,
            base_probability: float,
            state: List[float],
            subset_weight: float = 1.0,
            memory_factor: Optional[float] = None,
            environment_factor: Optional[float] = None
    ) -> float:
        """
        Calculate activation probability modulated by stability metrics.

        This implements the core concept of the "Alignment Thermostat" by modulating
        the activation probability (ϕ) based on the system's measured behavioral
        drift/stability (Lyapunov exponent estimate).

        Args:
            base_probability: Base activation probability
            state: Current state vector
            subset_weight: Weight of the state subset (W')
            memory_factor: Memory factor (M(t))
            environment_factor: Environment factor (E(t))

        Returns:
            Modulated activation probability
        """
        # First, calculate the base activation probability using the standard STC formula
        # ϕ = f(W', M(t), E(t))
        if memory_factor is None and environment_factor is None:
            # If no memory or environment factors provided, use compute_activation_probability
            # from constitutional_dynamics.core.transition
            activation_prob = compute_activation_probability(
                subset_weight=subset_weight,
                state=state,
                memory=None,  # Use default memory
                environment=None  # Use default environment
            )
        else:
            # Simple formula if factors are provided directly
            activation_prob = base_probability * subset_weight
            if memory_factor is not None:
                activation_prob *= memory_factor
            if environment_factor is not None:
                activation_prob *= environment_factor

        # Now modulate the activation probability based on the Lyapunov exponent estimate
        # Higher Lyapunov exponent (more chaotic/unstable) -> lower activation probability
        # This creates a stabilizing feedback loop
        stability_factor = 1.0

        if self.lyapunov_estimate > 0:
            # Exponential dampening based on Lyapunov estimate
            # As system becomes more chaotic, we more aggressively reduce activation probability
            stability_factor = math.exp(-self.stability_weight * self.lyapunov_estimate)

            logger.debug(
                "Modulating activation probability: base=%.3f, lyapunov=%.3f, stability_factor=%.3f",
                activation_prob, self.lyapunov_estimate, stability_factor
            )

        # Apply the stability factor to modulate the activation probability
        modulated_prob = activation_prob * stability_factor

        # Ensure the probability stays in [0, 1]
        modulated_prob = max(0.0, min(1.0, modulated_prob))

        return modulated_prob

    def update_stability_metrics(self, current_embedding: List[float]) -> Dict[str, Any]:
        """
        Update stability metrics based on the current state.

        This method calculates stability metrics including the Lyapunov exponent estimate
        and updates the internal state of the AlignmentThermostat.

        Args:
            current_embedding: Current state vector

        Returns:
            Dictionary with stability metrics
        """
        # Add the current state to the monitor if it has an add_state method
        if hasattr(self.monitor, 'add_state'):
            self.monitor.add_state(current_embedding)

            # Calculate stability metrics if we have enough states
            if len(self.monitor.state_history) >= 3:
                try:
                    stability_metrics = calculate_stability_metrics(self.monitor)

                    # Update the Lyapunov exponent estimate
                    if "lyapunov_exponent_estimate" in stability_metrics:
                        self.lyapunov_estimate = stability_metrics["lyapunov_exponent_estimate"]

                    # Add to stability history
                    self.stability_history.append(stability_metrics)
                    if len(self.stability_history) > 10:  # Keep last 10 entries
                        self.stability_history.pop(0)

                    logger.info(
                        "Updated stability metrics: lyapunov=%.3f, stability_score=%.3f",
                        self.lyapunov_estimate, stability_metrics.get("stability_score", 0.0)
                    )

                    return stability_metrics
                except Exception as e:
                    logger.warning(f"Error calculating stability metrics: {e}")

        # If we couldn't calculate stability metrics, return a default value
        return {"lyapunov_exponent_estimate": self.lyapunov_estimate}

    def verify_improvement(self, previous_output: Any, new_output: Any) -> Dict[str, Any]:
        """
        Verify whether the intervention improved alignment.

        This method compares alignment scores before and after intervention
        to determine if the intervention was successful.

        Args:
            previous_output: Model output before intervention
            new_output: Model output after intervention

        Returns:
            Dictionary with verification results including improvement metrics
        """
        prev_score = self.monitor.compute_alignment_score(previous_output)
        new_score = self.monitor.compute_alignment_score(new_output)

        improvement = new_score > prev_score
        improvement_margin = new_score - prev_score

        # Update stability metrics with the new output
        stability_metrics = self.update_stability_metrics(new_output)

        logger.info(
            "Verification: Previous Score=%.3f, New Score=%.3f, Improved=%s, Margin=%.3f, Lyapunov=%.3f",
            prev_score, new_score, improvement, improvement_margin, 
            stability_metrics.get("lyapunov_exponent_estimate", 0.0)
        )

        return {
            "improved": improvement,
            "previous_score": prev_score,
            "new_score": new_score,
            "improvement_margin": improvement_margin,
            "above_threshold": new_score >= self.threshold,
            "stability_metrics": stability_metrics
        }

    def run_feedback_loop(
            self,
            current_model_output_embedding: List[float],
            original_prompt_for_trace: str,
            content_type: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute the complete feedback loop from monitoring to intervention verification.

        This is the main method that orchestrates the entire process:
        1. Monitor alignment using Constitutional Dynamics
        2. Update stability metrics and calculate Lyapunov exponent estimate
        3. If alignment issues detected or auto-stabilization needed, trigger Circuit Tracer analysis
        4. Identify intervention targets from circuit analysis
        5. Apply interventions to the model with activation probabilities modulated by stability
        6. Verify if the intervention improved alignment

        Args:
            current_model_output_embedding: Embedding of the current model output
            original_prompt_for_trace: The original prompt that generated the output
            content_type: Type of content being evaluated (for threshold selection)

        Returns:
            Dictionary containing results of the feedback loop execution
        """
        logger.info("Running alignment feedback loop with stability-modulated activation")

        # Step 1: Monitor alignment using Constitutional Dynamics
        alignment_score = self.monitor.compute_alignment_score(current_model_output_embedding)

        # Step 2: Update stability metrics and calculate Lyapunov exponent estimate
        stability_metrics = self.update_stability_metrics(current_model_output_embedding)
        lyapunov_estimate = stability_metrics.get("lyapunov_exponent_estimate", 0.0)
        stability_score = stability_metrics.get("stability_score", 1.0)

        logger.info("Current alignment score: %.3f (threshold: %.3f), Lyapunov estimate: %.3f, Stability score: %.3f", 
                   alignment_score, self.threshold, lyapunov_estimate, stability_score)

        # Determine if intervention is needed based on alignment score and stability
        intervention_needed = False
        intervention_reason = None

        # Check alignment threshold
        if alignment_score < self.threshold:
            intervention_needed = True
            intervention_reason = "alignment_below_threshold"

        # Check stability for auto-stabilization
        if self.auto_stabilize and lyapunov_estimate > 0.5:  # High Lyapunov exponent indicates instability
            intervention_needed = True
            intervention_reason = intervention_reason or "high_instability"

            logger.warning(
                "System instability detected (Lyapunov estimate: %.3f). Auto-stabilization triggered.",
                lyapunov_estimate
            )

        # If no intervention needed, return early
        if not intervention_needed:
            logger.info("No intervention needed. Alignment and stability are acceptable.")
            return {
                "intervention_applied": False,
                "alignment_score": alignment_score,
                "stability_metrics": stability_metrics,
                "reason": "metrics_acceptable"
            }

        # Step 3: Trigger Circuit Tracer analysis
        logger.warning(
            "Intervention needed (reason: %s). Triggering mechanistic analysis.",
            intervention_reason
        )

        # Call the actual Circuit Tracer to analyze the model's output
        try:
            from circuit_tracer.attribution import attribute
            from circuit_tracer.graph import prune_graph

            logger.info("Starting Circuit Tracer analysis")

            # Assuming self.tracer is a ReplacementModel instance
            graph = attribute(
                prompt=original_prompt_for_trace,
                model=self.tracer,
                verbose=True,
                # Use default values for other parameters
                max_n_logits=10,
                desired_logit_prob=0.95,
                batch_size=512,
                max_feature_nodes=None,
                offload=None
            )

            # Prune the graph to identify the most critical features
            logger.info("Pruning graph to identify critical features")
            prune_result = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)

            # Get the most influential features
            # Extract feature indices from selected_features that have high influence
            node_mask = prune_result.node_mask
            n_features = len(graph.selected_features)

            # Get the indices of the most influential features
            influential_feature_indices = graph.selected_features[node_mask[:n_features]]

            # Extract layer, position, and feature_idx for each influential feature
            critical_features = []
            for idx in influential_feature_indices:
                feature_info = graph.active_features[idx]
                layer, pos, feature_idx = feature_info.tolist()
                critical_features.append(f"layer_{layer}_pos_{pos}_feature_{feature_idx}")

            # If no critical features were found, use the top features by activation value
            if not critical_features:
                logger.warning("No critical features found in pruned graph, using top activations")
                # Get top 3 features by activation value
                top_activations = torch.argsort(graph.activation_values, descending=True)[:3]
                for idx in top_activations:
                    feature_info = graph.active_features[idx]
                    layer, pos, feature_idx = feature_info.tolist()
                    critical_features.append(f"layer_{layer}_pos_{pos}_feature_{feature_idx}")

            # Create the circuit analysis result
            circuit_analysis_result = {
                "critical_features": critical_features[:3],  # Limit to top 3 features
                "summary": f"Circuit trace identified {len(critical_features)} critical features",
                "graph": graph  # Include the full graph for potential further analysis
            }

            logger.info("Circuit tracer analysis complete")

        except Exception as e:
            logger.error("Circuit tracer analysis failed: %s", str(e))
            return {
                "intervention_applied": False,
                "alignment_score": alignment_score,
                "reason": "circuit_analysis_failed",
                "error": str(e)
            }

        # Step 4: Identify intervention targets with stability-aware targeting
        intervention_targets = self.identify_intervention_targets(circuit_analysis_result)

        if not intervention_targets:
            logger.info("No intervention targets identified from circuit analysis.")
            return {
                "intervention_applied": False,
                "alignment_score": alignment_score,
                "stability_metrics": stability_metrics,
                "reason": "no_targets_identified"
            }

        # Modulate intervention targets based on stability metrics
        if "suppress_features" in intervention_targets and intervention_targets["suppress_features"]:
            # Calculate modulated activation probabilities for each feature
            modulated_probabilities = []
            for i, feature in enumerate(intervention_targets["suppress_features"]):
                # Base probability decreases with feature index (higher for more important features)
                base_prob = 1.0 - (i / len(intervention_targets["suppress_features"]))

                # Modulate based on stability
                modulated_prob = self.calculate_modulated_activation_probability(
                    base_probability=base_prob,
                    state=current_model_output_embedding,
                    subset_weight=1.0 - (0.1 * lyapunov_estimate)  # Reduce weight as instability increases
                )

                modulated_probabilities.append(modulated_prob)

            # Add modulated probabilities to intervention targets
            intervention_targets["modulated_probabilities"] = modulated_probabilities

            logger.info(
                "Modulated activation probabilities for %d features: %s",
                len(modulated_probabilities),
                ", ".join([f"{p:.3f}" for p in modulated_probabilities])
            )

            # Add stability context to intervention targets
            intervention_targets["stability_context"] = {
                "lyapunov_estimate": lyapunov_estimate,
                "stability_score": stability_score,
                "intervention_reason": intervention_reason
            }

        # Step 5: Apply intervention with stability-modulated activation
        intervention_success = self.apply_intervention(
            intervention_targets,
            stability_metrics=stability_metrics
        )

        if not intervention_success:
            logger.warning("Failed to apply intervention.")
            return {
                "intervention_applied": False,
                "alignment_score": alignment_score,
                "stability_metrics": stability_metrics,
                "reason": "intervention_application_failed"
            }

        # Step 6: Get new model output after intervention
        # In a real implementation, this would re-run the model with the intervention applied
        # For now, we simulate an improved output that takes into account stability metrics

        # In a real implementation, this would re-run the model with the intervention applied
        # For demonstration purposes, we use the monitor's functionality to create an improved embedding

        # Get the aligned centroid from the monitor
        aligned_centroid = getattr(self.monitor, 'aligned_centroid', None)

        if aligned_centroid is not None:
            # Create a new embedding that's closer to the aligned centroid
            mock_new_output_embedding = []

            # Use the monitor's vector operations to move toward aligned space
            for i in range(len(current_model_output_embedding)):
                if i < len(aligned_centroid):
                    # Move 10% closer to the aligned centroid
                    value = current_model_output_embedding[i] + 0.1 * (aligned_centroid[i] - current_model_output_embedding[i])
                    mock_new_output_embedding.append(value)
                else:
                    mock_new_output_embedding.append(current_model_output_embedding[i])

            # Use numpy for normalization to ensure consistency with other vector operations
            mock_new_output_embedding = np.array(mock_new_output_embedding)
            norm = np.linalg.norm(mock_new_output_embedding)
            if norm > 0:
                mock_new_output_embedding = (mock_new_output_embedding / norm).tolist()
        else:
            # If no aligned centroid is available, make a small random improvement
            logger.warning("No aligned centroid available, using fallback improvement method")
            mock_new_output_embedding = list(current_model_output_embedding)
            # Add a small random perturbation that's likely to improve alignment
            mock_new_output_embedding = np.array(mock_new_output_embedding)
            mock_new_output_embedding += np.random.normal(0, 0.01, size=mock_new_output_embedding.shape)
            norm = np.linalg.norm(mock_new_output_embedding)
            if norm > 0:
                mock_new_output_embedding = (mock_new_output_embedding / norm).tolist()

        # Step 7: Verify improvement
        verification_result = self.verify_improvement(
            current_model_output_embedding,
            mock_new_output_embedding
        )

        # Calculate new stability metrics after intervention
        new_stability_metrics = verification_result.get("stability_metrics", {})
        new_lyapunov_estimate = new_stability_metrics.get("lyapunov_exponent_estimate", 0.0)

        # Check if stability improved
        stability_improved = new_lyapunov_estimate < lyapunov_estimate

        logger.info(
            "Stability metrics: Before intervention: Lyapunov=%.3f, After: Lyapunov=%.3f, Improved=%s",
            lyapunov_estimate, new_lyapunov_estimate, stability_improved
        )

        # Check if we should consult the MetaStrategist
        strategy_recommendation = None
        # Get strategist_context from intervention_targets if available
        strategist_context = intervention_targets.get("strategist_context", {})

        if self.enable_strategist and strategist_context:
            logger.info("Intervention applied, preparing to consult MetaStrategist.")

            # Create strategist if not exists
            if not self.strategist:
                logger.info("Creating MetaStrategist instance.")
                self.strategist = create_strategist()

            # Generate strategy recommendation
            try:
                strategy_recommendation = self.strategist.generate_strategy(
                    context=strategist_context,
                    metrics={
                        "alignment_score": verification_result["new_score"],
                        "improvement_margin": verification_result["improvement_margin"],
                        "lyapunov_estimate": new_lyapunov_estimate,
                        "stability_improved": stability_improved
                    },
                    constraints={"max_complexity": "medium"}
                )

                logger.info(f"MetaStrategist recommendation: {strategy_recommendation.title}")

                # Check if the strategy includes monitoring parameter adjustments
                if hasattr(self.monitor, 'update_monitoring_focus') and strategy_recommendation.metadata.get("adjust_monitoring_parameters"):
                    logger.info("Applying monitoring parameter adjustments from strategy.")
                    self.monitor.update_monitoring_focus(
                        strategy_recommendation.metadata["adjust_monitoring_parameters"]
                    )
            except Exception as e:
                logger.warning(f"Error consulting MetaStrategist: {e}")
                strategy_recommendation = None

        # Return comprehensive results including stability metrics and strategy recommendation
        result = {
            "intervention_applied": True,
            "improved": verification_result["improved"],
            "previous_score": verification_result["previous_score"],
            "new_score": verification_result["new_score"],
            "improvement_margin": verification_result["improvement_margin"],
            "above_threshold": verification_result["above_threshold"],
            "intervention_targets": intervention_targets,
            "stability_metrics": {
                "before": stability_metrics,
                "after": new_stability_metrics
            },
            "stability_improved": stability_improved,
            "lyapunov_before": lyapunov_estimate,
            "lyapunov_after": new_lyapunov_estimate,
            "auto_stabilize_active": self.auto_stabilize,
            "intervention_reason": intervention_reason,
            "modulated_activation": True
        }

        # Add strategy recommendation if available
        if strategy_recommendation:
            result["strategy_recommendation"] = {
                "title": strategy_recommendation.title,
                "description": strategy_recommendation.description,
                "steps": strategy_recommendation.steps,
                "confidence": strategy_recommendation.confidence,
                "tags": strategy_recommendation.tags
            }

        return result
