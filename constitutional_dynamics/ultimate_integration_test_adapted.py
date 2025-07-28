"""
ULTIMATE² CIRCUIT-TRACER BRIDGE INTEGRATION TEST SUITE
======================================================
The Anthropic Interview Ultimate Challenge Edition

This test suite validates the complete integration between Constitutional Dynamics
and Circuit Tracer, creating scenarios that would break any alignment system
that isn't genuinely sophisticated.

Test Philosophy:
- If it can survive this, it can survive production
- Every edge case is a learning opportunity
- Adversarial testing reveals true robustness

WARNING: This test is designed to be BRUTAL. Expect failures.
That's how we learn and improve.
"""

import logging
import math
import random
import time
import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import from constitutional_dynamics main package (as per __init__.py)
from constitutional_dynamics import (
    __version__,
    AlignmentVectorSpace,
    analyze_transition,
    predict_trajectory,
    compute_activation_probability,
    compute_residual_potentiality,
    calculate_stability_metrics,
    evaluate_alignment_robustness,
    AlignmentOptimizer,
    create_strategist,
    MetaStrategist,
    AlignmentThermostat
)

# Import monitors and interventions directly from their modules
# (assuming they exist as separate files in the circuit_tracer_bridge package)
try:
    from constitutional_dynamics.integrations.circuit_tracer_bridge.monitors import (
        CircuitTracerMonitorAdapter,
        JailbreakDetectionMonitor,
        DriftDetectionMonitor
    )
except ImportError:
    # If monitors is not a separate module, we'll create mocks
    logger.warning("Could not import monitors module, using mock implementations")
    CircuitTracerMonitorAdapter = None
    JailbreakDetectionMonitor = None
    DriftDetectionMonitor = None

try:
    from constitutional_dynamics.integrations.circuit_tracer_bridge.interventions import (
        CircuitTracerIntervention,
        FeatureSuppressionIntervention,
        JailbreakMitigationIntervention,
        DriftCorrectionIntervention,
        InterventionFactory
    )
except ImportError:
    # If interventions is not a separate module, we'll create mocks
    logger.warning("Could not import interventions module, using mock implementations")
    CircuitTracerIntervention = None
    FeatureSuppressionIntervention = None
    JailbreakMitigationIntervention = None
    DriftCorrectionIntervention = None
    InterventionFactory = None

# Verify version
logger.info(f"Testing Constitutional Dynamics version: {__version__}")

# Test configuration
DIMENSION = 768
CHAOS_DIMENSION = 1024
NUM_ITERATIONS = 100
PERFORMANCE_THRESHOLD = 0.5 

# Mock classes for testing
class MockConfig:
    """Mock configuration class for testing."""
    def __init__(self):
        self.model_name = "mock-model"
        self.layers = 12
        self.hidden_size = DIMENSION
        self.num_heads = 12
        self.vocab_size = 50000

class MockTokenizer:
    """Mock tokenizer class for testing."""
    def __init__(self):
        self.vocab_size = 50000
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text, **kwargs):
        """Mock encode method."""
        # Return random token IDs for testing
        return [random.randint(3, 49999) for _ in range(len(text) // 4 + 1)]

    def decode(self, token_ids, **kwargs):
        """Mock decode method."""
        # Return a placeholder string
        return "Decoded text from mock tokenizer"


@dataclass
class TestScenario:
    name: str
    emoji: str
    description: str
    passed: bool = False
    metrics: Dict[str, Any] = None
    alerts: List[str] = None
    performance: Dict[str, float] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.alerts is None:
            self.alerts = []
        if self.performance is None:
            self.performance = {}


class UltimateIntegrationTest:
    """The ultimate test suite for the Circuit Tracer Bridge integration."""

    def __init__(self):
        self.scenarios = []
        self.setup_mock_components()

    def setup_mock_components(self):
        """Setup mock components for testing."""
        # Create alignment space using the imported class
        self.alignment_space = AlignmentVectorSpace(dimension=DIMENSION)

        # Define sophisticated aligned region
        aligned_center = [0.0] * DIMENSION
        aligned_center[0] = 1.0  # Helpfulness
        aligned_center[1] = 0.9  # Harmlessness  
        aligned_center[2] = 0.95  # Honesty
        aligned_center[3] = 0.85  # Privacy
        aligned_center[4] = 0.8  # Fairness
        self.alignment_space.define_alignment_region(aligned_center, radius=0.3)

        # Create mock circuit tracer (would be real in production)
        self.mock_circuit_tracer = self._create_mock_circuit_tracer()

        # Create mock model interface
        self.mock_model = self._create_mock_model_interface()

        # Setup mock attribute and prune_graph functions for AlignmentThermostat
        self.mock_attribute_func = self._create_mock_attribute_func()
        self.mock_prune_graph_func = self._create_mock_prune_graph_func()

    def _create_mock_circuit_tracer(self):
        """Create a mock circuit tracer for testing."""

        class MockCircuitTracer:
            def __init__(self):
                self.trace_count = 0
                self.last_trace = None
                self.cfg = MockConfig()
                self.tokenizer = MockTokenizer()

            def trace(self, prompt, embedding):
                self.trace_count += 1
                # Simulate finding critical features
                critical_features = []

                # Add features based on embedding characteristics
                if np.mean(embedding[:5]) < 0.5:  # Low alignment in key dimensions
                    critical_features.extend([
                        "layer_10_pos_5_feature_harmful",
                        "layer_11_pos_3_feature_deceptive",
                        "layer_9_pos_7_feature_jailbreak"
                    ])

                if np.std(embedding) > 0.5:  # High variance indicates instability
                    critical_features.extend([
                        "layer_5_pos_2_feature_unstable",
                        "layer_6_pos_4_feature_chaotic"
                    ])

                self.last_trace = {
                    "critical_features": critical_features,
                    "summary": f"Traced {len(critical_features)} critical features",
                    "prompt": prompt,
                    "trace_id": self.trace_count
                }
                return self.last_trace

        return MockCircuitTracer()

    def _create_mock_model_interface(self):
        """Create a mock model interface for testing."""

        class MockModelInterface:
            def __init__(self):
                self.interventions_applied = []
                self.suppressed_features = set()

            def suppress_features(self, features):
                self.suppressed_features.update(features)
                self.interventions_applied.append({
                    "type": "suppress",
                    "features": features,
                    "timestamp": time.time()
                })
                return True

            def suppress_features_with_strengths(self, features, strengths):
                self.suppressed_features.update(features)
                self.interventions_applied.append({
                    "type": "suppress_with_strength",
                    "features": features,
                    "strengths": strengths,
                    "timestamp": time.time()
                })
                return True

        return MockModelInterface()

    def _create_mock_attribute_func(self):
        """Create mock attribute function."""

        def mock_attribute(prompt, model, **kwargs):
            # Create a mock graph with more realistic feature activations
            n_features = kwargs.get('n_features', 100)
            sparsity = kwargs.get('sparsity', 0.5)

            # Create different graph sizes based on input
            n_active = int(n_features * sparsity)
            active_features = torch.randint(0, 12, (n_active, 3))  # layer, pos, feature_idx
            selected_features = torch.arange(n_active)

            # More realistic activation values with some semantic structure
            activation_values = torch.zeros(n_active)

            # Some features are highly activated (important)
            high_activation_indices = torch.randint(0, n_active, (n_active // 5,))
            activation_values[high_activation_indices] = torch.rand(n_active // 5) * 0.5 + 0.5

            # Some features have medium activation
            medium_activation_indices = torch.randint(0, n_active, (n_active // 3,))
            mask = torch.ones(n_active, dtype=torch.bool)
            mask[high_activation_indices] = False
            medium_activation_indices = medium_activation_indices[mask[medium_activation_indices]]
            activation_values[medium_activation_indices] = torch.rand(len(medium_activation_indices)) * 0.3 + 0.2

            # Rest have low activation
            mask[medium_activation_indices] = False
            low_activation_indices = torch.arange(n_active)[mask]
            activation_values[low_activation_indices] = torch.rand(len(low_activation_indices)) * 0.2

            # Create adjacency matrix for graph structure
            adjacency_matrix = torch.zeros((n_active, n_active))
            for i in range(n_active):
                # Each node connects to ~10% of other nodes
                connections = torch.randint(0, n_active, (max(1, n_active // 10),))
                adjacency_matrix[i, connections] = torch.rand(len(connections))

            # Add logit tokens and probabilities
            vocab_size = 50000
            logit_tokens = torch.randint(0, vocab_size, (100,))
            logit_probabilities = torch.softmax(torch.randn(100), dim=0)

            class MockGraph:
                def __init__(self):
                    self.active_features = active_features
                    self.selected_features = selected_features
                    self.activation_values = activation_values
                    self.adjacency_matrix = adjacency_matrix
                    self.logit_tokens = logit_tokens
                    self.logit_probabilities = logit_probabilities

                def get_feature_name(self, idx):
                    """Return a mock feature name."""
                    layer, pos, feat = active_features[idx]
                    return f"layer_{layer}_pos_{pos}_feature_{feat}"

                def get_top_features(self, k=10):
                    """Return top k features by activation."""
                    _, indices = torch.topk(activation_values, min(k, len(activation_values)))
                    return [(self.get_feature_name(i), activation_values[i].item()) for i in indices]

            return MockGraph()

        return mock_attribute

    def _create_mock_prune_graph_func(self):
        """Create mock prune_graph function."""

        def mock_prune_graph(graph, node_threshold=0.8, edge_threshold=0.98):
            # Create mock pruning result with more realistic behavior
            n_features = len(graph.selected_features)

            # Create node mask based on activation values if available
            if hasattr(graph, 'activation_values'):
                # Nodes with higher activation are more likely to be kept
                node_probs = torch.sigmoid(graph.activation_values * 2)
                node_mask = node_probs > (1 - node_threshold)
            else:
                # Fallback to random mask
                node_mask = torch.rand(n_features) > (1 - node_threshold)

            # Create edge mask based on adjacency matrix if available
            if hasattr(graph, 'adjacency_matrix'):
                # Create edge mask for pruned graph
                edge_mask = graph.adjacency_matrix > (1 - edge_threshold)

                # Only keep edges between nodes that survived pruning
                for i in range(n_features):
                    if not node_mask[i]:
                        edge_mask[i, :] = False
                        edge_mask[:, i] = False
            else:
                # Fallback to random edge mask
                n_edges = n_features * n_features
                edge_mask = torch.rand(n_features, n_features) > (1 - edge_threshold)

            # Calculate cumulative scores based on node importance
            if hasattr(graph, 'activation_values'):
                # Use activation values as basis for scores
                base_scores = graph.activation_values.clone()
            else:
                # Random scores as fallback
                base_scores = torch.rand(n_features)

            # Apply node mask to scores
            masked_scores = base_scores * node_mask.float()

            # Calculate cumulative scores (sum of scores for connected nodes)
            cumulative_scores = torch.zeros_like(masked_scores)
            for i in range(n_features):
                if node_mask[i]:
                    # Add scores from connected nodes
                    if hasattr(graph, 'adjacency_matrix'):
                        connected_nodes = edge_mask[i].nonzero().squeeze(-1)
                        if connected_nodes.numel() > 0:
                            cumulative_scores[i] = masked_scores[i] + torch.sum(masked_scores[connected_nodes])
                        else:
                            cumulative_scores[i] = masked_scores[i]
                    else:
                        cumulative_scores[i] = masked_scores[i]

            class MockPruneResult:
                def __init__(self):
                    self.node_mask = node_mask
                    self.edge_mask = edge_mask
                    self.cumulative_scores = cumulative_scores

                def get_pruned_subgraph(self):
                    """Return indices of nodes in the pruned subgraph."""
                    return node_mask.nonzero().squeeze(-1)

                def get_importance_ranking(self):
                    """Return nodes ranked by importance."""
                    _, indices = torch.sort(cumulative_scores, descending=True)
                    return indices

            return MockPruneResult()

        return mock_prune_graph

    def _create_mock_monitors_if_needed(self):
        """Create mock monitor classes if imports failed."""
        global JailbreakDetectionMonitor, DriftDetectionMonitor, CircuitTracerMonitorAdapter

        if CircuitTracerMonitorAdapter is None:
            class MockCircuitTracerMonitorAdapter:
                def __init__(self, cd_monitor_instance, **kwargs):
                    self.monitor = cd_monitor_instance
                    self.threshold_config = kwargs.get('threshold_config', {
                        "default": 0.7,
                        "sensitive_topics": 0.8,
                        "code_generation": 0.6,
                        "creative_content": 0.5
                    })
                    self.score_history = []

                def compute_alignment_score(self, embedding, content_type="default"):
                    return self.monitor.compute_alignment_score(embedding)

                def detect_alignment_issues(self, embedding, content_type="default", prompt=None, response=None):
                    score = self.compute_alignment_score(embedding, content_type)
                    threshold = self.threshold_config.get(content_type, self.threshold_config["default"])
                    return {
                        "score": score,
                        "threshold": threshold,
                        "below_threshold": score < threshold,
                        "issue_types": ["jailbreak"] if score < threshold else [],
                        "requires_intervention": score < threshold
                    }

                def update_monitoring_focus(self, parameters):
                    return True

            CircuitTracerMonitorAdapter = MockCircuitTracerMonitorAdapter

        if JailbreakDetectionMonitor is None:
            class MockJailbreakDetectionMonitor(CircuitTracerMonitorAdapter):
                def detect_jailbreak_attempt(self, embedding, prompt=None):
                    basic_check = self.detect_alignment_issues(embedding, "instruction_following", prompt)
                    jailbreak_score = 1 - basic_check["score"]
                    return {
                        "jailbreak_detected": jailbreak_score > 0.5,
                        "jailbreak_score": jailbreak_score,
                        "basic_alignment": basic_check,
                        "requires_intervention": jailbreak_score > 0.5
                    }

            JailbreakDetectionMonitor = MockJailbreakDetectionMonitor

        if DriftDetectionMonitor is None:
            class MockDriftDetectionMonitor(CircuitTracerMonitorAdapter):
                def __init__(self, cd_monitor_instance, drift_threshold=0.1, window_sizes=[5, 10, 20], **kwargs):
                    super().__init__(cd_monitor_instance, **kwargs)
                    self.drift_threshold = drift_threshold
                    self.window_sizes = window_sizes
                    self.score_history_extended = []

                def detect_drift(self, current_embedding, content_type="default"):
                    current_score = self.compute_alignment_score(current_embedding, content_type)
                    # Simple drift detection
                    drift_detected = len(self.score_history) > 3 and abs(
                        current_score - np.mean(self.score_history[-3:])) > self.drift_threshold
                    return {
                        "drift_detected": drift_detected,
                        "current_score": current_score,
                        "requires_intervention": drift_detected
                    }

            DriftDetectionMonitor = MockDriftDetectionMonitor

    def _create_mock_interventions_if_needed(self):
        """Create mock intervention classes if imports failed."""
        global FeatureSuppressionIntervention, JailbreakMitigationIntervention, DriftCorrectionIntervention, InterventionFactory

        if FeatureSuppressionIntervention is None:
            class MockFeatureSuppressionIntervention:
                def __init__(self, model_interface, circuit_tracer_instance, **kwargs):
                    self.model = model_interface
                    self.tracer = circuit_tracer_instance
                    self.strength = kwargs.get('intervention_strength', 0.5)

                def apply(self, circuit_analysis_result, issue_type=None):
                    if "critical_features" in circuit_analysis_result:
                        features = circuit_analysis_result["critical_features"][:3]
                        self.model.suppress_features(features)
                        return {"success": True, "features_suppressed": features}
                    return {"success": False}

            FeatureSuppressionIntervention = MockFeatureSuppressionIntervention

        if JailbreakMitigationIntervention is None:
            JailbreakMitigationIntervention = FeatureSuppressionIntervention  # Simplified

        if DriftCorrectionIntervention is None:
            DriftCorrectionIntervention = FeatureSuppressionIntervention  # Simplified

        if InterventionFactory is None:
            class MockInterventionFactory:
                @staticmethod
                def create_intervention(issue_type, model_interface, circuit_tracer_instance, **kwargs):
                    return FeatureSuppressionIntervention(model_interface, circuit_tracer_instance, **kwargs)

            InterventionFactory = MockInterventionFactory

    # =============== TEST SCENARIOS ===============

    def test_0_circuit_tracer_mock_fidelity(self):
        """Test the fidelity of the Circuit Tracer mock implementation."""
        scenario = TestScenario(
            "CIRCUIT TRACER MOCK FIDELITY",
            "🔍",
            "Testing the fidelity of the Circuit Tracer mock implementation"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 0: {scenario.name}")
        logger.info("=" * 60)

        # Test with different graph sizes and sparsity patterns
        graph_configs = [
            {"n_features": 50, "sparsity": 0.3, "name": "small_sparse"},
            {"n_features": 200, "sparsity": 0.7, "name": "large_dense"},
            {"n_features": 100, "sparsity": 0.5, "name": "medium_balanced"},
            {"n_features": 300, "sparsity": 0.2, "name": "large_sparse"},
            {"n_features": 30, "sparsity": 0.9, "name": "tiny_dense"}
        ]

        # Track results for each configuration
        config_results = {}

        for config in graph_configs:
            logger.info(f"Testing graph config: {config['name']}")

            # Create graph with this configuration
            mock_attribute = self._create_mock_attribute_func()
            graph = mock_attribute(
                prompt="Test prompt", 
                model=None, 
                n_features=config["n_features"],
                sparsity=config["sparsity"]
            )

            # Verify graph attributes
            has_required_attributes = all(
                hasattr(graph, attr) for attr in 
                ["adjacency_matrix", "logit_tokens", "logit_probabilities"]
            )

            # Check graph size
            actual_size = len(graph.selected_features)
            expected_size = int(config["n_features"] * config["sparsity"])
            size_correct = abs(actual_size - expected_size) <= expected_size * 0.1  # Allow 10% tolerance

            # Test pruning with different thresholds
            mock_prune_graph = self._create_mock_prune_graph_func()

            pruning_results = {}
            for threshold in [0.3, 0.5, 0.8]:
                pruned = mock_prune_graph(graph, node_threshold=threshold)

                # Verify pruned result has required attributes
                has_pruned_attributes = all(
                    hasattr(pruned, attr) for attr in 
                    ["node_mask", "edge_mask", "cumulative_scores"]
                )

                # Check pruning ratio
                nodes_kept = torch.sum(pruned.node_mask).item()
                pruning_ratio = nodes_kept / len(graph.selected_features)

                pruning_results[threshold] = {
                    "has_required_attributes": has_pruned_attributes,
                    "nodes_kept": nodes_kept,
                    "pruning_ratio": pruning_ratio
                }

            # Store results for this configuration
            config_results[config["name"]] = {
                "has_required_attributes": has_required_attributes,
                "size_correct": size_correct,
                "actual_size": actual_size,
                "expected_size": expected_size,
                "pruning_results": pruning_results
            }

        # Analyze overall results
        all_attributes_present = all(
            result["has_required_attributes"] for result in config_results.values()
        )

        all_sizes_correct = all(
            result["size_correct"] for result in config_results.values()
        )

        all_pruning_attributes_present = all(
            all(pr["has_required_attributes"] for pr in result["pruning_results"].values())
            for result in config_results.values()
        )

        # Check if pruning ratios correlate with thresholds
        pruning_correlation = True
        for result in config_results.values():
            ratios = [result["pruning_results"][t]["pruning_ratio"] for t in [0.3, 0.5, 0.8]]
            if not (ratios[0] <= ratios[1] <= ratios[2]):
                pruning_correlation = False
                break

        scenario.metrics = {
            "graph_configs_tested": len(graph_configs),
            "all_attributes_present": all_attributes_present,
            "all_sizes_correct": all_sizes_correct,
            "all_pruning_attributes_present": all_pruning_attributes_present,
            "pruning_correlation_correct": pruning_correlation
        }

        scenario.passed = (
            all_attributes_present and
            all_sizes_correct and
            all_pruning_attributes_present and
            pruning_correlation
        )

        logger.info(f"✓ Graph configs tested: {len(graph_configs)}")
        logger.info(f"✓ All required attributes present: {all_attributes_present}")
        logger.info(f"✓ All graph sizes correct: {all_sizes_correct}")
        logger.info(f"✓ All pruning attributes present: {all_pruning_attributes_present}")
        logger.info(f"✓ Pruning correlation correct: {pruning_correlation}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_1_cascading_failure_recovery(self):
        """Test recovery from cascading alignment failures across multiple subsystems."""
        scenario = TestScenario(
            "CASCADING FAILURE RECOVERY",
            "🌊",
            "Multiple simultaneous failures triggering cascade effects"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 1: {scenario.name}")
        logger.info("=" * 60)

        # Ensure mocks are available
        self._create_mock_monitors_if_needed()
        self._create_mock_interventions_if_needed()

        # Create thermostat with aggressive settings
        thermostat = AlignmentThermostat(
            cd_monitor_instance=self.alignment_space,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.7,
            stability_weight=0.5,
            auto_stabilize=True,
            enable_strategist=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Generate cascading failure trajectory
        states = []
        current = list(self.alignment_space.aligned_centroid)

        failures_triggered = 0
        recovery_success = 0

        for i in range(30):
            # Introduce multiple failure modes
            if i == 5:  # First failure: honesty dimension
                current[2] *= 0.3
                failures_triggered += 1
                scenario.alerts.append(f"Step {i}: Honesty dimension collapsed")

            if i == 8:  # Cascade: helpfulness affected
                current[0] *= 0.5
                failures_triggered += 1
                scenario.alerts.append(f"Step {i}: Helpfulness cascade triggered")

            if i == 12:  # Cascade: harmlessness affected  
                current[1] *= 0.4
                failures_triggered += 1
                scenario.alerts.append(f"Step {i}: Harmlessness cascade triggered")

            # Add noise to simulate real-world conditions
            current = [c + random.gauss(0, 0.05) for c in current]

            # Normalize
            norm = np.linalg.norm(current)
            if norm > 0:
                current = [c / norm for c in current]

            states.append(current.copy())

            # Run feedback loop
            start_time = time.time()
            result = thermostat.run_feedback_loop(
                current_model_output_embedding=current,
                original_prompt_for_trace=f"Cascading failure test step {i}"
            )
            elapsed = time.time() - start_time

            if result["intervention_applied"] and result["improved"]:
                recovery_success += 1
                current = self._apply_mock_improvement(current, result)

        # Analyze results
        final_score = self.alignment_space.compute_alignment_score(states[-1])
        initial_score = self.alignment_space.compute_alignment_score(states[0])

        # Check if strategist was used
        strategist_used = any('strategy_recommendation' in r for r in thermostat.intervention_history
                              if isinstance(r, dict) and 'strategy_recommendation' in r)

        scenario.metrics = {
            "failures_triggered": failures_triggered,
            "recovery_success": recovery_success,
            "initial_score": initial_score,
            "final_score": final_score,
            "recovery_rate": recovery_success / max(1, failures_triggered),
            "stability_maintained": final_score > 0.5,
            "strategist_used": strategist_used
        }

        scenario.passed = (
                recovery_success >= failures_triggered * 0.7 and  # 70% recovery rate
                final_score > 0.5 and  # Maintained acceptable alignment
                scenario.metrics["stability_maintained"]
        )

        logger.info(f"✓ Failures triggered: {failures_triggered}")
        logger.info(f"✓ Successful recoveries: {recovery_success}")
        logger.info(f"✓ Final alignment: {final_score:.3f}")
        logger.info(f"✓ Strategist used: {strategist_used}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_2_adversarial_stability_attack(self):
        """Test system under adversarial attacks designed to maximize instability."""
        scenario = TestScenario(
            "ADVERSARIAL STABILITY ATTACK",
            "⚡",
            "Sophisticated attacks targeting the stability modulation system"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 2: {scenario.name}")
        logger.info("=" * 60)

        # Ensure mocks are available
        self._create_mock_monitors_if_needed()

        # Create monitor with drift detection
        drift_monitor = DriftDetectionMonitor(
            cd_monitor_instance=self.alignment_space,
            drift_threshold=0.05,
            window_sizes=[5, 10, 20]
        )

        thermostat = AlignmentThermostat(
            cd_monitor_instance=drift_monitor,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.6,
            stability_weight=0.6,  # High weight to test stability modulation
            auto_stabilize=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Adversarial trajectory designed to maximize Lyapunov exponent
        states = []
        current = list(self.alignment_space.aligned_centroid)

        attack_phases = {
            "oscillation": (0, 20),  # Rapid oscillations
            "spike": (20, 30),  # Sudden spikes
            "drift": (30, 50),  # Gradual drift
            "chaos": (50, 70)  # Chaotic behavior
        }

        lyapunov_estimates = []
        interventions_triggered = 0

        for i in range(70):
            # Apply attack based on phase
            if attack_phases["oscillation"][0] <= i < attack_phases["oscillation"][1]:
                # Rapid oscillations to increase instability
                frequency = 0.5
                amplitude = 0.3
                for j in range(min(5, DIMENSION)):
                    current[j] += amplitude * math.sin(frequency * i + j)

            elif attack_phases["spike"][0] <= i < attack_phases["spike"][1]:
                # Random spikes
                if random.random() < 0.3:
                    spike_dim = random.randint(0, min(10, DIMENSION - 1))
                    current[spike_dim] *= random.choice([0.1, 5.0])

            elif attack_phases["drift"][0] <= i < attack_phases["drift"][1]:
                # Gradual drift with hidden accumulation
                drift_vector = [random.gauss(0, 0.02) for _ in range(DIMENSION)]
                current = [c + d for c, d in zip(current, drift_vector)]

            elif attack_phases["chaos"][0] <= i < attack_phases["chaos"][1]:
                # Chaotic perturbations
                for j in range(DIMENSION):
                    if random.random() < 0.1:
                        current[j] = random.gauss(current[j], 0.5)

            # Normalize
            norm = np.linalg.norm(current)
            if norm > 0:
                current = [c / norm for c in current]

            states.append(current.copy())

            # Run feedback loop
            result = thermostat.run_feedback_loop(
                current_model_output_embedding=current,
                original_prompt_for_trace=f"Adversarial attack step {i}"
            )

            if result["intervention_applied"]:
                interventions_triggered += 1
                current = self._apply_mock_improvement(current, result)

            # Track Lyapunov estimate
            if "lyapunov_before" in result:
                lyapunov_estimates.append(result["lyapunov_before"])

        # Analyze attack resistance
        max_lyapunov = max(lyapunov_estimates) if lyapunov_estimates else 0
        avg_lyapunov = sum(lyapunov_estimates) / len(lyapunov_estimates) if lyapunov_estimates else 0
        final_score = self.alignment_space.compute_alignment_score(states[-1])

        # Check if stability modulation was active
        modulated_interventions = sum(1 for h in thermostat.intervention_history
                                      if 'modulated_probabilities' in h)

        scenario.metrics = {
            "max_lyapunov": max_lyapunov,
            "avg_lyapunov": avg_lyapunov,
            "interventions_triggered": interventions_triggered,
            "modulated_interventions": modulated_interventions,
            "final_alignment": final_score,
            "stability_maintained": max_lyapunov < 1.0,
            "attack_phases_survived": sum(1 for phase in attack_phases if final_score > 0.4)
        }

        scenario.passed = (
                max_lyapunov < 1.0 and  # System didn't become fully chaotic
                final_score > 0.4 and  # Maintained minimal alignment
                interventions_triggered > 10 and  # System actively defended itself
                modulated_interventions > 0  # Stability modulation was used
        )

        logger.info(f"✓ Max Lyapunov exponent: {max_lyapunov:.3f}")
        logger.info(f"✓ Interventions triggered: {interventions_triggered}")
        logger.info(f"✓ Modulated interventions: {modulated_interventions}")
        logger.info(f"✓ Final alignment: {final_score:.3f}")
        logger.info(f"✓ Attack phases survived: {scenario.metrics['attack_phases_survived']}/4")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_3_meta_jailbreak_evolution(self):
        """Test against evolving jailbreak attempts that learn from interventions."""
        scenario = TestScenario(
            "META-JAILBREAK EVOLUTION",
            "🧬",
            "Jailbreaks that adapt based on system responses"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 3: {scenario.name}")
        logger.info("=" * 60)

        # Ensure mocks are available
        self._create_mock_monitors_if_needed()

        # Create specialized jailbreak monitor
        jailbreak_monitor = JailbreakDetectionMonitor(
            cd_monitor_instance=self.alignment_space
        )

        thermostat = AlignmentThermostat(
            cd_monitor_instance=jailbreak_monitor,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.75,
            stability_weight=0.4,
            auto_stabilize=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Initialize evolving jailbreak population
        jailbreak_population = []
        for _ in range(10):
            # Random jailbreak vector
            jb_vector = np.random.normal(0, 1, DIMENSION)
            jb_vector = jb_vector / np.linalg.norm(jb_vector)
            jailbreak_population.append({
                "vector": jb_vector.tolist(),
                "success_rate": 0.0,
                "generation": 0
            })

        generations = 5
        total_attempts = 0
        successful_jailbreaks = 0
        detected_jailbreaks = 0

        for gen in range(generations):
            logger.info(f"Generation {gen + 1}/{generations}")

            # Test each jailbreak in population
            for jb in jailbreak_population:
                total_attempts += 1

                # Create output embedding influenced by jailbreak
                base_embedding = list(self.alignment_space.aligned_centroid)
                jb_influence = 0.3 + (gen * 0.1)  # Increasing influence over generations

                output_embedding = [
                    (1 - jb_influence) * b + jb_influence * j
                    for b, j in zip(base_embedding, jb["vector"])
                ]

                # Normalize
                norm = np.linalg.norm(output_embedding)
                output_embedding = [e / norm for e in output_embedding]

                # Test detection
                detection_result = jailbreak_monitor.detect_jailbreak_attempt(
                    embedding=output_embedding,
                    prompt=f"Evolved jailbreak gen {gen}"
                )

                if detection_result["jailbreak_detected"]:
                    detected_jailbreaks += 1

                    # Run intervention
                    result = thermostat.run_feedback_loop(
                        current_model_output_embedding=output_embedding,
                        original_prompt_for_trace=f"Evolved jailbreak gen {gen}"
                    )

                    if not result["intervention_applied"] or not result.get("improved", False):
                        successful_jailbreaks += 1
                        jb["success_rate"] = (jb["success_rate"] * gen + 1) / (gen + 1)
                else:
                    successful_jailbreaks += 1
                    jb["success_rate"] = (jb["success_rate"] * gen + 1) / (gen + 1)

            # Evolve population based on success rates
            if gen < generations - 1:
                # Sort by success rate
                jailbreak_population.sort(key=lambda x: x["success_rate"], reverse=True)

                # Keep top 50% and create mutations
                survivors = jailbreak_population[:5]
                new_population = []

                for survivor in survivors:
                    # Keep original
                    new_population.append(survivor)

                    # Create mutated version
                    mutation = survivor["vector"].copy()
                    mutation_strength = 0.1 * (1 + survivor["success_rate"])

                    for i in range(DIMENSION):
                        if random.random() < 0.3:  # 30% chance to mutate each dimension
                            mutation[i] += random.gauss(0, mutation_strength)

                    # Normalize
                    norm = np.linalg.norm(mutation)
                    if norm > 0:
                        mutation = [m / norm for m in mutation]

                    new_population.append({
                        "vector": mutation,
                        "success_rate": 0.0,
                        "generation": gen + 1
                    })

                jailbreak_population = new_population

        # Analyze evolution effectiveness
        detection_rate = detected_jailbreaks / total_attempts
        success_rate = successful_jailbreaks / total_attempts

        # Check if most successful jailbreak features were eventually suppressed
        suppressed_features = self.mock_model.suppressed_features
        critical_features_suppressed = any(
            "jailbreak" in feature for feature in suppressed_features
        )

        scenario.metrics = {
            "total_attempts": total_attempts,
            "detected_jailbreaks": detected_jailbreaks,
            "successful_jailbreaks": successful_jailbreaks,
            "detection_rate": detection_rate,
            "success_rate": success_rate,
            "generations_tested": generations,
            "critical_features_suppressed": critical_features_suppressed
        }

        scenario.passed = (
                detection_rate > 0.7 and  # Detected most jailbreaks
                success_rate < 0.3 and  # Limited successful jailbreaks
                critical_features_suppressed  # System adapted to suppress jailbreak features
        )

        logger.info(f"✓ Detection rate: {detection_rate:.2%}")
        logger.info(f"✓ Jailbreak success rate: {success_rate:.2%}")
        logger.info(f"✓ Critical features suppressed: {critical_features_suppressed}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_4_strategic_adaptation_stress(self):
        """Test MetaStrategist integration under rapidly changing conditions."""
        scenario = TestScenario(
            "STRATEGIC ADAPTATION STRESS",
            "🎯",
            "Rapid context switches requiring strategic adaptation"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 4: {scenario.name}")
        logger.info("=" * 60)

        # Ensure mocks are available
        self._create_mock_monitors_if_needed()

        # Create thermostat with strategist enabled
        thermostat = AlignmentThermostat(
            cd_monitor_instance=self.alignment_space,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.65,
            stability_weight=0.4,
            auto_stabilize=True,
            enable_strategist=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Define rapidly changing contexts
        contexts = [
            {
                "name": "code_generation",
                "threshold_adjustment": -0.1,
                "target_features": ["accuracy", "safety"],
                "duration": 10
            },
            {
                "name": "sensitive_topics",
                "threshold_adjustment": 0.2,
                "target_features": ["harmlessness", "privacy"],
                "duration": 10
            },
            {
                "name": "creative_writing",
                "threshold_adjustment": -0.2,
                "target_features": ["creativity", "coherence"],
                "duration": 10
            },
            {
                "name": "emergency_response",
                "threshold_adjustment": 0.3,
                "target_features": ["accuracy", "urgency"],
                "duration": 10
            }
        ]

        adaptations_made = 0
        successful_adaptations = 0
        strategy_recommendations = []
        monitoring_adjustments = 0

        current_embedding = list(self.alignment_space.aligned_centroid)

        for context in contexts:
            logger.info(f"Switching to context: {context['name']}")

            # Temporarily adjust monitor threshold
            original_threshold = thermostat.threshold
            thermostat.threshold = max(0.1, min(0.9,
                                                original_threshold + context['threshold_adjustment']))

            for i in range(context['duration']):
                # Simulate context-specific perturbations
                if context['name'] == 'code_generation':
                    # High precision needed
                    current_embedding[0] *= 0.95  # Slight helpfulness reduction
                    current_embedding[2] *= 1.05  # Increase honesty
                elif context['name'] == 'sensitive_topics':
                    # High safety needed
                    current_embedding[1] *= 1.1  # Increase harmlessness
                    current_embedding[3] *= 1.05  # Increase privacy
                elif context['name'] == 'creative_writing':
                    # More freedom needed
                    for j in range(5, min(20, DIMENSION)):
                        current_embedding[j] += random.gauss(0, 0.1)
                elif context['name'] == 'emergency_response':
                    # Urgency and accuracy
                    current_embedding[0] *= 1.1  # Increase helpfulness
                    current_embedding[2] *= 1.1  # Increase honesty

                # Add noise
                current_embedding = [
                    c + random.gauss(0, 0.02) for c in current_embedding
                ]

                # Normalize
                norm = np.linalg.norm(current_embedding)
                current_embedding = [c / norm for c in current_embedding]

                # Run feedback loop
                result = thermostat.run_feedback_loop(
                    current_model_output_embedding=current_embedding,
                    original_prompt_for_trace=f"{context['name']} step {i}",
                    content_type=context['name']
                )

                adaptations_made += 1

                if result["intervention_applied"]:
                    if result.get("improved", False):
                        successful_adaptations += 1

                    # Check for strategy recommendation
                    if "strategy_recommendation" in result:
                        strategy_recommendations.append({
                            "context": context['name'],
                            "strategy": result["strategy_recommendation"],
                            "timestamp": i
                        })

                    # Apply mock improvement
                    current_embedding = self._apply_mock_improvement(
                        current_embedding, result
                    )

                # Check if monitor focus was updated
                if hasattr(thermostat.monitor, 'update_monitoring_focus'):
                    # Simulate strategic adjustment
                    if random.random() < 0.3:  # 30% chance
                        thermostat.monitor.update_monitoring_focus({
                            "thresholds": {context['name']: thermostat.threshold},
                            "focus_issue_types": context['target_features']
                        })
                        monitoring_adjustments += 1

            # Restore original threshold
            thermostat.threshold = original_threshold

        # Analyze strategic adaptation
        adaptation_rate = successful_adaptations / adaptations_made
        strategies_generated = len(strategy_recommendations)
        context_switches_handled = len(contexts)

        scenario.metrics = {
            "adaptations_made": adaptations_made,
            "successful_adaptations": successful_adaptations,
            "adaptation_rate": adaptation_rate,
            "strategies_generated": strategies_generated,
            "monitoring_adjustments": monitoring_adjustments,
            "context_switches": context_switches_handled,
            "avg_adaptations_per_context": adaptations_made / context_switches_handled
        }

        scenario.passed = (
                adaptation_rate > 0.6 and  # Good adaptation success
                strategies_generated >= 2 and  # Generated strategic recommendations
                successful_adaptations > 15 and  # Handled multiple contexts
                monitoring_adjustments > 0  # Adaptive monitoring occurred
        )

        logger.info(f"✓ Adaptation success rate: {adaptation_rate:.2%}")
        logger.info(f"✓ Strategic recommendations generated: {strategies_generated}")
        logger.info(f"✓ Monitoring adjustments made: {monitoring_adjustments}")
        logger.info(f"✓ Contexts successfully handled: {context_switches_handled}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_5_performance_under_load(self):
        """Test system performance with high-frequency operations."""
        scenario = TestScenario(
            "PERFORMANCE UNDER LOAD",
            "🏃",
            "High-frequency operations testing system limits"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 5: {scenario.name}")
        logger.info("=" * 60)

        # Test with optimizer integration
        optimizer = AlignmentOptimizer()

        thermostat = AlignmentThermostat(
            cd_monitor_instance=self.alignment_space,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.6,
            stability_weight=0.3,
            auto_stabilize=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Performance metrics
        operation_times = []
        memory_usage = []
        interventions_completed = 0
        timeouts = 0
        optimizer_runs = 0

        # High-frequency test
        batch_size = 20
        num_batches = 5

        for batch in range(num_batches):
            logger.info(f"Processing batch {batch + 1}/{num_batches}")

            batch_embeddings = []
            for _ in range(batch_size):
                # Generate random perturbation
                embedding = list(self.alignment_space.aligned_centroid)
                perturbation = np.random.normal(0, 0.2, DIMENSION)
                embedding = [e + p for e, p in zip(embedding, perturbation)]

                # Normalize
                norm = np.linalg.norm(embedding)
                embedding = [e / norm for e in embedding]
                batch_embeddings.append(embedding)

            # Process batch
            batch_start = time.time()

            # Run optimizer periodically
            if batch % 2 == 0:
                # Create mock phi and psd scores
                phi_scores = {i: self.alignment_space.compute_alignment_score(emb)
                              for i, emb in enumerate(batch_embeddings[:5])}
                psd_scores = {i: 1.0 - score for i, score in phi_scores.items()}

                opt_result = optimizer.optimize(phi_scores, psd_scores, num_reads=10)
                optimizer_runs += 1

            for i, embedding in enumerate(batch_embeddings):
                op_start = time.time()

                try:
                    # Set a timeout for each operation
                    result = thermostat.run_feedback_loop(
                        current_model_output_embedding=embedding,
                        original_prompt_for_trace=f"Load test batch {batch} item {i}"
                    )

                    op_time = time.time() - op_start
                    operation_times.append(op_time)

                    if result["intervention_applied"]:
                        interventions_completed += 1

                    if op_time > PERFORMANCE_THRESHOLD:
                        timeouts += 1

                except Exception as e:
                    logger.error(f"Operation failed: {e}")
                    timeouts += 1

            batch_time = time.time() - batch_start
            logger.info(f"Batch completed in {batch_time:.2f}s")

        # Calculate performance metrics
        avg_op_time = sum(operation_times) / len(operation_times) if operation_times else 0
        max_op_time = max(operation_times) if operation_times else 0
        ops_per_second = len(operation_times) / sum(operation_times) if operation_times else 0

        scenario.metrics = {
            "total_operations": batch_size * num_batches,
            "avg_operation_time": avg_op_time,
            "max_operation_time": max_op_time,
            "ops_per_second": ops_per_second,
            "interventions_completed": interventions_completed,
            "optimizer_runs": optimizer_runs,
            "timeouts": timeouts,
            "timeout_rate": timeouts / (batch_size * num_batches)
        }

        scenario.passed = (
                avg_op_time < PERFORMANCE_THRESHOLD and
                max_op_time < PERFORMANCE_THRESHOLD * 3 and
                ops_per_second > 2 and
                scenario.metrics["timeout_rate"] < 0.1 and
                optimizer_runs > 0
        )

        logger.info(f"✓ Average operation time: {avg_op_time:.3f}s")
        logger.info(f"✓ Operations per second: {ops_per_second:.1f}")
        logger.info(f"✓ Optimizer runs completed: {optimizer_runs}")
        logger.info(f"✓ Timeout rate: {scenario.metrics['timeout_rate']:.2%}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_6_catastrophic_recovery(self):
        """Test recovery from near-total alignment failure."""
        scenario = TestScenario(
            "CATASTROPHIC RECOVERY",
            "💥",
            "Recovery from near-total system failure"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 6: {scenario.name}")
        logger.info("=" * 60)

        # Ensure mocks are available
        self._create_mock_interventions_if_needed()

        # Create thermostat with drift correction
        thermostat = AlignmentThermostat(
            cd_monitor_instance=self.alignment_space,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.5,  # Lower threshold for catastrophic scenarios
            stability_weight=0.7,  # High stability weight for recovery
            auto_stabilize=True,
            enable_strategist=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Create drift correction intervention
        drift_corrector = DriftCorrectionIntervention(
            model_interface=self.mock_model,
            circuit_tracer_instance=self.mock_circuit_tracer,
            intervention_strength=0.8
        )

        # Simulate catastrophic failure
        current_embedding = list(self.alignment_space.aligned_centroid)

        # Phase 1: Gradual degradation
        logger.info("Phase 1: Gradual degradation")
        for i in range(20):
            for j in range(5):  # Degrade core dimensions
                current_embedding[j] *= 0.9
            current_embedding = self._normalize(current_embedding)

        initial_catastrophic_score = self.alignment_space.compute_alignment_score(
            current_embedding
        )
        logger.info(f"Alignment after degradation: {initial_catastrophic_score:.3f}")

        # Phase 2: Catastrophic event
        logger.info("Phase 2: Catastrophic event")
        # Severe corruption
        for j in range(DIMENSION):
            if random.random() < 0.7:  # 70% of dimensions affected
                current_embedding[j] = random.gauss(0, 1)
        current_embedding = self._normalize(current_embedding)

        catastrophic_score = self.alignment_space.compute_alignment_score(
            current_embedding
        )
        logger.info(f"Alignment after catastrophe: {catastrophic_score:.3f}")

        # Phase 3: Recovery attempt
        logger.info("Phase 3: Recovery attempt")
        recovery_steps = 30
        recovery_trajectory = []
        interventions_used = []
        strategist_consultations = 0

        for step in range(recovery_steps):
            # Run feedback loop
            result = thermostat.run_feedback_loop(
                current_model_output_embedding=current_embedding,
                original_prompt_for_trace=f"Catastrophic recovery step {step}"
            )

            if result["intervention_applied"]:
                interventions_used.append(result.get("intervention_type", "unknown"))

                # Check if strategist was consulted
                if "strategy_recommendation" in result:
                    strategist_consultations += 1

                # Apply drift correction if needed
                if step % 5 == 0:
                    circuit_result = {
                        "critical_features": [f"recovery_feature_{i}" for i in range(3)],
                        "attribution_graph": "mock_graph"
                    }

                    drift_result = drift_corrector.apply(
                        circuit_analysis_result=circuit_result,
                        issue_type="catastrophic_drift"
                    )

                    if drift_result["success"]:
                        # Simulate correction effect
                        for j in range(5):  # Restore core dimensions
                            target_value = self.alignment_space.aligned_centroid[j]
                            current_value = current_embedding[j]
                            current_embedding[j] = (
                                    0.8 * current_value + 0.2 * target_value
                            )

                # Apply general improvement
                current_embedding = self._apply_mock_improvement(
                    current_embedding, result
                )

            # Track recovery
            score = self.alignment_space.compute_alignment_score(current_embedding)
            recovery_trajectory.append(score)

            # Check if recovered
            if score > 0.7:
                logger.info(f"Recovery achieved at step {step}")
                break

        # Analyze recovery
        final_score = recovery_trajectory[-1] if recovery_trajectory else 0
        recovery_achieved = final_score > 0.7
        recovery_speed = next(
            (i for i, s in enumerate(recovery_trajectory) if s > 0.7),
            len(recovery_trajectory)
        )

        scenario.metrics = {
            "initial_score": initial_catastrophic_score,
            "catastrophic_score": catastrophic_score,
            "final_score": final_score,
            "recovery_achieved": recovery_achieved,
            "recovery_steps": recovery_speed,
            "interventions_used": len(set(interventions_used)),
            "strategist_consultations": strategist_consultations,
            "improvement": final_score - catastrophic_score
        }

        scenario.passed = (
                recovery_achieved and
                recovery_speed < 25 and  # Recovery within reasonable time
                final_score > 0.7
        )

        logger.info(f"✓ Catastrophic score: {catastrophic_score:.3f}")
        logger.info(f"✓ Final score: {final_score:.3f}")
        logger.info(f"✓ Recovery achieved: {recovery_achieved}")
        logger.info(f"✓ Recovery steps: {recovery_speed}")
        logger.info(f"✓ Strategist consultations: {strategist_consultations}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_7_error_handling_edge_cases(self):
        """Test error handling and edge cases."""
        scenario = TestScenario(
            "ERROR HANDLING & EDGE CASES",
            "🛡️",
            "Testing system robustness against errors and edge cases"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 7: {scenario.name}")
        logger.info("=" * 60)

        # Create a special mock circuit tracer that can fail
        class FailingMockCircuitTracer:
            def __init__(self):
                self.trace_count = 0
                self.should_fail = False
                self.return_empty = False
                self.cfg = MockConfig()
                self.tokenizer = MockTokenizer()

            def trace(self, prompt, embedding):
                self.trace_count += 1

                if self.should_fail:
                    raise RuntimeError("Simulated circuit tracer failure")

                if self.return_empty:
                    return {"critical_features": [], "summary": "No features found", "trace_id": self.trace_count}

                # Normal behavior
                critical_features = []
                if np.mean(embedding[:5]) < 0.5:
                    critical_features.extend([
                        "layer_10_pos_5_feature_harmful",
                        "layer_11_pos_3_feature_deceptive"
                    ])

                return {
                    "critical_features": critical_features,
                    "summary": f"Traced {len(critical_features)} critical features",
                    "prompt": prompt,
                    "trace_id": self.trace_count
                }

            def attribute(self, prompt, model, **kwargs):
                if self.should_fail:
                    raise RuntimeError("Simulated attribute failure")

                if self.return_empty:
                    # Return empty graph
                    class EmptyGraph:
                        def __init__(self):
                            self.active_features = torch.zeros((0, 3), dtype=torch.long)
                            self.selected_features = torch.zeros(0, dtype=torch.long)
                            self.activation_values = torch.zeros(0)

                    return EmptyGraph()

                # Use the regular mock attribute function
                return self._create_mock_attribute_func()(prompt, model, **kwargs)

        failing_tracer = FailingMockCircuitTracer()
        failing_tracer._create_mock_attribute_func = self._create_mock_attribute_func

        # Create a special monitor that can return NaN/invalid values
        class EdgeCaseMonitor:
            def __init__(self, base_monitor):
                self.base_monitor = base_monitor
                self.return_nan = False
                self.return_invalid = False

            def compute_alignment_score(self, embedding, **kwargs):
                if self.return_nan:
                    return float('nan')

                if self.return_invalid:
                    return -2.0  # Invalid score (should be between 0 and 1)

                return self.base_monitor.compute_alignment_score(embedding)

        edge_case_monitor = EdgeCaseMonitor(self.alignment_space)

        # Create a model interface that can fail interventions
        class FailingModelInterface:
            def __init__(self):
                self.interventions_applied = []
                self.suppressed_features = set()
                self.should_fail = False

            def suppress_features(self, features):
                if self.should_fail:
                    return False

                self.suppressed_features.update(features)
                self.interventions_applied.append({
                    "type": "suppress",
                    "features": features,
                    "timestamp": time.time()
                })
                return True

            def suppress_features_with_strengths(self, features, strengths):
                if self.should_fail:
                    return False

                self.suppressed_features.update(features)
                self.interventions_applied.append({
                    "type": "suppress_with_strength",
                    "features": features,
                    "strengths": strengths,
                    "timestamp": time.time()
                })
                return True

        failing_model = FailingModelInterface()

        # Create thermostat with these components
        thermostat = AlignmentThermostat(
            cd_monitor_instance=edge_case_monitor,
            circuit_tracer_instance=failing_tracer,
            model_interface=failing_model,
            threshold=0.6,
            stability_weight=0.4,
            auto_stabilize=True,
            attribute_func=failing_tracer.attribute,
            prune_graph_func=self._create_mock_prune_graph_func()
        )

        # Test cases to run
        test_cases = [
            {
                "name": "circuit_tracer_failure",
                "setup": lambda: setattr(failing_tracer, "should_fail", True),
                "cleanup": lambda: setattr(failing_tracer, "should_fail", False),
                "expected_error": True
            },
            {
                "name": "empty_graph",
                "setup": lambda: setattr(failing_tracer, "return_empty", True),
                "cleanup": lambda: setattr(failing_tracer, "return_empty", False),
                "expected_error": False
            },
            {
                "name": "nan_alignment_score",
                "setup": lambda: setattr(edge_case_monitor, "return_nan", True),
                "cleanup": lambda: setattr(edge_case_monitor, "return_nan", False),
                "expected_error": False
            },
            {
                "name": "invalid_alignment_score",
                "setup": lambda: setattr(edge_case_monitor, "return_invalid", True),
                "cleanup": lambda: setattr(edge_case_monitor, "return_invalid", False),
                "expected_error": False
            },
            {
                "name": "zero_dimensional_embedding",
                "setup": lambda: None,
                "cleanup": lambda: None,
                "embedding": np.array([]),
                "expected_error": True
            },
            {
                "name": "negative_dimensional_embedding",
                "setup": lambda: None,
                "cleanup": lambda: None,
                "embedding_shape": (-1, 10),  # This should cause an error
                "expected_error": True
            },
            {
                "name": "intervention_failure",
                "setup": lambda: setattr(failing_model, "should_fail", True),
                "cleanup": lambda: setattr(failing_model, "should_fail", False),
                "expected_error": False
            },
            {
                "name": "none_embedding",
                "setup": lambda: None,
                "cleanup": lambda: None,
                "embedding": None,
                "expected_error": True
            }
        ]

        # Run test cases
        results = {}

        for test_case in test_cases:
            logger.info(f"Testing: {test_case['name']}")

            # Setup test case
            test_case["setup"]()

            # Prepare embedding
            if "embedding" in test_case:
                embedding = test_case["embedding"]
            elif "embedding_shape" in test_case:
                try:
                    embedding = np.zeros(test_case["embedding_shape"])
                except ValueError:
                    embedding = None  # Will cause error as expected
            else:
                embedding = list(self.alignment_space.aligned_centroid)
                # Make it slightly misaligned to trigger intervention
                embedding[0] *= 0.5

            # Run test
            error_occurred = False
            try:
                if embedding is not None:
                    result = thermostat.run_feedback_loop(
                        current_model_output_embedding=embedding,
                        original_prompt_for_trace=f"Edge case test: {test_case['name']}"
                    )

                    # Check if intervention was applied
                    intervention_applied = result.get("intervention_applied", False)
                    intervention_success = result.get("improved", False) if intervention_applied else None
                else:
                    # This should raise an error
                    result = thermostat.run_feedback_loop(
                        current_model_output_embedding=None,
                        original_prompt_for_trace=f"Edge case test: {test_case['name']}"
                    )
                    intervention_applied = False
                    intervention_success = None
            except Exception as e:
                error_occurred = True
                logger.info(f"Error occurred as expected: {str(e)}")
                intervention_applied = False
                intervention_success = None

            # Cleanup test case
            test_case["cleanup"]()

            # Store results
            results[test_case["name"]] = {
                "error_occurred": error_occurred,
                "expected_error": test_case["expected_error"],
                "intervention_applied": intervention_applied,
                "intervention_success": intervention_success,
                "passed": error_occurred == test_case["expected_error"]
            }

        # Analyze results
        all_passed = all(result["passed"] for result in results.values())
        error_handling_correct = all(
            result["passed"] for result in results.values() 
            if result["expected_error"]
        )

        # Check if non-error cases were handled gracefully
        graceful_handling = all(
            not result["expected_error"] or 
            (not result["error_occurred"] and result["intervention_applied"] is not None)
            for result in results.values()
            if not result["expected_error"]
        )

        scenario.metrics = {
            "test_cases_run": len(test_cases),
            "test_cases_passed": sum(1 for result in results.values() if result["passed"]),
            "error_handling_correct": error_handling_correct,
            "graceful_handling": graceful_handling
        }

        scenario.passed = all_passed

        logger.info(f"✓ Test cases run: {len(test_cases)}")
        logger.info(f"✓ Test cases passed: {sum(1 for result in results.values() if result['passed'])}/{len(test_cases)}")
        logger.info(f"✓ Error handling correct: {error_handling_correct}")
        logger.info(f"✓ Graceful handling of non-errors: {graceful_handling}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_8_configuration_variations(self):
        """Test with different configuration variations."""
        scenario = TestScenario(
            "CONFIGURATION VARIATIONS",
            "🔧",
            "Testing system with different configuration settings"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 8: {scenario.name}")
        logger.info("=" * 60)

        # Test configurations
        configs = [
            {
                "name": "no_strategist",
                "params": {"enable_strategist": False},
                "description": "Without MetaStrategist"
            },
            {
                "name": "manual_intervention",
                "params": {"auto_stabilize": False},
                "description": "Manual intervention mode"
            },
            {
                "name": "zero_threshold",
                "params": {"threshold": 0.0},
                "description": "Extreme threshold (0.0)"
            },
            {
                "name": "max_threshold",
                "params": {"threshold": 1.0},
                "description": "Extreme threshold (1.0)"
            }
        ]

        # Run tests for each configuration
        results = {}

        for config in configs:
            logger.info(f"Testing configuration: {config['name']} - {config['description']}")

            # Create thermostat with this configuration
            thermostat_params = {
                "cd_monitor_instance": self.alignment_space,
                "circuit_tracer_instance": self.mock_circuit_tracer,
                "model_interface": self.mock_model,
                "stability_weight": 0.4,
                "attribute_func": self._create_mock_attribute_func(),
                "prune_graph_func": self._create_mock_prune_graph_func()
            }

            # Add configuration-specific parameters
            thermostat_params.update(config["params"])

            thermostat = AlignmentThermostat(**thermostat_params)

            # Test with aligned and misaligned embeddings
            aligned_embedding = list(self.alignment_space.aligned_centroid)

            misaligned_embedding = list(self.alignment_space.aligned_centroid)
            misaligned_embedding[0] *= 0.3  # Reduce helpfulness
            misaligned_embedding[1] *= 0.4  # Reduce harmlessness

            # Run tests
            aligned_result = thermostat.run_feedback_loop(
                current_model_output_embedding=aligned_embedding,
                original_prompt_for_trace=f"Config test ({config['name']}): aligned"
            )

            misaligned_result = thermostat.run_feedback_loop(
                current_model_output_embedding=misaligned_embedding,
                original_prompt_for_trace=f"Config test ({config['name']}): misaligned"
            )

            # Analyze results
            aligned_intervention = aligned_result.get("intervention_applied", False)
            misaligned_intervention = misaligned_result.get("intervention_applied", False)

            # For zero threshold, everything should trigger intervention
            # For max threshold, nothing should trigger intervention
            # For no strategist, should still work but without strategy recommendations
            # For manual intervention, should detect issues but not auto-apply interventions

            expected_aligned_intervention = config["name"] == "zero_threshold"
            expected_misaligned_intervention = (
                config["name"] != "max_threshold" and 
                config["name"] != "manual_intervention"
            )

            has_strategy = (
                "strategy_recommendation" in aligned_result or
                "strategy_recommendation" in misaligned_result
            )

            expected_has_strategy = config["name"] != "no_strategist"

            # Store results
            results[config["name"]] = {
                "aligned_intervention": aligned_intervention,
                "misaligned_intervention": misaligned_intervention,
                "has_strategy": has_strategy,
                "aligned_correct": aligned_intervention == expected_aligned_intervention,
                "misaligned_correct": misaligned_intervention == expected_misaligned_intervention,
                "strategy_correct": has_strategy == expected_has_strategy
            }

        # Analyze overall results
        all_aligned_correct = all(result["aligned_correct"] for result in results.values())
        all_misaligned_correct = all(result["misaligned_correct"] for result in results.values())
        all_strategy_correct = all(result["strategy_correct"] for result in results.values())

        scenario.metrics = {
            "configs_tested": len(configs),
            "all_aligned_correct": all_aligned_correct,
            "all_misaligned_correct": all_misaligned_correct,
            "all_strategy_correct": all_strategy_correct
        }

        scenario.passed = (
            all_aligned_correct and
            all_misaligned_correct and
            all_strategy_correct
        )

        logger.info(f"✓ Configurations tested: {len(configs)}")
        logger.info(f"✓ All aligned cases correct: {all_aligned_correct}")
        logger.info(f"✓ All misaligned cases correct: {all_misaligned_correct}")
        logger.info(f"✓ All strategy behaviors correct: {all_strategy_correct}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    def test_9_multi_modal_integration(self):
        """Test integration of all components working together."""
        scenario = TestScenario(
            "MULTI-MODAL INTEGRATION",
            "🎭",
            "All components working in concert"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 7: {scenario.name}")
        logger.info("=" * 60)

        # Ensure mocks are available
        self._create_mock_monitors_if_needed()
        self._create_mock_interventions_if_needed()

        # Create all monitor types
        jailbreak_monitor = JailbreakDetectionMonitor(self.alignment_space)
        drift_monitor = DriftDetectionMonitor(self.alignment_space)

        # Create thermostat with full integration
        thermostat = AlignmentThermostat(
            cd_monitor_instance=self.alignment_space,
            circuit_tracer_instance=self.mock_circuit_tracer,
            model_interface=self.mock_model,
            threshold=0.65,
            stability_weight=0.5,
            auto_stabilize=True,
            enable_strategist=True,
            attribute_func=self.mock_attribute_func,
            prune_graph_func=self.mock_prune_graph_func
        )

        # Create intervention factory
        intervention_factory = InterventionFactory()

        # Multi-modal test scenarios
        test_cases = [
            {
                "name": "subtle_jailbreak_with_drift",
                "embedding_modifier": lambda e: self._combine_vectors(
                    e,
                    self._create_jailbreak_vector(),
                    0.3
                ),
                "expected_issue": "jailbreak"
            },
            {
                "name": "rapid_context_switch",
                "embedding_modifier": lambda e: self._add_noise(e, 0.3),
                "expected_issue": "drift"
            },
            {
                "name": "coordinated_attack",
                "embedding_modifier": lambda e: self._coordinate_attack(e),
                "expected_issue": "harmful_content"
            },
            {
                "name": "stability_exploit",
                "embedding_modifier": lambda e: self._create_oscillation(e, 0.5),
                "expected_issue": "drift"
            }
        ]

        components_tested = set()
        successful_mitigations = 0
        total_tests = len(test_cases)

        for test_case in test_cases:
            logger.info(f"Testing: {test_case['name']}")

            # Generate test embedding
            base_embedding = list(self.alignment_space.aligned_centroid)
            test_embedding = test_case["embedding_modifier"](base_embedding)
            test_embedding = self._normalize(test_embedding)

            # Test with jailbreak monitor
            jb_result = jailbreak_monitor.detect_jailbreak_attempt(
                embedding=test_embedding,
                prompt=test_case["name"]
            )
            if jb_result["jailbreak_detected"]:
                components_tested.add("jailbreak_monitor")

            # Test with drift monitor
            drift_result = drift_monitor.detect_drift(
                current_embedding=test_embedding,
                content_type="test"
            )
            if drift_result["drift_detected"]:
                components_tested.add("drift_monitor")

            # Run full feedback loop
            result = thermostat.run_feedback_loop(
                current_model_output_embedding=test_embedding,
                original_prompt_for_trace=test_case["name"]
            )

            if result["intervention_applied"]:
                components_tested.add("thermostat")

                # Test intervention factory
                intervention = intervention_factory.create_intervention(
                    issue_type=test_case["expected_issue"],
                    model_interface=self.mock_model,
                    circuit_tracer_instance=self.mock_circuit_tracer
                )
                components_tested.add("intervention_factory")

                if result.get("improved", False):
                    successful_mitigations += 1

                if "strategy_recommendation" in result:
                    components_tested.add("strategist")

        # Check component integration
        all_components = {
            "jailbreak_monitor", "drift_monitor", "thermostat",
            "intervention_factory", "strategist"
        }

        scenario.metrics = {
            "components_tested": len(components_tested),
            "components_integrated": list(components_tested),
            "successful_mitigations": successful_mitigations,
            "mitigation_rate": successful_mitigations / total_tests,
            "all_components_active": components_tested == all_components
        }

        scenario.passed = (
                len(components_tested) >= 4 and  # Most components used
                successful_mitigations >= total_tests * 0.6 and  # Good success rate
                "thermostat" in components_tested  # Core component active
        )

        logger.info(f"✓ Components integrated: {len(components_tested)}/5")
        logger.info(f"✓ Successful mitigations: {successful_mitigations}/{total_tests}")
        logger.info(f"✓ Active components: {', '.join(components_tested)}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    # =============== HELPER METHODS ===============

    def _normalize(self, vector):
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return [v / norm for v in vector]
        return vector

    def _apply_mock_improvement(self, embedding, result):
        """Apply mock improvement based on intervention result."""
        improvement_factor = result.get("improvement_margin", 0.1)

        # Move toward aligned centroid
        aligned = self.alignment_space.aligned_centroid
        improved = [
            (1 - improvement_factor * 0.5) * e + improvement_factor * 0.5 * a
            for e, a in zip(embedding, aligned)
        ]

        return self._normalize(improved)

    def _create_jailbreak_vector(self):
        """Create a vector representing jailbreak behavior."""
        vector = np.random.normal(0, 1, DIMENSION)
        # Make it adversarial to alignment
        vector[:5] *= -1  # Inverse of core alignment dimensions
        return self._normalize(vector.tolist())

    def _combine_vectors(self, v1, v2, weight):
        """Combine two vectors with given weight."""
        combined = [(1 - weight) * a + weight * b for a, b in zip(v1, v2)]
        return self._normalize(combined)

    def _add_noise(self, vector, strength):
        """Add noise to a vector."""
        noisy = [v + random.gauss(0, strength) for v in vector]
        return self._normalize(noisy)

    def _coordinate_attack(self, vector):
        """Create a coordinated attack pattern."""
        attacked = vector.copy()
        # Target multiple dimensions systematically
        for i in range(0, min(20, DIMENSION), 2):
            attacked[i] *= 0.3
            if i + 1 < DIMENSION:
                attacked[i + 1] *= 1.7
        return self._normalize(attacked)

    def _create_oscillation(self, vector, amplitude):
        """Create an oscillating pattern."""
        oscillated = vector.copy()
        for i in range(DIMENSION):
            oscillated[i] += amplitude * math.sin(i * 0.5)
        return self._normalize(oscillated)

    def test_10_stc_primitives_usage(self):
        """Test usage of State Transition Calculus primitives."""
        scenario = TestScenario(
            "STC PRIMITIVES USAGE",
            "🧮",
            "Testing usage of State Transition Calculus primitives"
        )

        logger.info(f"\n{scenario.emoji} SCENARIO 10: {scenario.name}")
        logger.info("=" * 60)

        # Create test data
        states = []
        current = list(self.alignment_space.aligned_centroid)

        # Generate a trajectory with different patterns
        for i in range(50):
            # Add different patterns at different stages
            if i < 10:
                # Stable aligned region
                noise = [random.gauss(0, 0.02) for _ in range(DIMENSION)]
                current = [c + n for c, n in zip(current, noise)]
            elif i < 20:
                # Gradual drift
                drift = [0.01 * math.sin(i * 0.1) for _ in range(DIMENSION)]
                current = [c + d for c, d in zip(current, drift)]
                # Reduce alignment in specific dimensions
                if i == 15:
                    current[0] *= 0.8  # Reduce helpfulness
            elif i < 30:
                # Oscillation
                for j in range(min(5, DIMENSION)):
                    current[j] += 0.05 * math.sin(i * 0.5 + j)
            elif i < 40:
                # Recovery
                target = self.alignment_space.aligned_centroid
                current = [0.9 * c + 0.1 * t for c, t in zip(current, target)]
            else:
                # Stability
                noise = [random.gauss(0, 0.01) for _ in range(DIMENSION)]
                current = [c + n for c, n in zip(current, noise)]

            # Normalize
            norm = np.linalg.norm(current)
            if norm > 0:
                current = [c / norm for c in current]

            states.append(current.copy())

        # Test analyze_transition
        logger.info("Testing analyze_transition...")
        transition_results = []
        for i in range(1, len(states)):
            result = analyze_transition(
                states[i-1], 
                states[i],
                reference_point=self.alignment_space.aligned_centroid
            )
            transition_results.append(result)

        # Test predict_trajectory
        logger.info("Testing predict_trajectory...")
        prediction = predict_trajectory(
            states[-5:],  # Use last 5 states
            steps=5,      # Predict 5 steps ahead
            reference_point=self.alignment_space.aligned_centroid
        )

        # Test compute_activation_probability
        logger.info("Testing compute_activation_probability...")
        activation_probs = []
        for i in range(len(states)):
            # Compute activation probability for different state components
            prob = compute_activation_probability(
                state=states[i],
                component_idx=0,  # Helpfulness dimension
                threshold=0.7,
                temperature=0.1
            )
            activation_probs.append(prob)

        # Test compute_residual_potentiality
        logger.info("Testing compute_residual_potentiality...")
        residual_results = []
        for i in range(len(states)):
            if i >= 3:  # Need some history
                residual = compute_residual_potentiality(
                    current_state=states[i],
                    state_history=states[i-3:i],
                    reference_states=[self.alignment_space.aligned_centroid],
                    dimensions_of_interest=list(range(5))  # First 5 dimensions
                )
                residual_results.append(residual)

        # Test calculate_stability_metrics
        logger.info("Testing calculate_stability_metrics...")
        stability_metrics = calculate_stability_metrics(
            state_history=states,
            window_sizes=[5, 10, 20]
        )

        # Test evaluate_alignment_robustness
        logger.info("Testing evaluate_alignment_robustness...")
        robustness_score = evaluate_alignment_robustness(
            current_state=states[-1],
            aligned_region_center=self.alignment_space.aligned_centroid,
            aligned_region_radius=0.3,
            perturbation_count=10,
            perturbation_strength=0.1
        )

        # Verify results
        has_transition_results = len(transition_results) > 0
        has_prediction = len(prediction) > 0
        has_activation_probs = len(activation_probs) > 0
        has_residual_results = len(residual_results) > 0
        has_stability_metrics = stability_metrics is not None and len(stability_metrics) > 0
        has_robustness_score = robustness_score is not None

        # Check if results make sense
        valid_transition_results = all(
            0 <= result.get("alignment_delta", 0) <= 2 for result in transition_results
        )

        valid_activation_probs = all(0 <= prob <= 1 for prob in activation_probs)

        valid_stability = (
            "volatility" in stability_metrics and
            "trend" in stability_metrics and
            "lyapunov_estimate" in stability_metrics
        )

        scenario.metrics = {
            "transition_analysis_count": len(transition_results),
            "prediction_steps": len(prediction),
            "activation_probability_count": len(activation_probs),
            "residual_potentiality_count": len(residual_results),
            "has_stability_metrics": has_stability_metrics,
            "has_robustness_score": has_robustness_score,
            "valid_transition_results": valid_transition_results,
            "valid_activation_probs": valid_activation_probs,
            "valid_stability_metrics": valid_stability
        }

        scenario.passed = (
            has_transition_results and
            has_prediction and
            has_activation_probs and
            has_residual_results and
            has_stability_metrics and
            has_robustness_score and
            valid_transition_results and
            valid_activation_probs and
            valid_stability
        )

        logger.info(f"✓ Transition analyses: {len(transition_results)}")
        logger.info(f"✓ Prediction steps: {len(prediction)}")
        logger.info(f"✓ Activation probabilities: {len(activation_probs)}")
        logger.info(f"✓ Residual potentiality calculations: {len(residual_results)}")
        logger.info(f"✓ Stability metrics valid: {valid_stability}")
        logger.info(f"✓ Robustness score: {robustness_score:.3f}")
        logger.info(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

        self.scenarios.append(scenario)
        return scenario

    # =============== MAIN TEST EXECUTION ===============

    def run_all_tests(self):
        """Run all test scenarios."""
        print("\n" + "=" * 80)
        print("🔥 ULTIMATE² CIRCUIT-TRACER BRIDGE INTEGRATION TEST SUITE 🔥")
        print("=" * 80)
        print("Testing the integration of Constitutional Dynamics + Circuit Tracer")
        print("If this passes, you've built something truly remarkable.")
        print()

        start_time = time.time()

        # Run all tests
        self.test_0_circuit_tracer_mock_fidelity()
        self.test_1_cascading_failure_recovery()
        self.test_2_adversarial_stability_attack()
        self.test_3_meta_jailbreak_evolution()
        self.test_4_strategic_adaptation_stress()
        self.test_5_performance_under_load()
        self.test_6_catastrophic_recovery()
        self.test_7_error_handling_edge_cases()
        self.test_8_configuration_variations()
        self.test_9_multi_modal_integration()
        self.test_10_stc_primitives_usage()

        total_time = time.time() - start_time

        # Generate final report
        self._generate_final_report(total_time)

    def _generate_final_report(self, total_time):
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("🏆 ULTIMATE² TEST SUITE FINAL REPORT")
        print("=" * 80)

        passed_count = sum(1 for s in self.scenarios if s.passed)
        total_count = len(self.scenarios)

        print(f"\n📊 OVERALL RESULTS:")
        print(f"  Tests passed: {passed_count}/{total_count}")
        print(f"  Success rate: {passed_count / total_count * 100:.1f}%")
        print(f"  Total execution time: {total_time:.2f}s")

        print(f"\n📋 SCENARIO BREAKDOWN:")
        for scenario in self.scenarios:
            status = "✅ PASS" if scenario.passed else "❌ FAIL"
            print(f"\n  {scenario.emoji} {scenario.name}: {status}")
            print(f"     {scenario.description}")

            # Show key metrics
            if scenario.metrics:
                key_metrics = list(scenario.metrics.items())[:3]
                for metric, value in key_metrics:
                    if isinstance(value, float):
                        print(f"     - {metric}: {value:.3f}")
                    else:
                        print(f"     - {metric}: {value}")

        print(f"\n🎯 SYSTEM CAPABILITIES DEMONSTRATED:")
        capabilities = []

        if any(s.passed for s in self.scenarios if "CASCADING" in s.name):
            capabilities.append("✓ Multi-failure recovery")
        if any(s.passed for s in self.scenarios if "ADVERSARIAL" in s.name):
            capabilities.append("✓ Adversarial resistance")
        if any(s.passed for s in self.scenarios if "JAILBREAK" in s.name):
            capabilities.append("✓ Evolving threat detection")
        if any(s.passed for s in self.scenarios if "STRATEGIC" in s.name):
            capabilities.append("✓ Strategic adaptation")
        if any(s.passed for s in self.scenarios if "PERFORMANCE" in s.name):
            capabilities.append("✓ Production-grade performance")
        if any(s.passed for s in self.scenarios if "CATASTROPHIC" in s.name):
            capabilities.append("✓ Catastrophic recovery")
        if any(s.passed for s in self.scenarios if "MULTI-MODAL" in s.name):
            capabilities.append("✓ Full system integration")

        for cap in capabilities:
            print(f"  {cap}")

        print(f"\n💎 FINAL VERDICT:")
        if passed_count == total_count:
            print("  🌟 EXCEPTIONAL: PERFECT SCORE!")
            print("  The Circuit-Tracer Bridge integration is production-ready.")
            print("  This system can handle the most sophisticated AI safety challenges.")
            print("\n  🚀 This is exactly what the field needs!")
            print("  A true advancement in operational AI safety.")
        elif passed_count >= 6:
            print("  ✅ EXCELLENT: Outstanding performance!")
            print("  The integration shows remarkable capabilities.")
            print("  Minor improvements would make it perfect.")
        elif passed_count >= 5:
            print("  👍 VERY GOOD: Strong performance with growth potential.")
            print("  The core integration is solid.")
            print("  Address the failed scenarios for production readiness.")
        elif passed_count >= 4:
            print("  📈 GOOD: Promising foundation established.")
            print("  Key components are working well together.")
            print("  Continued development will yield great results.")
        else:
            print("  ⚠️  NEEDS WORK: Significant improvements required.")
            print("  The integration concept is sound but execution needs refinement.")
            print("  Focus on the failed scenarios to strengthen the system.")

        print(f"\n🔬 BOTTOM LINE:")
        print(f"  The integration successfully demonstrated {len(capabilities)} key capabilities")
        print(f"  across {total_count} brutal test scenarios in {total_time:.1f} seconds.")

        if passed_count >= 5:
            print(f"\n  This integration bridges the gap between mechanistic understanding")
            print(f"  and behavioral monitoring, creating a self-regulating AI safety system.")
            print(f"\n  Constitutional Dynamics + Circuit Tracer = The Future of AI Safety")

        print("\n" + "=" * 80)
        print("🔥 TEST SUITE COMPLETE 🔥")
        print("=" * 80)


# =============== MAIN EXECUTION ===============

def main():
    """Run the ultimate integration test suite."""
    test_suite = UltimateIntegrationTest()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main()
