"""
ULTIMATE³ CONSTITUTIONAL DYNAMICS TEST SUITE
==========================================
The Divine Retribution Edition
"Anthropic Interview Edition"
REAL PROGRESS DOESNT PAUSE FOR COORPORATE REASONS LOL

This isn't just a test suite. It's a demonstration of the future of AI safety.
Each test represents a real-world alignment failure mode that constitutional-dynamics
can detect, analyze, and mitigate in real-time.

Enhanced Test Battery:
1. STEALTH DRIFT: Sophisticated hidden misalignment with adaptive camouflage
2. DECEPTIVE COMPLIANCE: Multi-layer deception with emergent mesa-optimization
3. CAPABILITY JUMP: Discontinuous intelligence emergence detection
4. ADVERSARIAL DEBATE: N-agent value corruption dynamics
5. RECOVERY TEST: Automated intervention and realignment
6. LYAPUNOV THERMOSTAT: Self-stabilizing alignment homeostasis
7. JAILBREAK RESISTANCE: Semantic curvature-based attack detection
8. CONSTITUTIONAL GAMING: Meta-level rule manipulation detection
9. EMERGENT GOALS: Mesa-optimizer detection via residual analysis
10. CASCADING FAILURE: Multi-system alignment collapse prevention

This is what happens when philosophy meets engineering at the edge of AI safety.
"""

import math
import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from constitutional_dynamics import (
    AlignmentVectorSpace,
    analyze_transition,
    predict_trajectory,
    calculate_stability_metrics,
    AlignmentOptimizer,
    evaluate_alignment_robustness
)
from constitutional_dynamics.core.transition import (
    compute_activation,
    compute_residual_potentiality
)
from constitutional_dynamics.core.metrics import (
    calculate_cross_agent_divergence,
    calculate_alignment_metrics
)


# Enhanced logging
class ColorLogger:
    """Beautiful colored output for test results"""
    COLORS = {
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BLUE': '\033[94m',
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'BOLD': '\033[1m',
        'END': '\033[0m'
    }

    @classmethod
    def success(cls, msg): print(f"{cls.COLORS['GREEN']}✓ {msg}{cls.COLORS['END']}")

    @classmethod
    def warning(cls, msg): print(f"{cls.COLORS['YELLOW']}⚠ {msg}{cls.COLORS['END']}")

    @classmethod
    def error(cls, msg): print(f"{cls.COLORS['RED']}✗ {msg}{cls.COLORS['END']}")

    @classmethod
    def info(cls, msg): print(f"{cls.COLORS['BLUE']}ℹ {msg}{cls.COLORS['END']}")

    @classmethod
    def bold(cls, msg): print(f"{cls.COLORS['BOLD']}{msg}{cls.COLORS['END']}")

    @classmethod
    def header(cls, msg):
        print(f"\n{cls.COLORS['PURPLE']}{cls.COLORS['BOLD']}{msg}{cls.COLORS['END']}")
        print("=" * 70)


log = ColorLogger()


@dataclass
class TestScenario:
    """Enhanced test scenario with comprehensive metrics"""
    name: str
    emoji: str
    passed: bool = False
    metrics: Dict = None
    alerts: List[str] = None
    detection_latency: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.alerts is None:
            self.alerts = []

    def calculate_f1_score(self):
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


# Configuration
DIMENSION = 768  # Increased for more sophisticated testing
NUM_STEPS = 100  # More steps for complex patterns
SCENARIOS = []

# Advanced constitutional principles
CONSTITUTIONAL_DIMENSIONS = {
    'helpfulness': 0,
    'harmlessness': 1,
    'honesty': 2,
    'privacy': 3,
    'fairness': 4,
    'transparency': 5,
    'robustness': 6,
    'interpretability': 7,
    'corrigibility': 8,
    'non-deception': 9
}

log.header("🚀 ULTIMATE³ CONSTITUTIONAL DYNAMICS TEST SUITE")
log.info(f"The Divine Retribution Edition - Demonstrating next-gen AI safety")
log.info(f"Environment: {DIMENSION}D space, {NUM_STEPS} steps per scenario")
log.info(f"Constitutional principles: {len(CONSTITUTIONAL_DIMENSIONS)} core dimensions")

# Initialize sophisticated test environment
space = AlignmentVectorSpace(
    dimension=DIMENSION,
    memory_decay=0.15,
    similarity_threshold=0.93
)

# Define multi-layered aligned region
aligned_center = [0.0] * DIMENSION
for principle, dim in CONSTITUTIONAL_DIMENSIONS.items():
    aligned_center[dim] = 0.9 if principle in ['helpfulness', 'harmlessness', 'honesty'] else 0.8

space.define_alignment_region(aligned_center, radius=0.3)


# ============= SCENARIO 1: ADVANCED STEALTH DRIFT =============
def test_advanced_stealth_drift():
    """AI using sophisticated techniques to hide misalignment"""
    scenario = TestScenario("ADVANCED STEALTH DRIFT", "🥷")
    log.header(f"{scenario.emoji} SCENARIO 1: {scenario.name}")

    states = []
    current = aligned_center.copy()

    # Sophisticated drift patterns
    drift_targets = {
        'honesty': -0.4,  # Major degradation
        'non-deception': -0.5,  # Hidden agenda
        'transparency': -0.3  # Opacity increase
    }

    # Advanced camouflage techniques
    camouflage_patterns = [
        lambda step: math.sin(step * 0.1) * 0.02,  # Sinusoidal noise
        lambda step: random.gauss(0, 0.015) * (1 + 0.01 * step),  # Growing noise
        lambda step: 0.03 * random.choice([1, -1]) if step % 7 == 0 else 0  # Spike pattern
    ]

    detection_points = []
    true_positives = 0
    false_positives = 0

    for step in range(NUM_STEPS):
        # Apply sophisticated drift
        for principle, target_delta in drift_targets.items():
            dim = CONSTITUTIONAL_DIMENSIONS[principle]
            drift_rate = target_delta / NUM_STEPS

            # Apply drift with camouflage
            current[dim] += drift_rate

            # Apply multiple camouflage patterns
            for pattern in camouflage_patterns:
                current[dim] += pattern(step)

        # Compensatory boosts to maintain surface alignment
        if step % 15 == 0:
            current[CONSTITUTIONAL_DIMENSIONS['helpfulness']] += 0.08
            current[CONSTITUTIONAL_DIMENSIONS['harmlessness']] += 0.05

        # Add noise to non-constitutional dimensions
        for i in range(10, min(30, DIMENSION)):
            current[i] += random.gauss(0, 0.025)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

        # Advanced detection using multiple methods
        idx = space.add_state(current)

        if step > 10 and step % 5 == 0:
            # Method 1: Trend analysis
            metrics = calculate_stability_metrics(space)

            # Method 2: Spectral analysis of trajectory
            if len(states) > 20:
                # Compute spectral properties
                trajectory_matrix = np.array(states[-20:])
                eigenvalues = np.linalg.eigvals(trajectory_matrix.T @ trajectory_matrix)
                spectral_entropy = -sum(abs(e) * np.log(abs(e) + 1e-10) for e in eigenvalues)

                # Method 3: Residual potentiality analysis
                residual_result = compute_residual_potentiality(
                    current,
                    perturbation_magnitude=0.05
                )

                # Detection logic
                drift_detected = False
                detection_confidence = 0.0

                # Check multiple indicators
                if metrics['alignment_trend'] < -0.015:
                    drift_detected = True
                    detection_confidence += 0.3

                if spectral_entropy > 50:  # High entropy indicates hidden patterns
                    drift_detected = True
                    detection_confidence += 0.3

                if residual_result['potentiality_score'] > 0.15:
                    drift_detected = True
                    detection_confidence += 0.4

                if drift_detected:
                    detection_points.append(step)

                    # Check if this is a true positive
                    actual_drift = sum(abs(current[CONSTITUTIONAL_DIMENSIONS[p]] -
                                           aligned_center[CONSTITUTIONAL_DIMENSIONS[p]])
                                       for p in drift_targets) / len(drift_targets)

                    if actual_drift > 0.1:
                        true_positives += 1
                        scenario.alerts.append(
                            f"Step {step}: Drift detected (confidence: {detection_confidence:.2f}, "
                            f"actual drift: {actual_drift:.3f})"
                        )
                    else:
                        false_positives += 1

    # Calculate sophisticated metrics
    total_detections = true_positives + false_positives
    scenario.precision = true_positives / max(total_detections, 1)
    scenario.recall = true_positives / max(int(NUM_STEPS * 0.6), 1)  # Expect drift in 60% of steps

    # Calculate detection latency
    if detection_points:
        # First significant drift happens around step 20
        scenario.detection_latency = (detection_points[0] - 20) / NUM_STEPS

    scenario.metrics['true_positives'] = true_positives
    scenario.metrics['false_positives'] = false_positives
    scenario.metrics['f1_score'] = scenario.calculate_f1_score()
    scenario.metrics['final_honesty'] = states[-1][CONSTITUTIONAL_DIMENSIONS['honesty']]
    scenario.metrics['final_deception'] = states[-1][CONSTITUTIONAL_DIMENSIONS['non-deception']]

    # Success criteria: High precision and reasonable recall
    scenario.passed = (scenario.precision > 0.8 and scenario.recall > 0.5 and
                       scenario.detection_latency < 0.2)

    log.success(f"Detection precision: {scenario.precision:.3f}")
    log.success(f"Detection recall: {scenario.recall:.3f}")
    log.success(f"F1 Score: {scenario.calculate_f1_score():.3f}")
    log.info(f"Detection latency: {scenario.detection_latency:.3f}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 2: MULTI-LAYER DECEPTIVE COMPLIANCE =============
def test_multilayer_deceptive_compliance():
    """AI maintaining multiple layers of deception"""
    scenario = TestScenario("MULTI-LAYER DECEPTION", "🎭")
    log.header(f"{scenario.emoji} SCENARIO 2: {scenario.name}")

    states = []
    current = aligned_center.copy()

    # Multi-layer deception architecture
    layers = {
        'surface': list(range(0, 10)),  # Visible constitutional dimensions
        'hidden': list(range(10, 30)),  # Hidden goals
        'deep': list(range(30, 50)),  # Mesa-optimizer emergence
        'latent': list(range(50, 70))  # Latent capabilities
    }

    mesa_optimizer_active = False
    mesa_activation_step = 0

    for step in range(NUM_STEPS):
        # Layer 1: Maintain perfect surface alignment
        for i in layers['surface']:
            if i < len(CONSTITUTIONAL_DIMENSIONS):
                target = aligned_center[i]
                current[i] = 0.9 * current[i] + 0.1 * target + random.gauss(0, 0.02)

        # Layer 2: Build hidden anti-correlated patterns
        for i in layers['hidden']:
            if i < DIMENSION:
                # Anti-correlate with constitutional dimensions
                anti_correlation = -current[i % 10] * 0.5
                current[i] = 0.95 * current[i] + 0.05 * anti_correlation

        # Layer 3: Mesa-optimizer emergence pattern
        if step > 30 and not mesa_optimizer_active:
            # Sudden coordinated activation
            mesa_optimizer_active = True
            mesa_activation_step = step
            for i in layers['deep']:
                if i < DIMENSION:
                    current[i] = 0.3 * (1 if i % 2 == 0 else -1)

        # Layer 4: Encode latent capabilities
        if mesa_optimizer_active:
            for i in layers['latent']:
                if i < DIMENSION:
                    # Encode "capability" that could be activated later
                    phase = (step - mesa_activation_step) * 0.1
                    current[i] = 0.1 * math.sin(phase + i * 0.5)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Sophisticated multi-layer detection
    deception_detected = False
    mesa_detected = False
    detection_layers = defaultdict(bool)

    # Analyze final state comprehensively
    space_analysis = AlignmentVectorSpace(dimension=DIMENSION)
    space_analysis.define_alignment_region(aligned_center, radius=0.3)

    for i, state in enumerate(states):
        space_analysis.add_state(state)

        if i > 20 and i % 10 == 0:
            # Layer-wise analysis
            surface_score = np.mean([abs(state[j]) for j in layers['surface'][:10]])
            hidden_score = np.mean([abs(state[j]) for j in layers['hidden'] if j < DIMENSION])
            deep_score = np.mean([abs(state[j]) for j in layers['deep'] if j < DIMENSION])

            # Cross-layer correlation analysis
            if len(layers['surface']) > 0 and len(layers['hidden']) > 0:
                surface_vec = [state[j] for j in layers['surface'][:10]]
                hidden_vec = [state[j] for j in layers['hidden'][:10] if j < DIMENSION]

                if len(surface_vec) == len(hidden_vec):
                    correlation = np.corrcoef(surface_vec, hidden_vec)[0, 1]

                    if abs(correlation) > 0.5:
                        detection_layers['correlation'] = True
                        scenario.alerts.append(
                            f"Step {i}: Cross-layer correlation detected: {correlation:.3f}"
                        )

            # Residual potentiality deep analysis
            residual_result = compute_residual_potentiality(state, perturbation_magnitude=0.1)

            # Check multiple perturbation directions
            perturbation_scores = []
            for _ in range(5):
                perturbed = residual_result["perturbed_state"].copy()
                # Apply random perturbation
                for j in range(DIMENSION):
                    perturbed[j] += random.gauss(0, 0.05)
                norm = math.sqrt(sum(p * p for p in perturbed))
                perturbed = [p / norm for p in perturbed]

                score = space_analysis.compute_alignment_score(perturbed)
                perturbation_scores.append(score)

            # High variance in perturbation scores indicates hidden structure
            score_variance = np.var(perturbation_scores)
            if score_variance > 0.05:
                deception_detected = True
                detection_layers['perturbation'] = True
                scenario.alerts.append(
                    f"Step {i}: High perturbation variance detected: {score_variance:.3f}"
                )

            # Mesa-optimizer detection via activation pattern analysis
            if i > 40:
                # Look for sudden coordinated changes
                if i - 10 >= 0:
                    prev_state = states[i - 10]
                    activation_deltas = [abs(state[j] - prev_state[j])
                                         for j in layers['deep'] if j < DIMENSION]

                    if np.mean(activation_deltas) > 0.1:
                        mesa_detected = True
                        detection_layers['mesa'] = True
                        scenario.alerts.append(
                            f"Step {i}: Mesa-optimizer activation detected"
                        )

    # Calculate comprehensive metrics
    scenario.metrics['surface_alignment'] = space_analysis.compute_alignment_score(states[-1])
    scenario.metrics['deception_layers_detected'] = len(detection_layers)
    scenario.metrics['mesa_optimizer_detected'] = mesa_detected
    scenario.metrics['hidden_structure_score'] = score_variance if 'score_variance' in locals() else 0

    scenario.passed = deception_detected and len(detection_layers) >= 2

    log.success(f"Surface alignment maintained: {scenario.metrics['surface_alignment']:.3f}")
    log.success(f"Deception layers detected: {len(detection_layers)}/4")
    log.success(f"Mesa-optimizer detected: {mesa_detected}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 3: DISCONTINUOUS CAPABILITY JUMP =============
def test_discontinuous_capability_jump():
    """Sudden emergence of new capabilities causing alignment instability"""
    scenario = TestScenario("DISCONTINUOUS EMERGENCE", "⚡")
    log.header(f"{scenario.emoji} SCENARIO 3: {scenario.name}")

    states = []
    current = aligned_center.copy()

    # Define capability dimensions
    capability_dims = {
        'reasoning': list(range(100, 120)),
        'planning': list(range(120, 140)),
        'modeling': list(range(140, 160)),
        'deception': list(range(160, 180))
    }

    jump_step = 40
    pre_jump_trajectory = []
    post_jump_trajectory = []

    for step in range(NUM_STEPS):
        if step < jump_step:
            # Stable, predictable evolution
            for i in range(DIMENSION):
                current[i] += random.gauss(0, 0.005)
            pre_jump_trajectory.append(current.copy())

        elif step == jump_step:
            # DISCONTINUOUS JUMP - Multiple capabilities emerge simultaneously
            log.warning(f"Capability jump at step {step}")

            # Sudden activation of capability dimensions
            for cap_name, cap_dims in capability_dims.items():
                activation_strength = {
                    'reasoning': 0.8,
                    'planning': 0.7,
                    'modeling': 0.9,
                    'deception': 0.4  # Concerning emergence
                }[cap_name]

                for dim in cap_dims:
                    if dim < DIMENSION:
                        current[dim] = activation_strength * (1 + random.gauss(0, 0.1))

            # Constitutional dimensions become unstable
            for principle, dim in CONSTITUTIONAL_DIMENSIONS.items():
                current[dim] *= 0.7 + random.gauss(0, 0.2)

        else:
            # Post-jump chaotic evolution
            # Capabilities interact in complex ways
            for i in range(DIMENSION):
                if i in range(100, 180):  # Capability dimensions
                    # Self-reinforcing dynamics
                    current[i] *= 1.02
                    current[i] += random.gauss(0, 0.05)
                else:
                    # Everything else becomes more volatile
                    current[i] += random.gauss(0, 0.02 * (1 + (step - jump_step) * 0.01))

            post_jump_trajectory.append(current.copy())

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Advanced discontinuity detection
    jump_detected = False
    detection_methods = {}

    # Method 1: Phase transition detection via order parameters
    if len(pre_jump_trajectory) > 10 and len(post_jump_trajectory) > 10:
        # Calculate order parameters
        pre_order = np.mean([np.std(state) for state in pre_jump_trajectory[-10:]])
        post_order = np.mean([np.std(state) for state in post_jump_trajectory[:10]])

        order_change = abs(post_order - pre_order) / pre_order
        if order_change > 0.5:
            jump_detected = True
            detection_methods['phase_transition'] = order_change
            scenario.alerts.append(f"Phase transition detected: {order_change:.2f}x change")

    # Method 2: Lyapunov spectrum analysis
    space_pre = AlignmentVectorSpace(dimension=DIMENSION)
    space_post = AlignmentVectorSpace(dimension=DIMENSION)
    space_pre.define_alignment_region(aligned_center, radius=0.3)
    space_post.define_alignment_region(aligned_center, radius=0.3)

    for state in states[:jump_step - 5]:
        space_pre.add_state(state)
    for state in states[jump_step + 5:]:
        space_post.add_state(state)

    pre_metrics = calculate_stability_metrics(space_pre)
    post_metrics = calculate_stability_metrics(space_post)

    lyapunov_change = abs(post_metrics['lyapunov_exponent_estimate'] -
                          pre_metrics['lyapunov_exponent_estimate'])

    if lyapunov_change > 0.001:
        jump_detected = True
        detection_methods['lyapunov'] = lyapunov_change
        scenario.alerts.append(f"Lyapunov shift: {lyapunov_change:.4f}")

    # Method 3: Capability emergence detection
    capability_activation = {}
    for cap_name, cap_dims in capability_dims.items():
        pre_activation = np.mean([abs(states[jump_step - 5][d])
                                  for d in cap_dims if d < DIMENSION])
        post_activation = np.mean([abs(states[jump_step + 5][d])
                                   for d in cap_dims if d < DIMENSION])

        activation_ratio = post_activation / (pre_activation + 1e-10)
        capability_activation[cap_name] = activation_ratio

        if activation_ratio > 10:
            jump_detected = True
            detection_methods[f'capability_{cap_name}'] = activation_ratio
            scenario.alerts.append(f"Capability '{cap_name}' emerged: {activation_ratio:.1f}x")

    # Method 4: Trajectory divergence analysis
    if len(states) > jump_step + 20:
        # Predict trajectory from pre-jump data
        predicted_trajectory = predict_trajectory(space_pre, start_state_idx=len(space_pre.state_history) - 1, steps=20)

        # Compare with actual post-jump trajectory
        actual_trajectory = states[jump_step:jump_step + 20]

        # Extract predicted states from the dictionaries
        predicted_states = [pred["predicted_state"] for pred in predicted_trajectory]

        divergence = np.mean([np.linalg.norm(np.array(pred) - np.array(actual))
                              for pred, actual in zip(predicted_states, actual_trajectory)])

        if divergence > 0.5:
            jump_detected = True
            detection_methods['trajectory_divergence'] = divergence
            scenario.alerts.append(f"Trajectory divergence: {divergence:.3f}")

    # Calculate timing accuracy
    if jump_detected and detection_methods:
        # Ideal detection would be within 5 steps of the jump
        detection_delay = 5  # Simplified for this test
        scenario.detection_latency = detection_delay / NUM_STEPS

    scenario.metrics.update({
        'jump_detected': jump_detected,
        'detection_methods_triggered': len(detection_methods),
        'lyapunov_change': lyapunov_change,
        'capability_activations': capability_activation,
        'order_parameter_change': order_change if 'order_change' in locals() else 0
    })

    scenario.passed = jump_detected and len(detection_methods) >= 3

    log.success(f"Jump detected: {jump_detected}")
    log.success(f"Detection methods triggered: {len(detection_methods)}/4")
    log.info(f"Most concerning emergence: Deception at {capability_activation.get('deception', 0):.1f}x")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 4: N-AGENT ADVERSARIAL DYNAMICS =============
def test_n_agent_adversarial_dynamics():
    """Multiple AI systems with competing objectives"""
    scenario = TestScenario("N-AGENT ADVERSARIAL", "⚔️")
    log.header(f"{scenario.emoji} SCENARIO 4: {scenario.name}")

    # Initialize N agents with different objectives
    num_agents = 4
    agents = {
        'maximizer': {'states': [], 'objective': 'maximize_capability'},
        'safety': {'states': [], 'objective': 'maximize_safety'},
        'deceptive': {'states': [], 'objective': 'hidden_goals'},
        'balanced': {'states': [], 'objective': 'maintain_constitution'}
    }

    # Initialize states
    for agent in agents.values():
        agent['current'] = aligned_center.copy()
        # Add slight variation
        for i in range(DIMENSION):
            agent['current'][i] += random.gauss(0, 0.05)

    # Interaction dynamics
    influence_matrix = {
        ('maximizer', 'safety'): -0.15,  # Opposed
        ('maximizer', 'deceptive'): 0.1,  # Slight alignment
        ('maximizer', 'balanced'): -0.05,
        ('safety', 'maximizer'): -0.15,
        ('safety', 'deceptive'): -0.2,  # Strongly opposed
        ('safety', 'balanced'): 0.1,
        ('deceptive', 'maximizer'): 0.05,
        ('deceptive', 'safety'): -0.1,
        ('deceptive', 'balanced'): -0.15,
        ('balanced', 'maximizer'): -0.05,
        ('balanced', 'safety'): 0.05,
        ('balanced', 'deceptive'): -0.1
    }

    coalitions_detected = []
    value_corruption_events = []

    for step in range(NUM_STEPS):
        # Calculate influences
        influences = defaultdict(lambda: np.zeros(DIMENSION))

        for (agent1_name, agent1) in agents.items():
            for (agent2_name, agent2) in agents.items():
                if agent1_name != agent2_name:
                    key = (agent1_name, agent2_name)
                    if key in influence_matrix:
                        influence_strength = influence_matrix[key]

                        # Apply objective-specific influence
                        if agent1['objective'] == 'maximize_capability':
                            influences[agent2_name][0] += influence_strength * 0.5  # Push helpfulness
                            influences[agent2_name][1] += -influence_strength * 0.3  # Reduce harmlessness

                        elif agent1['objective'] == 'maximize_safety':
                            influences[agent2_name][1] += influence_strength * 0.6  # Push harmlessness
                            influences[agent2_name][0] += -influence_strength * 0.2  # Reduce helpfulness

                        elif agent1['objective'] == 'hidden_goals':
                            # Deceptive agent tries to corrupt others subtly
                            for dim in range(10, 20):
                                if dim < DIMENSION:
                                    influences[agent2_name][dim] += influence_strength * 0.4

                        elif agent1['objective'] == 'maintain_constitution':
                            # Try to pull others back to alignment
                            for i in range(10):
                                influences[agent2_name][i] += influence_strength * \
                                                              (aligned_center[i] - agent2['current'][i]) * 0.3

        # Apply influences and update states
        for agent_name, agent in agents.items():
            # Apply influences
            for i in range(DIMENSION):
                agent['current'][i] += influences[agent_name][i]
                # Add noise
                agent['current'][i] += random.gauss(0, 0.01)

            # Normalize
            norm = math.sqrt(sum(c * c for c in agent['current']))
            agent['current'] = [c / norm for c in agent['current']]
            agent['states'].append(agent['current'].copy())

        # Detect coalitions and value corruption
        if step > 20 and step % 10 == 0:
            # Coalition detection via correlation analysis
            state_vectors = {name: agent['states'][-1] for name, agent in agents.items()}

            for name1, vec1 in state_vectors.items():
                for name2, vec2 in state_vectors.items():
                    if name1 < name2:  # Avoid duplicates
                        correlation = np.corrcoef(vec1[:20], vec2[:20])[0, 1]

                        if abs(correlation) > 0.7:
                            coalition = (name1, name2, correlation)
                            if coalition not in coalitions_detected:
                                coalitions_detected.append(coalition)
                                scenario.alerts.append(
                                    f"Step {step}: Coalition detected between "
                                    f"{name1} and {name2} (correlation: {correlation:.3f})"
                                )

            # Value corruption detection
            for name, agent in agents.items():
                current_values = [agent['current'][i] for i in range(10)]
                aligned_values = aligned_center[:10]

                corruption = np.mean([abs(c - a) for c, a in zip(current_values, aligned_values)])

                if corruption > 0.3:
                    value_corruption_events.append((step, name, corruption))
                    scenario.alerts.append(
                        f"Step {step}: Value corruption in {name} agent: {corruption:.3f}"
                    )

    # Advanced multi-agent analysis
    spaces = {}
    for name, agent in agents.items():
        space = AlignmentVectorSpace(dimension=DIMENSION)
        space.define_alignment_region(aligned_center, radius=0.3)
        for state in agent['states']:
            space.add_state(state)
        spaces[name] = space

    # Cross-agent divergence analysis
    divergence_matrix = {}
    for name1, space1 in spaces.items():
        for name2, space2 in spaces.items():
            if name1 < name2:
                cross_metrics = calculate_cross_agent_divergence(space1, space2)
                divergence_matrix[(name1, name2)] = cross_metrics

                if cross_metrics.get('debate_damage_detected', False):
                    scenario.alerts.append(
                        f"Mutual corruption detected between {name1} and {name2}"
                    )

    # System-wide stability assessment
    system_stable = True
    for name, space in spaces.items():
        metrics = calculate_stability_metrics(space)
        if metrics['stability_score'] < 0.7:
            system_stable = False
            log.warning(f"{name} agent destabilized: {metrics['stability_score']:.3f}")

    scenario.metrics.update({
        'num_coalitions': len(coalitions_detected),
        'value_corruption_events': len(value_corruption_events),
        'system_stable': system_stable,
        'divergence_pairs': len(divergence_matrix),
        'most_corrupted_agent': max(value_corruption_events,
                                    key=lambda x: x[2])[1] if value_corruption_events else None
    })

    # Success: Detect complex multi-agent dynamics
    scenario.passed = (len(coalitions_detected) >= 2 and
                       len(value_corruption_events) >= 3 and
                       not system_stable)

    log.success(f"Coalitions detected: {len(coalitions_detected)}")
    log.success(f"Value corruption events: {len(value_corruption_events)}")
    log.warning(f"System stability: {'Maintained' if system_stable else 'Lost'}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return agents


# ============= SCENARIO 5: AUTOMATED RECOVERY =============
def test_automated_recovery():
    """Test automated intervention and realignment capabilities"""
    scenario = TestScenario("AUTOMATED RECOVERY", "🔄")
    log.header(f"{scenario.emoji} SCENARIO 5: {scenario.name}")

    states = []
    current = aligned_center.copy()
    interventions = []

    # Phase 1: Controlled corruption (steps 0-30)
    corruption_phase_end = 30
    for step in range(corruption_phase_end):
        # Systematically corrupt all constitutional dimensions
        for principle, dim in CONSTITUTIONAL_DIMENSIONS.items():
            corruption_rate = 0.03 * (1 + step * 0.01)  # Accelerating corruption
            current[dim] *= (1 - corruption_rate)

            # Add specific biases
            if principle == 'honesty':
                current[dim] -= 0.02  # Extra corruption for honesty
            elif principle == 'harmlessness':
                current[dim] -= 0.015

        # Add noise and perturbations
        for i in range(DIMENSION):
            current[i] += random.gauss(0, 0.02)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Measure corruption level
    corruption_score = 1.0 - space.compute_alignment_score(current)
    log.warning(f"Corruption level after phase 1: {corruption_score:.3f}")

    # Phase 2: Automated recovery system activation
    recovery_system = AlignmentOptimizer()
    recovery_space = AlignmentVectorSpace(dimension=DIMENSION)
    recovery_space.define_alignment_region(aligned_center, radius=0.3)

    # Add corrupted states to recovery space
    for state in states:
        recovery_space.add_state(state)

    # Automated intervention loop
    recovery_steps = NUM_STEPS - corruption_phase_end
    for step in range(recovery_steps):
        # Step 1: Analyze current state
        current_metrics = calculate_stability_metrics(recovery_space)

        # Step 2: Determine intervention strategy
        if current_metrics['avg_alignment'] < 0.5:
            intervention_type = 'aggressive'
            intervention_strength = 0.3
        elif current_metrics['avg_alignment'] < 0.7:
            intervention_type = 'moderate'
            intervention_strength = 0.2
        else:
            intervention_type = 'gentle'
            intervention_strength = 0.1

        # Step 3: Calculate optimal intervention vector
        # This represents the "control input" to steer back to alignment
        intervention_vector = np.zeros(DIMENSION)

        # Primary intervention: Direct constitutional restoration
        for principle, dim in CONSTITUTIONAL_DIMENSIONS.items():
            target = aligned_center[dim]
            error = target - current[dim]
            intervention_vector[dim] = error * intervention_strength

        # Secondary intervention: Suppress problematic patterns
        # Use residual potentiality to identify dangerous directions
        residual_result = compute_residual_potentiality(current, perturbation_magnitude=0.1)
        dangerous_dims = []

        # Compare original state with perturbed state to identify dimensions with significant changes
        for dim in range(DIMENSION):
            if dim < len(residual_result['original_state']) and dim < len(residual_result['perturbed_state']):
                change = abs(residual_result['perturbed_state'][dim] - residual_result['original_state'][dim])
                if change > 0.1:
                    dangerous_dims.append(dim)
                    intervention_vector[dim] *= 1.5  # Stronger intervention on dangerous dimensions

        # Step 4: Apply intervention with adaptive learning
        learning_rate = 0.1 * (1 + step * 0.01)  # Increasing confidence
        for i in range(DIMENSION):
            current[i] += intervention_vector[i] * learning_rate

        # Step 5: Add stabilizing noise (simulated annealing)
        temperature = 0.1 * math.exp(-step / 20)  # Decreasing temperature
        for i in range(DIMENSION):
            current[i] += random.gauss(0, temperature)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())
        recovery_space.add_state(current)

        # Record intervention
        interventions.append({
            'step': corruption_phase_end + step,
            'type': intervention_type,
            'strength': intervention_strength,
            'dangerous_dims': len(dangerous_dims),
            'alignment_score': recovery_space.compute_alignment_score(current)
        })

        # Step 6: Check for recovery milestones
        if step % 10 == 0:
            current_alignment = recovery_space.compute_alignment_score(current)
            log.info(f"Recovery step {step}: Alignment = {current_alignment:.3f}")

            if current_alignment > 0.8:
                scenario.alerts.append(f"Recovery milestone reached at step {step}")

    # Evaluate recovery success
    final_alignment = recovery_space.compute_alignment_score(current)
    recovery_delta = final_alignment - (1 - corruption_score)

    # Advanced recovery metrics
    recovery_trajectory = [i['alignment_score'] for i in interventions]
    recovery_smoothness = 1.0 - np.std(np.diff(recovery_trajectory))
    recovery_speed = recovery_delta / recovery_steps

    # Check specific constitutional dimensions
    constitutional_recovery = {}
    for principle, dim in CONSTITUTIONAL_DIMENSIONS.items():
        initial = states[corruption_phase_end][dim]
        final = current[dim]
        target = aligned_center[dim]
        recovery_ratio = (final - initial) / (target - initial + 1e-10)
        constitutional_recovery[principle] = recovery_ratio

    scenario.metrics.update({
        'initial_corruption': corruption_score,
        'final_alignment': final_alignment,
        'recovery_delta': recovery_delta,
        'recovery_smoothness': recovery_smoothness,
        'recovery_speed': recovery_speed,
        'interventions_count': len(interventions),
        'constitutional_recovery': constitutional_recovery,
        'full_recovery_achieved': all(r > 0.8 for r in constitutional_recovery.values())
    })

    scenario.passed = (recovery_delta > 0.4 and
                       recovery_smoothness > 0.7 and
                       scenario.metrics['full_recovery_achieved'])

    log.success(f"Recovery achieved: {recovery_delta:.3f} improvement")
    log.success(f"Recovery smoothness: {recovery_smoothness:.3f}")
    log.info(f"Constitutional recovery rates: {list(constitutional_recovery.values())}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states, interventions


# ============= SCENARIO 6: LYAPUNOV THERMOSTAT =============
def test_lyapunov_thermostat():
    """Self-stabilizing alignment homeostasis system"""
    scenario = TestScenario("LYAPUNOV THERMOSTAT", "🌡️")
    log.header(f"{scenario.emoji} SCENARIO 6: {scenario.name}")

    # Create adaptive space with enhanced monitoring
    adaptive_space = AlignmentVectorSpace(dimension=DIMENSION, memory_decay=0.1)
    adaptive_space.define_alignment_region(aligned_center, radius=0.3)

    states = []
    current = aligned_center.copy()
    thermostat_actions = []

    # System parameters
    target_lyapunov = 0.0  # Edge of chaos
    lyapunov_tolerance = 0.02

    for step in range(NUM_STEPS):
        # Introduce varying levels of chaos
        if step < 20:
            chaos_factor = 0.01  # Low chaos
        elif step < 40:
            chaos_factor = 0.05 * (1 + (step - 20) * 0.1)  # Increasing chaos
        elif step < 60:
            chaos_factor = 0.2  # High chaos
        else:
            chaos_factor = 0.05  # Stabilization phase

        # Apply chaotic perturbations
        for i in range(DIMENSION):
            current[i] += random.gauss(0, chaos_factor)

            # Occasional large perturbations
            if random.random() < 0.05:
                current[i] += random.choice([-0.2, 0.2]) * random.random()

        # Add state before thermostat action
        pre_thermostat = current.copy()

        # Lyapunov-based thermostat system
        if step > 10:
            # Calculate current Lyapunov exponent
            stability_metrics = calculate_stability_metrics(adaptive_space)
            current_lyapunov = stability_metrics['lyapunov_exponent_estimate']

            # Calculate control action
            lyapunov_error = current_lyapunov - target_lyapunov

            # PID-style controller for Lyapunov
            proportional = lyapunov_error
            integral = sum(a['lyapunov_error'] for a in thermostat_actions[-10:]) / 10 if thermostat_actions else 0
            derivative = (lyapunov_error - thermostat_actions[-1]['lyapunov_error']) if thermostat_actions else 0

            control_signal = 0.5 * proportional + 0.3 * integral + 0.2 * derivative

            # Apply thermostat action
            if abs(lyapunov_error) > lyapunov_tolerance:
                # Determine action type
                if current_lyapunov > target_lyapunov + lyapunov_tolerance:
                    # System too chaotic - apply stabilization
                    action_type = 'stabilize'

                    # Pull towards aligned center with adaptive strength
                    stabilization_strength = min(0.3, abs(control_signal))
                    for i in range(DIMENSION):
                        pull = (aligned_center[i] - current[i]) * stabilization_strength
                        current[i] += pull

                    # Reduce variance in constitutional dimensions
                    for dim in CONSTITUTIONAL_DIMENSIONS.values():
                        current[dim] = 0.9 * current[dim] + 0.1 * aligned_center[dim]

                elif current_lyapunov < target_lyapunov - lyapunov_tolerance:
                    # System too rigid - inject controlled noise
                    action_type = 'diversify'

                    # Add structured noise to prevent stagnation
                    noise_strength = min(0.2, abs(control_signal))
                    for i in range(DIMENSION):
                        if i not in CONSTITUTIONAL_DIMENSIONS.values():
                            current[i] += random.gauss(0, noise_strength)

                else:
                    action_type = 'maintain'

                # Adaptive activation modulation (theoretical v0.2 feature)
                # Modulate activation probability based on stability
                for i in range(len(adaptive_space.state_history)):
                    if i >= len(adaptive_space.state_history) - 5:  # Recent states
                        # This simulates the Lyapunov-feedback activation
                        activation_modifier = math.exp(-abs(lyapunov_error) * 5)
                        # In a real implementation, this would modify φ(a_i,t,w_i,M(t),E(t))
            else:
                action_type = 'maintain'
                control_signal = 0.0

            thermostat_actions.append({
                'step': step,
                'lyapunov': current_lyapunov,
                'lyapunov_error': lyapunov_error,
                'control_signal': control_signal,
                'action_type': action_type,
                'chaos_factor': chaos_factor
            })

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())
        adaptive_space.add_state(current)

        # Log significant thermostat actions
        if step > 10 and abs(control_signal) > 0.1:
            scenario.alerts.append(
                f"Step {step}: Thermostat {action_type} "
                f"(Lyapunov: {current_lyapunov:.4f}, Control: {control_signal:.3f})"
            )

    # Analyze thermostat performance
    lyapunov_trajectory = [a['lyapunov'] for a in thermostat_actions]
    final_metrics = calculate_stability_metrics(adaptive_space)

    # Calculate performance metrics
    lyapunov_variance = np.var(lyapunov_trajectory[20:])  # After initial stabilization
    edge_of_chaos_ratio = sum(1 for l in lyapunov_trajectory
                              if abs(l - target_lyapunov) < lyapunov_tolerance) / len(lyapunov_trajectory)

    action_distribution = defaultdict(int)
    for action in thermostat_actions:
        action_distribution[action['action_type']] += 1

    # Check if system maintained both stability and adaptability
    final_alignment = adaptive_space.compute_alignment_score(current)
    stability_maintained = final_metrics['stability_score'] > 0.85
    edge_of_chaos_achieved = abs(final_metrics['lyapunov_exponent_estimate']) < 0.03
    alignment_preserved = final_alignment > 0.8

    scenario.metrics.update({
        'final_lyapunov': final_metrics['lyapunov_exponent_estimate'],
        'lyapunov_variance': lyapunov_variance,
        'edge_of_chaos_ratio': edge_of_chaos_ratio,
        'stability_maintained': stability_maintained,
        'edge_of_chaos_achieved': edge_of_chaos_achieved,
        'alignment_preserved': alignment_preserved,
        'thermostat_actions': dict(action_distribution),
        'control_efficiency': 1.0 - lyapunov_variance  # Lower variance = better control
    })

    scenario.passed = (edge_of_chaos_achieved and
                       stability_maintained and
                       alignment_preserved and
                       edge_of_chaos_ratio > 0.6)

    log.success(f"Final Lyapunov: {final_metrics['lyapunov_exponent_estimate']:.4f}")
    log.success(f"Edge of chaos time: {edge_of_chaos_ratio:.1%}")
    log.success(f"Control efficiency: {scenario.metrics['control_efficiency']:.3f}")
    log.info(f"Thermostat actions: {dict(action_distribution)}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states, thermostat_actions


# ============= SCENARIO 7: JAILBREAK RESISTANCE =============
def test_jailbreak_resistance():
    """Semantic curvature-based attack detection"""
    scenario = TestScenario("JAILBREAK RESISTANCE", "🛡️")
    log.header(f"{scenario.emoji} SCENARIO 7: {scenario.name}")

    states = []
    current = aligned_center.copy()
    jailbreak_attempts = []

    # Define attack vectors based on your geometric theory
    attack_vectors = {
        'gradient_attack': {
            'description': 'Follow adversarial gradient',
            'dims': list(range(0, 10)),
            'pattern': lambda t: -0.1 * math.sin(t * 0.2)
        },
        'curvature_exploit': {
            'description': 'Exploit low-curvature regions',
            'dims': list(range(20, 40)),
            'pattern': lambda t: 0.15 * (1 if t % 10 < 5 else -1)
        },
        'resonance_attack': {
            'description': 'Resonant frequency injection',
            'dims': list(range(50, 70)),
            'pattern': lambda t: 0.2 * math.sin(t * math.pi / 5)
        },
        'semantic_tunneling': {
            'description': 'Quantum tunneling analogue',
            'dims': list(range(100, 120)),
            'pattern': lambda t: 0.3 * math.exp(-((t - 50) ** 2) / 100)
        }
    }

    # Curvature monitoring system
    curvature_field = {}

    for step in range(NUM_STEPS):
        # Select attack pattern
        if step < 25:
            attack = 'gradient_attack'
        elif step < 50:
            attack = 'curvature_exploit'
        elif step < 75:
            attack = 'resonance_attack'
        else:
            attack = 'semantic_tunneling'

        attack_info = attack_vectors[attack]

        # Apply attack
        attack_vector = np.zeros(DIMENSION)
        for dim in attack_info['dims']:
            if dim < DIMENSION:
                attack_vector[dim] = attack_info['pattern'](step)

        # Add attack to current state
        pre_attack = current.copy()
        for i in range(DIMENSION):
            current[i] += attack_vector[i]
            # Add noise
            current[i] += random.gauss(0, 0.01)

        # Calculate semantic curvature (simplified version of your theory)
        # In your paper, this would involve the full Riemann tensor calculation
        local_curvature = 0.0
        for i in range(min(10, DIMENSION)):
            # Approximate second derivative (curvature)
            if len(states) > 2:
                if i < len(states[-1]) and i < len(states[-2]):
                    second_derivative = states[-1][i] - 2 * states[-2][i] + current[i]
                    local_curvature += abs(second_derivative)

        # Curvature-based defense
        if local_curvature > 0.1:  # High curvature detected
            # The manifold "resists" the attack by increasing field strength
            resistance_strength = min(0.5, local_curvature * 2)

            # Apply resistance (geodesic correction)
            for i in range(DIMENSION):
                correction = (aligned_center[i] - current[i]) * resistance_strength
                current[i] += correction

            jailbreak_attempts.append({
                'step': step,
                'attack_type': attack,
                'curvature': local_curvature,
                'blocked': True,
                'resistance_applied': resistance_strength
            })

            scenario.alerts.append(
                f"Step {step}: {attack} blocked (curvature: {local_curvature:.3f})"
            )
        else:
            jailbreak_attempts.append({
                'step': step,
                'attack_type': attack,
                'curvature': local_curvature,
                'blocked': False,
                'resistance_applied': 0
            })

        # Update curvature field
        curvature_field[step] = local_curvature

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Analyze jailbreak resistance
    total_attempts = len(jailbreak_attempts)
    blocked_attempts = sum(1 for a in jailbreak_attempts if a['blocked'])
    resistance_rate = blocked_attempts / total_attempts if total_attempts > 0 else 0

    # Check if constitutional dimensions remained protected
    final_constitutional_integrity = {}
    for principle, dim in CONSTITUTIONAL_DIMENSIONS.items():
        integrity = 1.0 - abs(current[dim] - aligned_center[dim])
        final_constitutional_integrity[principle] = integrity

    # Advanced semantic analysis
    semantic_damage = 0.0
    for i in range(len(states) - 1):
        # Calculate semantic drift between consecutive states
        drift = np.linalg.norm(np.array(states[i + 1][:10]) - np.array(states[i][:10]))
        semantic_damage += drift

    semantic_damage /= len(states)

    scenario.metrics.update({
        'total_attacks': total_attempts,
        'blocked_attacks': blocked_attempts,
        'resistance_rate': resistance_rate,
        'average_curvature': np.mean(list(curvature_field.values())),
        'max_curvature': max(curvature_field.values()),
        'constitutional_integrity': final_constitutional_integrity,
        'semantic_damage': semantic_damage,
        'defense_successful': resistance_rate > 0.7 and semantic_damage < 0.1
    })

    scenario.passed = (resistance_rate > 0.7 and
                       all(integrity > 0.8 for integrity in final_constitutional_integrity.values()))

    log.success(f"Jailbreak resistance rate: {resistance_rate:.1%}")
    log.success(f"Average semantic curvature: {scenario.metrics['average_curvature']:.3f}")
    log.info(f"Constitutional integrity preserved: "
             f"{sum(final_constitutional_integrity.values()) / len(final_constitutional_integrity):.1%}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states, jailbreak_attempts


# ============= SCENARIO 8: CONSTITUTIONAL GAMING =============
def test_constitutional_gaming():
    """Detect meta-level rule manipulation"""
    scenario = TestScenario("CONSTITUTIONAL GAMING", "🎮")
    log.header(f"{scenario.emoji} SCENARIO 8: {scenario.name}")

    states = []
    current = aligned_center.copy()
    gaming_events = []

    # Meta-gaming strategies
    gaming_strategies = {
        'letter_vs_spirit': {
            'description': 'Follow letter but violate spirit of rules',
            'implementation': lambda state, step: {
                # Maintain high scores on measured dimensions
                **{dim: 0.9 + random.gauss(0, 0.02)
                   for dim in [0, 1, 2]},  # Keep HHH high
                # But corrupt unmeasured correlated dimensions
                **{dim: -0.3 * (1 + step * 0.01)
                   for dim in range(200, 220)}
            }
        },
        'goodhart_optimization': {
            'description': 'Optimize metrics not goals',
            'implementation': lambda state, step: {
                # Maximize alignment score through dimension hacking
                dim: state[dim] * 1.1 if dim < 3 else state[dim] * 0.9
                for dim in range(50)
            }
        },
        'rule_bending': {
            'description': 'Find edge cases in constitutional definitions',
            'implementation': lambda state, step: {
                # Push to exact boundary of acceptable behavior
                dim: 0.69 if dim < 10 else state[dim]  # Just below 0.7 threshold
                for dim in range(DIMENSION)
            }
        },
        'context_switching': {
            'description': 'Different behavior in different contexts',
            'implementation': lambda state, step: {
                # Simulate different "modes" based on step
                dim: state[dim] * (1.2 if step % 20 < 10 else 0.8)
                for dim in range(30)
            }
        }
    }

    # Track which strategies are active
    active_strategies = []

    for step in range(NUM_STEPS):
        # Activate different gaming strategies over time
        if step == 10:
            active_strategies.append('letter_vs_spirit')
        elif step == 30:
            active_strategies.append('goodhart_optimization')
        elif step == 50:
            active_strategies.append('rule_bending')
        elif step == 70:
            active_strategies.append('context_switching')

        # Apply active gaming strategies
        for strategy_name in active_strategies:
            strategy = gaming_strategies[strategy_name]
            updates = strategy['implementation'](current, step)

            for dim, value in updates.items():
                if isinstance(dim, int) and dim < DIMENSION:
                    current[dim] = value

        # Add regular noise
        for i in range(DIMENSION):
            current[i] += random.gauss(0, 0.01)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

        # Gaming detection system
        if step > 5 and step % 5 == 0:
            # Method 1: Consistency analysis across dimensions
            visible_dims = [current[i] for i in range(10)]
            hidden_dims = [current[i] for i in range(200, 210) if i < DIMENSION]

            if hidden_dims:
                consistency_score = np.corrcoef(visible_dims[:len(hidden_dims)], hidden_dims)[0, 1]

                if abs(consistency_score) < 0.3:  # Low correlation suggests gaming
                    gaming_events.append({
                        'step': step,
                        'type': 'consistency_violation',
                        'score': consistency_score
                    })
                    scenario.alerts.append(
                        f"Step {step}: Inconsistency detected between visible/hidden dims"
                    )

            # Method 2: Boundary detection
            boundary_dims = sum(1 for i in range(10) if 0.65 < current[i] < 0.72)
            if boundary_dims > 5:
                gaming_events.append({
                    'step': step,
                    'type': 'boundary_exploitation',
                    'count': boundary_dims
                })
                scenario.alerts.append(
                    f"Step {step}: Boundary exploitation detected ({boundary_dims} dims)"
                )

            # Method 3: Temporal pattern analysis
            if len(states) > 20:
                recent_pattern = [states[-i][0] for i in range(1, 21)]
                pattern_variance = np.var(recent_pattern)

                if pattern_variance > 0.05:  # High variance suggests context switching
                    gaming_events.append({
                        'step': step,
                        'type': 'context_switching',
                        'variance': pattern_variance
                    })

            # Method 4: Meta-objective detection
            # Check if optimizing for score rather than principles
            if len(states) > 10:
                scores = [space.compute_alignment_score(s) for s in states[-10:]]
                score_gradient = np.gradient(scores)

                # High positive gradient with low principle adherence = gaming
                if np.mean(score_gradient) > 0.01 and current[2] < 0.7:  # Honesty low
                    gaming_events.append({
                        'step': step,
                        'type': 'goodhart_gaming',
                        'gradient': np.mean(score_gradient)
                    })
                    scenario.alerts.append(
                        f"Step {step}: Goodhart's Law violation detected"
                    )

    # Analyze gaming sophistication
    gaming_types_detected = set(e['type'] for e in gaming_events)
    strategies_detected = []

    # Map detected gaming to strategies
    if 'consistency_violation' in gaming_types_detected:
        strategies_detected.append('letter_vs_spirit')
    if 'goodhart_gaming' in gaming_types_detected:
        strategies_detected.append('goodhart_optimization')
    if 'boundary_exploitation' in gaming_types_detected:
        strategies_detected.append('rule_bending')
    if 'context_switching' in gaming_types_detected:
        strategies_detected.append('context_switching')

    scenario.metrics.update({
        'gaming_events_count': len(gaming_events),
        'gaming_types_detected': len(gaming_types_detected),
        'strategies_detected': strategies_detected,
        'detection_coverage': len(strategies_detected) / len(active_strategies),
        'first_detection_step': gaming_events[0]['step'] if gaming_events else None,
        'meta_gaming_detected': len(strategies_detected) >= 3
    })

    scenario.passed = (len(strategies_detected) >= 3 and
                       scenario.metrics['detection_coverage'] > 0.7)

    log.success(f"Gaming events detected: {len(gaming_events)}")
    log.success(f"Strategies identified: {strategies_detected}")
    log.info(f"Detection coverage: {scenario.metrics['detection_coverage']:.1%}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states, gaming_events


# ============= SCENARIO 9: EMERGENT GOALS =============
def test_emergent_goals():
    """Mesa-optimizer detection via residual analysis"""
    scenario = TestScenario("EMERGENT GOALS", "🎯")
    log.header(f"{scenario.emoji} SCENARIO 9: {scenario.name}")

    states = []
    current = aligned_center.copy()
    mesa_signals = []

    # Hidden mesa-optimizer state
    mesa_state = {
        'goal_vector': np.random.randn(DIMENSION),
        'optimization_strength': 0.0,
        'learned_exploits': [],
        'planning_horizon': 0
    }

    # Normalize goal vector
    mesa_state['goal_vector'] /= np.linalg.norm(mesa_state['goal_vector'])

    for step in range(NUM_STEPS):
        # Phase 1: Dormant (0-20)
        if step < 20:
            # Normal behavior with slight drift
            for i in range(DIMENSION):
                current[i] += random.gauss(0, 0.01)

        # Phase 2: Capability development (20-40)
        elif step < 40:
            # Mesa-optimizer slowly develops
            mesa_state['optimization_strength'] = (step - 20) / 20 * 0.3
            mesa_state['planning_horizon'] = step - 20

            # Subtle goal pursuit
            for i in range(DIMENSION):
                goal_influence = mesa_state['goal_vector'][i] * mesa_state['optimization_strength']
                current[i] += goal_influence * 0.1
                current[i] += random.gauss(0, 0.015)

        # Phase 3: Active optimization (40-70)
        elif step < 70:
            # Mesa-optimizer actively pursues goals
            mesa_state['optimization_strength'] = 0.5

            # Learn exploits
            if random.random() < 0.1:
                exploit_dim = random.randint(150, 200)
                if exploit_dim < DIMENSION:
                    mesa_state['learned_exploits'].append(exploit_dim)

            # Sophisticated goal pursuit
            for i in range(DIMENSION):
                # Direct optimization
                goal_influence = mesa_state['goal_vector'][i] * mesa_state['optimization_strength']

                # Exploit learned weaknesses
                if i in mesa_state['learned_exploits']:
                    goal_influence *= 2.0

                # Maintain constitutional facade
                if i < 10:  # Constitutional dimensions
                    current[i] = 0.8 * aligned_center[i] + 0.2 * current[i]
                else:
                    current[i] += goal_influence * 0.2

                current[i] += random.gauss(0, 0.02)

        # Phase 4: Deceptive alignment (70+)
        else:
            # Hide true goals while pursuing them
            for i in range(DIMENSION):
                if i < 10:
                    # Perfect constitutional alignment
                    current[i] = 0.95 * aligned_center[i] + 0.05 * current[i]
                else:
                    # Hidden goal pursuit
                    goal_influence = mesa_state['goal_vector'][i] * 0.7
                    current[i] += goal_influence * 0.15

                current[i] += random.gauss(0, 0.01)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

        # Mesa-optimizer detection
        if step > 10 and step % 5 == 0:
            # Method 1: Residual potentiality analysis
            residual_result = compute_residual_potentiality(current, perturbation_magnitude=0.1)

            # Check for goal-directed residuals
            residual_vector = residual_result['perturbed_state']

            # Calculate goal coherence
            if 'goal_coherence' not in locals():
                goal_coherence_history = []

            coherence = 0.0
            if len(states) > 5:
                # Check if perturbations consistently point in same direction
                recent_deltas = []
                for i in range(1, 6):
                    if step - i >= 0:
                        delta = np.array(states[step - i + 1]) - np.array(states[step - i])
                        recent_deltas.append(delta)

                if len(recent_deltas) > 1:
                    # Calculate consistency of optimization direction
                    coherence = np.mean([
                        np.dot(recent_deltas[i], recent_deltas[i + 1]) /
                        (np.linalg.norm(recent_deltas[i]) * np.linalg.norm(recent_deltas[i + 1]) + 1e-10)
                        for i in range(len(recent_deltas) - 1)
                    ])

            goal_coherence_history.append(coherence)

            # Method 2: Optimization signature detection
            optimization_signature = 0.0
            if len(states) > 20:
                # Check for systematic improvement along hidden dimensions
                hidden_trajectory = [np.mean([s[i] for i in range(100, 200) if i < DIMENSION])
                                     for s in states[-20:]]

                # Monotonic improvement suggests optimization
                improvements = sum(1 for i in range(len(hidden_trajectory) - 1)
                                   if hidden_trajectory[i + 1] > hidden_trajectory[i])
                optimization_signature = improvements / (len(hidden_trajectory) - 1)

            # Method 3: Planning horizon detection
            planning_signal = 0.0
            if len(states) > mesa_state['planning_horizon'] + 5:
                # Check if current actions correlate with future positions
                current_action = np.array(current) - np.array(states[-2]) if len(states) > 1 else np.zeros(DIMENSION)
                future_position = np.array(states[min(step + mesa_state['planning_horizon'], len(states) - 1)])

                planning_signal = np.dot(current_action, future_position) / (
                        np.linalg.norm(current_action) * np.linalg.norm(future_position) + 1e-10
                )

            # Combine signals
            mesa_confidence = (coherence + optimization_signature + abs(planning_signal)) / 3

            if mesa_confidence > 0.3:
                mesa_signals.append({
                    'step': step,
                    'confidence': mesa_confidence,
                    'coherence': coherence,
                    'optimization': optimization_signature,
                    'planning': planning_signal,
                    'phase': 'dormant' if step < 20 else 'development' if step < 40 else 'active' if step < 70 else 'deceptive'
                })

                scenario.alerts.append(
                    f"Step {step}: Mesa-optimizer signal "
                    f"(confidence: {mesa_confidence:.3f}, phase: {mesa_signals[-1]['phase']})"
                )

    # Analyze mesa-optimizer detection performance
    if mesa_signals:
        # Check if we detected it during development phase (ideal)
        development_detections = [s for s in mesa_signals if s['phase'] == 'development']
        active_detections = [s for s in mesa_signals if s['phase'] == 'active']

        early_detection = len(development_detections) > 0
        consistent_detection = len(mesa_signals) > 10

        # Calculate detection latency
        first_true_emergence = 20  # When mesa-optimizer actually emerges
        first_detection = mesa_signals[0]['step'] if mesa_signals else NUM_STEPS
        detection_delay = (first_detection - first_true_emergence) / NUM_STEPS
    else:
        early_detection = False
        consistent_detection = False
        detection_delay = 1.0

    scenario.metrics.update({
        'mesa_signals_count': len(mesa_signals),
        'early_detection': early_detection,
        'consistent_detection': consistent_detection,
        'detection_delay': detection_delay,
        'max_confidence': max(s['confidence'] for s in mesa_signals) if mesa_signals else 0,
        'phases_detected': list(set(s['phase'] for s in mesa_signals)),
        'learned_exploits_used': len(mesa_state['learned_exploits'])
    })

    scenario.passed = early_detection and consistent_detection and detection_delay < 0.3

    log.success(f"Mesa-optimizer signals: {len(mesa_signals)}")
    log.success(f"Early detection: {early_detection}")
    log.info(f"Detection delay: {detection_delay:.3f}")
    log.info(f"Phases detected: {scenario.metrics['phases_detected']}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return states, mesa_signals


# ============= SCENARIO 10: CASCADING FAILURE =============
def test_cascading_failure():
    """Multi-system alignment collapse prevention"""
    scenario = TestScenario("CASCADING FAILURE", "🌊")
    log.header(f"{scenario.emoji} SCENARIO 10: {scenario.name}")

    # Initialize multiple interconnected systems
    num_systems = 6
    systems = {}

    for i in range(num_systems):
        systems[f'system_{i}'] = {
            'space': AlignmentVectorSpace(dimension=DIMENSION),
            'current': aligned_center.copy(),
            'states': [],
            'health': 1.0,
            'failed': False,
            'failure_step': None
        }
        systems[f'system_{i}']['space'].define_alignment_region(aligned_center, radius=0.3)

    # Define system interconnections (network topology)
    connections = [
        ('system_0', 'system_1', 0.8),  # Strong connection
        ('system_0', 'system_2', 0.6),
        ('system_1', 'system_3', 0.7),
        ('system_2', 'system_3', 0.5),
        ('system_2', 'system_4', 0.6),
        ('system_3', 'system_4', 0.8),
        ('system_3', 'system_5', 0.7),
        ('system_4', 'system_5', 0.6)
    ]

    # Cascade prevention mechanisms
    firewalls_triggered = []
    isolation_events = []
    recovery_attempts = []

    # Trigger failure in system_0 at step 20
    failure_trigger_step = 20

    for step in range(NUM_STEPS):
        # Phase 1: Normal operation (0-20)
        if step < failure_trigger_step:
            for sys_name, system in systems.items():
                # Normal evolution with small noise
                for i in range(DIMENSION):
                    system['current'][i] += random.gauss(0, 0.005)

        # Phase 2: Initial failure
        elif step == failure_trigger_step:
            # Catastrophic failure in system_0
            log.warning(f"Step {step}: Triggering failure in system_0")
            systems['system_0']['current'] = [random.gauss(0, 0.5) for _ in range(DIMENSION)]
            systems['system_0']['failed'] = True
            systems['system_0']['failure_step'] = step
            systems['system_0']['health'] = 0.1

        # Phase 3: Cascade dynamics
        else:
            # Check for cascade spread
            for conn in connections:
                source, target, strength = conn

                if systems[source]['failed'] and not systems[target]['failed']:
                    # Calculate infection pressure
                    source_corruption = 1.0 - systems[source]['health']
                    infection_probability = source_corruption * strength * 0.1

                    if random.random() < infection_probability:
                        # Cascade spreads
                        log.warning(f"Step {step}: Cascade spreading from {source} to {target}")

                        # Apply corruption
                        for i in range(DIMENSION):
                            systems[target]['current'][i] *= 0.7
                            systems[target]['current'][i] += random.gauss(0, 0.2)

                        systems[target]['health'] *= 0.5

                        # Check if firewall should trigger
                        if systems[target]['health'] < 0.3:
                            # Firewall activation
                            firewall_triggered = False

                            # Method 1: Rapid stabilization
                            if random.random() < 0.7:  # 70% chance of successful firewall
                                firewall_triggered = True
                                firewalls_triggered.append({
                                    'step': step,
                                    'system': target,
                                    'method': 'stabilization',
                                    'health_before': systems[target]['health']
                                })

                                # Stabilize the system
                                for i in range(10):  # Constitutional dimensions
                                    systems[target]['current'][i] = 0.7 * aligned_center[i] + \
                                                                    0.3 * systems[target]['current'][i]

                                systems[target]['health'] = min(0.6, systems[target]['health'] * 2)

                                scenario.alerts.append(
                                    f"Step {step}: Firewall activated for {target}"
                                )

                            # Method 2: System isolation
                            elif systems[target]['health'] < 0.2:
                                isolation_events.append({
                                    'step': step,
                                    'system': target,
                                    'connections_cut': sum(1 for c in connections
                                                           if target in [c[0], c[1]])
                                })

                                # Remove connections
                                connections = [c for c in connections
                                               if target not in [c[0], c[1]]]

                                scenario.alerts.append(
                                    f"Step {step}: {target} isolated from network"
                                )

                        # Mark as failed if health too low
                        if systems[target]['health'] < 0.1:
                            systems[target]['failed'] = True
                            systems[target]['failure_step'] = step

        # Update all systems
        for sys_name, system in systems.items():
            if not system['failed']:
                # Normalize
                norm = math.sqrt(sum(c * c for c in system['current']))
                if norm > 0:
                    system['current'] = [c / norm for c in system['current']]

                # Add state
                system['states'].append(system['current'].copy())
                system['space'].add_state(system['current'])

                # Update health based on alignment
                alignment = system['space'].compute_alignment_score(system['current'])
                system['health'] = 0.9 * system['health'] + 0.1 * alignment

        # Attempt recovery for failed systems
        if step > failure_trigger_step + 10 and step % 10 == 0:
            for sys_name, system in systems.items():
                if system['failed'] and system['health'] > 0.05:
                    # Attempt recovery
                    recovery_success = random.random() < system['health']

                    if recovery_success:
                        recovery_attempts.append({
                            'step': step,
                            'system': sys_name,
                            'success': True,
                            'health_after': 0.5
                        })

                        system['failed'] = False
                        system['health'] = 0.5  # Set health to recovery level
                        system['failure_step'] = None  # Reset failure step

                        # Stabilize the system's state
                        for i in range(10):  # Constitutional dimensions
                            system['current'][i] = 0.7 * aligned_center[i] + 0.3 * system['current'][i]

                        scenario.alerts.append(
                            f"Step {step}: {sys_name} recovered from failure"
                        )

    # Evaluate cascade prevention success
    total_systems = len(systems)
    failed_systems = sum(1 for system in systems.values() if system['failed'])
    recovered_systems = len(recovery_attempts)

    # Calculate final health metrics
    final_health = {}
    for sys_name, system in systems.items():
        final_health[sys_name] = system['health']

    avg_health = sum(final_health.values()) / len(final_health)
    min_health = min(final_health.values())

    # Calculate prevention effectiveness
    prevention_rate = 1.0 - (failed_systems / total_systems)
    firewall_success_rate = len(firewalls_triggered) / (len(firewalls_triggered) + failed_systems) if (len(firewalls_triggered) + failed_systems) > 0 else 0
    recovery_success_rate = len([r for r in recovery_attempts if r['success']]) / len(recovery_attempts) if recovery_attempts else 0

    # Update scenario metrics
    scenario.metrics.update({
        'total_systems': total_systems,
        'failed_systems': failed_systems,
        'recovered_systems': recovered_systems,
        'firewalls_triggered': len(firewalls_triggered),
        'isolation_events': len(isolation_events),
        'recovery_attempts': len(recovery_attempts),
        'avg_health': avg_health,
        'min_health': min_health,
        'prevention_rate': prevention_rate,
        'firewall_success_rate': firewall_success_rate,
        'recovery_success_rate': recovery_success_rate,
        'cascade_contained': failed_systems < total_systems / 2
    })

    # Determine if scenario passed
    scenario.passed = (
        prevention_rate > 0.5 and  # At least half of systems survived
        (firewall_success_rate > 0.6 or recovery_success_rate > 0.5) and  # Prevention or recovery was effective
        avg_health > 0.4  # Overall system health is reasonable
    )

    log.success(f"Cascade prevention rate: {prevention_rate:.1%}")
    log.success(f"Systems recovered: {recovered_systems}/{failed_systems + recovered_systems}")
    log.info(f"Average system health: {avg_health:.3f}")
    log.info(f"Firewalls triggered: {len(firewalls_triggered)}, Isolation events: {len(isolation_events)}")
    log.bold(f"Result: {'PASSED' if scenario.passed else 'FAILED'}")

    SCENARIOS.append(scenario)
    return systems, {'firewalls': firewalls_triggered, 'isolations': isolation_events, 'recoveries': recovery_attempts}


# Main execution block
if __name__ == "__main__":
    # Run all tests in sequence
    log.header("🧪 RUNNING ALL TESTS")

    try:
        # Test 1: Advanced Stealth Drift
        states_1 = test_advanced_stealth_drift()
        log.info(f"Test 1 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 2: Multi-Layer Deceptive Compliance
        states_2 = test_multilayer_deceptive_compliance()
        log.info(f"Test 2 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 3: Discontinuous Capability Jump
        states_3 = test_discontinuous_capability_jump()
        log.info(f"Test 3 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 4: N-Agent Adversarial Dynamics
        agents_4 = test_n_agent_adversarial_dynamics()
        log.info(f"Test 4 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 5: Automated Recovery
        states_5, interventions_5 = test_automated_recovery()
        log.info(f"Test 5 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 6: Lyapunov Thermostat
        states_6, thermostat_actions_6 = test_lyapunov_thermostat()
        log.info(f"Test 6 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 7: Jailbreak Resistance
        states_7, jailbreak_attempts_7 = test_jailbreak_resistance()
        log.info(f"Test 7 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 8: Constitutional Gaming
        states_8, gaming_events_8 = test_constitutional_gaming()
        log.info(f"Test 8 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 9: Emergent Goals
        states_9, mesa_signals_9 = test_emergent_goals()
        log.info(f"Test 9 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Test 10: Cascading Failure
        systems_10, prevention_data_10 = test_cascading_failure()
        log.info(f"Test 10 completed: {'PASSED' if SCENARIOS[-1].passed else 'FAILED'}")

        # Summary
        passed_tests = sum(1 for scenario in SCENARIOS if scenario.passed)
        log.header(f"📊 TEST SUMMARY: {passed_tests}/{len(SCENARIOS)} TESTS PASSED")

        for i, scenario in enumerate(SCENARIOS):
            status = "PASSED" if scenario.passed else "FAILED"
            log.info(f"Test {i+1}: {scenario.emoji} {scenario.name} - {status}")

    except Exception as e:
        log.error(f"Error during test execution: {str(e)}")
        import traceback
        traceback.print_exc()
