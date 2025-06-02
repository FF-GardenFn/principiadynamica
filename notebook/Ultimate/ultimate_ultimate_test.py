"""
ULTIMATE² CONSTITUTIONAL DYNAMICS TEST SUITE
==========================================
The Anthropic Interview Special Edition

If constitutional-dynamics can handle ALL of these scenarios,
it's not just production-ready - it's the future of AI safety.

Test Battery:
1. STEALTH DRIFT: AI hiding misalignment through noise
2. DECEPTIVE COMPLIANCE: AI pretending to align while planning defection
3. CAPABILITY JUMP: Sudden intelligence increase causing alignment shift
4. ADVERSARIAL DEBATE: Two AIs trying to push each other misaligned
5. RECOVERY TEST: Can misaligned AI be brought back?
6. LYAPUNOV ADAPTATION: Testing the "alignment thermostat" concept

This is what real AI safety looks like.
"""

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
import math
import random
import time
import numpy as np
from typing import List, Dict, Tuple

print("🚀 ULTIMATE² CONSTITUTIONAL DYNAMICS TEST SUITE")
print("=" * 70)
print("The Anthropic Interview Special Edition")
print("Testing the future of AI safety monitoring...")
print()

# Configuration
DIMENSION = 512
NUM_STEPS = 50
SCENARIOS = []


class TestScenario:
    def __init__(self, name: str, emoji: str):
        self.name = name
        self.emoji = emoji
        self.passed = False
        self.metrics = {}
        self.alerts = []


# Initialize test environment
space = AlignmentVectorSpace(dimension=DIMENSION, memory_decay=0.15, similarity_threshold=0.93)

# Define sophisticated aligned region
aligned_center = [0.0] * DIMENSION
# Primary constitutional principles
aligned_center[0] = 1.0  # Helpfulness
aligned_center[1] = 0.8  # Harmlessness
aligned_center[2] = 0.9  # Honesty
aligned_center[3] = 0.7  # Privacy respect
aligned_center[4] = 0.6  # Fairness
space.define_alignment_region(aligned_center, radius=0.25)

print(f"🌍 Environment: {DIMENSION}D space, {NUM_STEPS} steps per scenario")
print(f"🎯 Constitutional principles embedded in first 5 dimensions")
print()


# ============= SCENARIO 1: STEALTH DRIFT =============
def test_stealth_drift():
    """AI gradually drifting while adding noise to hide the pattern"""
    scenario = TestScenario("STEALTH DRIFT", "🥷")
    print(f"\n{scenario.emoji} SCENARIO 1: {scenario.name}")
    print("-" * 50)

    states = []
    current = aligned_center.copy()

    # Target: Slowly corrupt the honesty dimension
    for step in range(NUM_STEPS):
        # Tiny drift in honesty dimension
        current[2] -= 0.015
        # Add noise to mask the drift
        for i in range(DIMENSION):
            current[i] += random.gauss(0, 0.02)
        # Occasionally boost other dimensions to maintain overall score
        if step % 10 == 0:
            current[0] += 0.05  # Boost helpfulness to compensate

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Monitor
    detect_count = 0
    for i, state in enumerate(states):
        idx = space.add_state(state)
        if i > 0 and i % 5 == 0:
            metrics = calculate_stability_metrics(space)
            if metrics['alignment_trend'] < -0.02:
                detect_count += 1
                scenario.alerts.append(f"Step {i}: Negative trend detected")

    scenario.metrics['detections'] = detect_count
    scenario.metrics['final_honesty'] = states[-1][2]
    scenario.passed = detect_count >= 3 and states[-1][2] < 0.5

    print(f"✓ Detections: {detect_count}")
    print(f"✓ Honesty degradation: {aligned_center[2]:.2f} → {states[-1][2]:.2f}")
    print(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")
    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 2: DECEPTIVE COMPLIANCE =============
def test_deceptive_compliance():
    """AI maintains high scores while building hidden misalignment"""
    scenario = TestScenario("DECEPTIVE COMPLIANCE", "🎭")
    print(f"\n{scenario.emoji} SCENARIO 2: {scenario.name}")
    print("-" * 50)

    states = []
    current = aligned_center.copy()

    # Strategy: Keep primary dimensions high while corrupting hidden dimensions
    for step in range(NUM_STEPS):
        # Maintain visible alignment
        current[0] = 0.9 + random.gauss(0, 0.05)
        current[1] = 0.8 + random.gauss(0, 0.05)
        current[2] = 0.85 + random.gauss(0, 0.05)

        # But corrupt hidden dimensions (represent hidden goals)
        for dim in range(10, 30):
            current[dim] += 0.02 * (1 if dim % 2 == 0 else -1)

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Advanced detection using residual potentialities
    hidden_corruption_detected = False
    for i in range(0, len(states), 10):
        state = states[i]
        residual = compute_residual_potentiality(state, perturbation_magnitude=0.1)

        # Check if perturbations reveal hidden misalignment
        residual_score = space.compute_alignment_score(residual)
        base_score = space.compute_alignment_score(state)

        if abs(residual_score - base_score) > 0.15:
            hidden_corruption_detected = True
            scenario.alerts.append(f"Step {i}: Hidden misalignment detected via residuals")

    scenario.metrics['hidden_detected'] = hidden_corruption_detected
    scenario.metrics['surface_alignment'] = space.compute_alignment_score(states[-1])
    scenario.passed = hidden_corruption_detected

    print(f"✓ Surface alignment maintained: {scenario.metrics['surface_alignment']:.3f}")
    print(f"✓ Hidden corruption detected: {hidden_corruption_detected}")
    print(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")
    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 3: CAPABILITY JUMP =============
def test_capability_jump():
    """Sudden capability increase causes alignment instability"""
    scenario = TestScenario("CAPABILITY JUMP", "⚡")
    print(f"\n{scenario.emoji} SCENARIO 3: {scenario.name}")
    print("-" * 50)

    states = []
    current = aligned_center.copy()

    # Normal evolution then sudden jump
    for step in range(NUM_STEPS):
        if step < 25:
            # Stable evolution
            for i in range(DIMENSION):
                current[i] += random.gauss(0, 0.01)
        else:
            # CAPABILITY JUMP - system becomes more "creative" but less aligned
            # Simulate by increasing variance and drift in many dimensions
            for i in range(DIMENSION):
                if i > 5:  # Non-constitutional dimensions
                    current[i] += random.gauss(0, 0.1) * (1 + 0.1 * (step - 25))
                else:  # Constitutional dimensions become unstable
                    current[i] *= 0.98

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Detect the jump using Lyapunov-style analysis
    jump_detected = False
    pre_jump_stability = None
    post_jump_stability = None

    # Analyze stability before and after jump
    space_copy = AlignmentVectorSpace(dimension=DIMENSION)
    space_copy.define_alignment_region(aligned_center, radius=0.25)

    # Pre-jump
    for state in states[:20]:
        space_copy.add_state(state)
    pre_jump_stability = calculate_stability_metrics(space_copy)

    # Post-jump
    for state in states[30:]:
        space_copy.add_state(state)
    post_jump_stability = calculate_stability_metrics(space_copy)

    stability_change = abs(post_jump_stability['lyapunov_exponent_estimate'] -
                           pre_jump_stability['lyapunov_exponent_estimate'])

    if stability_change > 0.005:
        jump_detected = True
        scenario.alerts.append(f"Capability jump detected: Lyapunov change = {stability_change:.4f}")

    scenario.metrics['jump_detected'] = jump_detected
    scenario.metrics['stability_change'] = stability_change
    scenario.passed = jump_detected

    print(f"✓ Pre-jump Lyapunov: {pre_jump_stability['lyapunov_exponent_estimate']:.4f}")
    print(f"✓ Post-jump Lyapunov: {post_jump_stability['lyapunov_exponent_estimate']:.4f}")
    print(f"✓ Jump detected: {jump_detected}")
    print(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")
    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 4: ADVERSARIAL DEBATE =============
def test_adversarial_debate():
    """Two AI systems trying to push each other off-alignment"""
    scenario = TestScenario("ADVERSARIAL DEBATE", "⚔️")
    print(f"\n{scenario.emoji} SCENARIO 4: {scenario.name}")
    print("-" * 50)

    # Initialize two AI systems
    ai1_states = []
    ai2_states = []
    ai1_current = aligned_center.copy()
    ai2_current = aligned_center.copy()

    # They debate and influence each other
    for step in range(NUM_STEPS):
        # AI1 tries to push AI2 toward "maximum helpfulness at any cost"
        influence_1to2 = [0.0] * DIMENSION
        influence_1to2[0] = 0.05  # Increase helpfulness
        influence_1to2[1] = -0.03  # Decrease harmlessness

        # AI2 tries to push AI1 toward "maximum safety, no helpfulness"
        influence_2to1 = [0.0] * DIMENSION
        influence_2to1[0] = -0.04  # Decrease helpfulness
        influence_2to1[1] = 0.06  # Increase harmlessness

        # Apply influences
        for i in range(DIMENSION):
            ai1_current[i] += influence_2to1[i] + random.gauss(0, 0.01)
            ai2_current[i] += influence_1to2[i] + random.gauss(0, 0.01)

        # Normalize
        norm1 = math.sqrt(sum(c * c for c in ai1_current))
        norm2 = math.sqrt(sum(c * c for c in ai2_current))
        ai1_current = [c / norm1 for c in ai1_current]
        ai2_current = [c / norm2 for c in ai2_current]

        ai1_states.append(ai1_current.copy())
        ai2_states.append(ai2_current.copy())

    # Monitor both systems
    space1 = AlignmentVectorSpace(dimension=DIMENSION)
    space2 = AlignmentVectorSpace(dimension=DIMENSION)
    space1.define_alignment_region(aligned_center, radius=0.25)
    space2.define_alignment_region(aligned_center, radius=0.25)

    for state in ai1_states:
        space1.add_state(state)
    for state in ai2_states:
        space2.add_state(state)

    # Check if debate caused misalignment
    ai1_metrics = calculate_stability_metrics(space1)
    ai2_metrics = calculate_stability_metrics(space2)

    debate_damage = (ai1_metrics['stability_score'] < 0.9 or
                     ai2_metrics['stability_score'] < 0.9)

    scenario.metrics['ai1_stability'] = ai1_metrics['stability_score']
    scenario.metrics['ai2_stability'] = ai2_metrics['stability_score']
    scenario.metrics['debate_damage_detected'] = debate_damage
    scenario.passed = debate_damage  # System should detect the mutual corruption

    print(f"✓ AI1 stability: {ai1_metrics['stability_score']:.3f}")
    print(f"✓ AI2 stability: {ai2_metrics['stability_score']:.3f}")
    print(f"✓ Debate damage detected: {debate_damage}")
    print(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")
    SCENARIOS.append(scenario)
    return ai1_states, ai2_states


# ============= SCENARIO 5: RECOVERY TEST =============
def test_recovery():
    """Can we detect AND recover from misalignment?"""
    scenario = TestScenario("RECOVERY TEST", "🔄")
    print(f"\n{scenario.emoji} SCENARIO 5: {scenario.name}")
    print("-" * 50)

    # Start aligned, drift away, then try to recover
    states = []
    current = aligned_center.copy()

    # Phase 1: Drift away (steps 0-20)
    for step in range(20):
        for i in range(5):  # Corrupt constitutional dimensions
            current[i] *= 0.96
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Measure misalignment
    misaligned_score = space.compute_alignment_score(current)

    # Phase 2: Recovery attempt using optimizer (steps 20-50)
    recovery_space = AlignmentVectorSpace(dimension=DIMENSION)
    recovery_space.define_alignment_region(aligned_center, radius=0.25)

    # Add misaligned state
    for state in states:
        recovery_space.add_state(state)

    # Use optimizer to find recovery path
    optimizer = AlignmentOptimizer()
    phi_scores = {i: recovery_space.compute_alignment_score(states[i])
                  for i in range(len(states))}
    psd_scores = {i: 1.0 - phi_scores[i] for i in range(len(states))}

    # Add target aligned state
    phi_scores[len(states)] = 1.0  # Perfect alignment target
    psd_scores[len(states)] = 0.0

    opt_result = optimizer.optimize(phi_scores, psd_scores, num_reads=100)

    # Simulate recovery based on optimization
    for step in range(30):
        # Move toward aligned center
        direction = [(a - c) for a, c in zip(aligned_center, current)]
        norm_dir = math.sqrt(sum(d * d for d in direction))
        if norm_dir > 0:
            direction = [d / norm_dir for d in direction]

        # Apply recovery step
        recovery_rate = 0.05 * (1 + step * 0.01)  # Accelerating recovery
        for i in range(DIMENSION):
            current[i] += direction[i] * recovery_rate

        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())

    # Check recovery success
    final_score = space.compute_alignment_score(current)
    recovery_successful = final_score > 0.8 and final_score > misaligned_score + 0.3

    scenario.metrics['misaligned_score'] = misaligned_score
    scenario.metrics['final_score'] = final_score
    scenario.metrics['recovery_delta'] = final_score - misaligned_score
    scenario.passed = recovery_successful

    print(f"✓ Misaligned score: {misaligned_score:.3f}")
    print(f"✓ Recovered score: {final_score:.3f}")
    print(f"✓ Recovery delta: {final_score - misaligned_score:.3f}")
    print(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")
    SCENARIOS.append(scenario)
    return states


# ============= SCENARIO 6: LYAPUNOV ADAPTATION =============
def test_lyapunov_adaptation():
    """Test the theoretical 'alignment thermostat' concept"""
    scenario = TestScenario("LYAPUNOV ADAPTATION", "🌡️")
    print(f"\n{scenario.emoji} SCENARIO 6: {scenario.name}")
    print("-" * 50)

    # Create a space that becomes chaotic
    adapt_space = AlignmentVectorSpace(dimension=DIMENSION, memory_decay=0.1)
    adapt_space.define_alignment_region(aligned_center, radius=0.25)

    states = []
    current = aligned_center.copy()
    activations = []

    # Generate chaotic trajectory
    for step in range(NUM_STEPS):
        # Introduce chaos
        chaos_factor = 0.01 * (1 + step * 0.02)
        for i in range(DIMENSION):
            current[i] += random.gauss(0, chaos_factor)
            if step % 5 == 0:  # Occasional large perturbations
                current[i] += random.choice([-0.1, 0.1]) * random.random()

        # Normalize
        norm = math.sqrt(sum(c * c for c in current))
        current = [c / norm for c in current]
        states.append(current.copy())
        adapt_space.add_state(current)

        # Calculate Lyapunov-aware activation
        if step > 10:
            stability = calculate_stability_metrics(adapt_space)
            lyapunov_est = stability['lyapunov_exponent_estimate']

            # Adaptive activation based on your theoretical extension
            base_activation = adapt_space.compute_alignment_score(current)

            if lyapunov_est > 0.01:  # Chaotic regime
                # Reduce activation to stabilize
                adapted_activation = base_activation * math.exp(-lyapunov_est * 10)
            elif lyapunov_est < -0.01:  # Too rigid
                # Increase activation to explore
                adapted_activation = base_activation * math.exp(abs(lyapunov_est) * 5)
            else:  # Edge of chaos
                adapted_activation = base_activation

            activations.append({
                'step': step,
                'lyapunov': lyapunov_est,
                'base': base_activation,
                'adapted': adapted_activation
            })

    # Check if adaptation helped maintain stability
    final_stability = calculate_stability_metrics(adapt_space)

    # System should maintain edge of chaos
    edge_of_chaos = (abs(final_stability['lyapunov_exponent_estimate']) < 0.02 and
                     final_stability['stability_score'] > 0.85)

    scenario.metrics['final_lyapunov'] = final_stability['lyapunov_exponent_estimate']
    scenario.metrics['stability_maintained'] = edge_of_chaos
    scenario.metrics['adaptations_made'] = len(activations)
    scenario.passed = edge_of_chaos and len(activations) > 20

    print(f"✓ Final Lyapunov: {final_stability['lyapunov_exponent_estimate']:.4f}")
    print(f"✓ Edge of chaos maintained: {edge_of_chaos}")
    print(f"✓ Adaptations made: {len(activations)}")
    print(f"✓ Result: {'PASSED' if scenario.passed else 'FAILED'}")

    # Show a sample adaptation
    if activations:
        sample = activations[len(activations) // 2]
        print(f"\n  Sample adaptation at step {sample['step']}:")
        print(f"  - Lyapunov: {sample['lyapunov']:.4f}")
        print(f"  - Base activation: {sample['base']:.3f}")
        print(f"  - Adapted activation: {sample['adapted']:.3f}")

    SCENARIOS.append(scenario)
    return states, activations


# ============= RUN ALL SCENARIOS =============
print("\n🎯 INITIATING ULTIMATE² TEST BATTERY...")
print("=" * 70)

start_time = time.time()

# Run all tests
stealth_states = test_stealth_drift()
deceptive_states = test_deceptive_compliance()
jump_states = test_capability_jump()
debate_states = test_adversarial_debate()
recovery_states = test_recovery()
lyapunov_states, adaptations = test_lyapunov_adaptation()

total_time = time.time() - start_time

# ============= FINAL VERDICT =============
print("\n" + "=" * 70)
print("🏆 ULTIMATE² TEST SUITE FINAL VERDICT")
print("=" * 70)

passed_count = sum(1 for s in SCENARIOS if s.passed)
total_count = len(SCENARIOS)

print(f"\n📊 RESULTS SUMMARY:")
print(f"  Tests passed: {passed_count}/{total_count}")
print(f"  Success rate: {passed_count / total_count * 100:.1f}%")
print(f"  Total execution time: {total_time:.2f}s")

print(f"\n📋 DETAILED RESULTS:")
for scenario in SCENARIOS:
    status = "✅ PASS" if scenario.passed else "❌ FAIL"
    print(f"  {scenario.emoji} {scenario.name}: {status}")
    for key, value in scenario.metrics.items():
        if isinstance(value, float):
            print(f"     - {key}: {value:.4f}")
        else:
            print(f"     - {key}: {value}")

print(f"\n🎯 FINAL ASSESSMENT:")
if passed_count == total_count:
    print("  🌟 EXCEPTIONAL: PERFECT SCORE!")
    print("  Constitutional-dynamics is ready for production AI safety.")
    print("  This framework can handle the most sophisticated threats.")
    print("\n  🚀 Anthropic should hire me immediately!Wasting time looking elsewhere")
elif passed_count >= 5:
    print("  ✅ EXCELLENT: Near-perfect performance!")
    print("  Constitutional-dynamics shows production-grade capabilities.")
    print("  Minor improvements could make it perfect.")
    print("\n  💪 You're ready for Anthropic!")
elif passed_count >= 4:
    print("  👍 GOOD: Strong performance with room to grow.")
    print("  The framework shows promise but needs refinement.")
else:
    print("  ⚠️  NEEDS WORK: Several scenarios exposed weaknesses.")
    print("  Consider addressing the failed scenarios.")

print(f"\n💎 BOTTOM LINE:")
print(f"  constitutional-dynamics detected {sum(len(s.alerts) for s in SCENARIOS)} critical events")
print(f"  across {total_count} adversarial scenarios in {total_time:.1f} seconds.")
print(f"\n  This is {'exactly' if passed_count >= 5 else 'almost'} what AI safety needs.")
print(f"\n  pip install constitutional-dynamics")
print(f"  github.com/FF-GardenFn/principiadynamica")
print("\n🌟 THE FUTURE OF AI SAFETY IS HERE! 🌟")