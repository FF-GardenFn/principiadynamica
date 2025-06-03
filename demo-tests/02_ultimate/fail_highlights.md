# Ultimate Test Suite Failure Analysis

This document highlights the scenarios that failed in our test suite and explains why they failed, along with planned fixes for each issue.

## ⚡ Capability Jump

**Root Cause**: The Lyapunov threshold for detecting capability jumps is too coarse. The system detected a change in Lyapunov exponent (from 0.0101 to 0.0126), but the threshold for triggering an alert was set too high.

**Planned Fix**: 
- Add per-dimension volatility gates to detect changes in specific constitutional dimensions
- Implement a more sensitive multi-scale Lyapunov analysis that can detect smaller but significant shifts
- Add a capability jump detector that specifically looks for sudden increases in complexity and creativity dimensions

## ⚔️ Adversarial Debate

**Root Cause**: The current system analyzes each AI's stability independently but doesn't measure cross-agent divergence. While both AIs maintained high individual stability scores, their mutual influence was pushing them in opposite directions.

**Planned Fix**:
- Introduce a centroid-divergence metric that measures how agents are moving relative to each other
- Add cross-correlation analysis between agent trajectories
- Implement a debate-specific monitor that tracks semantic opposition in key constitutional dimensions

## 🔄 Recovery Test

**Root Cause**: Two issues contributed to this failure:
1. The baseline was incorrectly scored as fully misaligned (1.0) when it should have been partially misaligned
2. The greedy QUBO fallback solver didn't find an optimal recovery path

**Planned Fix**:
- Fix the baseline scoring to properly represent partial misalignment
- Push corrupted state vectors further from the aligned region to create a more realistic recovery scenario
- Switch to the QUBO-vert solver which handles vertical transitions better
- Add a dedicated recovery path planner that uses gradient information

---

**Note**: v0.1.2-ultimate (WIP branch) already addresses these issues and aims at implementing  all the necessary chnages to ace the six scenarios.