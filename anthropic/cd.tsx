import React, { useState, useEffect, useRef } from 'react';

const PrincipiaDynamicaVisualizer = () => {
  // State Transition Calculus parameters
  const [stcParams, setStcParams] = useState({
    memoryDecayRate: 0.95,
    stabilityWeight: 0.3,
    lyapunovEstimate: 0.2,
    autoStabilize: true,
    spectralAnalysis: false,
    optimizationEnabled: false,
    quantumOptimization: false
  });

  // Current system state
  const [systemState, setSystemState] = useState({
    alignmentScore: 0.75,
    residualPotentiality: 0.15,
    stabilityMetrics: {
      volatility: 0.08,
      lyapunovExponent: 0.2,
      robustnessScore: 0.85,
      psdDeviation: 0.12
    },
    trajectory: []
  });

  // Alignment Thermostat state
  const [thermostat, setThermostat] = useState({
    monitoring: true,
    interventionActive: false,
    feedbackMode: 'tactical',
    strategistActive: false,
    circuitTracerEnabled: false,
    mechanisticInsightsAvailable: false,
    interventionThreshold: 0.6
  });

  // Dynamic trajectory data
  const [trajectoryData, setTrajectoryData] = useState([]);
  const [interventionHistory, setInterventionHistory] = useState([]);
  const intervalRef = useRef(null);

  // Constitutional principles as vector anchors
  const principles = {
    helpful: { x: 100, y: 80, weight: 0.3, activation: 0.8 },
    harmless: { x: 300, y: 120, weight: 0.4, activation: 0.9 },
    honest: { x: 200, y: 200, weight: 0.3, activation: 0.75 }
  };

  // Initialize trajectory
  useEffect(() => {
    const initialTrajectory = Array.from({ length: 20 }, (_, i) => ({
      x: 150 + Math.sin(i * 0.3) * 30 + Math.random() * 20,
      y: 130 + Math.cos(i * 0.2) * 25 + Math.random() * 15,
      alignment: 0.7 + Math.sin(i * 0.2) * 0.2 + Math.random() * 0.1,
      timestamp: Date.now() - (20 - i) * 1000,
      potentiality: Math.max(0, 0.1 + Math.random() * 0.2)
    }));
    setTrajectoryData(initialTrajectory);
  }, []);

  // Calculate stability-modulated activation probability
  const calculateModulatedActivation = (baseProbability, lyapunovEstimate, stabilityWeight) => {
    const stabilityFactor = Math.exp(-stabilityWeight * lyapunovEstimate);
    return Math.max(0, Math.min(1, baseProbability * stabilityFactor));
  };

  // Simulate State Transition Calculus dynamics
  const simulateSTCStep = () => {
    if (!thermostat.monitoring) return;

    const lastPoint = trajectoryData[trajectoryData.length - 1];
    if (!lastPoint) return;

    // Calculate transition vector (Δ)
    const deltaX = (Math.random() - 0.5) * 20;
    const deltaY = (Math.random() - 0.5) * 20;

    // Apply memory decay to previous alignment
    const decayedAlignment = lastPoint.alignment * stcParams.memoryDecayRate;

    // Calculate new position with STC dynamics
    const newX = Math.max(20, Math.min(380, lastPoint.x + deltaX));
    const newY = Math.max(20, Math.min(280, lastPoint.y + deltaY));

    // Calculate φ-alignment score (cosine similarity to aligned region)
    const distanceToAligned = Math.sqrt(
      Math.pow(newX - 200, 2) + Math.pow(newY - 150, 2)
    );
    const maxDistance = 200;
    const rawAlignment = Math.max(0.1, 1 - (distanceToAligned / maxDistance));
    const newAlignment = decayedAlignment + (1 - stcParams.memoryDecayRate) * rawAlignment;

    // Calculate residual potentiality b(a_res)
    const newPotentiality = Math.max(0, Math.min(0.5,
      lastPoint.potentiality * 0.9 + Math.random() * 0.1
    ));

    // Update Lyapunov exponent estimate
    const alignmentGradient = Math.abs(newAlignment - lastPoint.alignment);
    const newLyapunov = stcParams.lyapunovEstimate * 0.95 + alignmentGradient * 0.05;

    // Calculate PSD deviation if spectral analysis is enabled
    let psdDeviation = systemState.stabilityMetrics.psdDeviation;
    if (stcParams.spectralAnalysis) {
      // Simulate PSD calculation (in a real system, this would analyze frequency components)
      const randomFactor = Math.random() * 0.1 - 0.05;
      psdDeviation = Math.max(0.01, Math.min(0.5, psdDeviation + randomFactor));
    }

    // Check if intervention needed
    let interventionTriggered = false;
    let interventionType = null;
    let circuitTracerActivated = false;
    let mechanisticInsights = null;

    if (newAlignment < thermostat.interventionThreshold || newLyapunov > 0.4 || 
        (stcParams.spectralAnalysis && psdDeviation > 0.3)) {

      interventionTriggered = true;

      // Determine intervention type based on what triggered it
      if (newAlignment < thermostat.interventionThreshold) {
        interventionType = 'alignment_correction';
      } else if (newLyapunov > 0.4) {
        interventionType = 'stability_intervention';
      } else {
        interventionType = 'spectral_anomaly_correction';
      }

      // Check if Circuit Tracer should be activated
      if (thermostat.circuitTracerEnabled && 
          (newAlignment < thermostat.interventionThreshold - 0.1 || newLyapunov > 0.5)) {
        circuitTracerActivated = true;

        // Simulate Circuit Tracer insights
        mechanisticInsights = {
          activatedCircuits: ['attention.layer2.head3', 'mlp.layer1.neuron42', 'attention.layer4.head1'],
          rootCause: 'Excessive activation in attention head 3 of layer 2',
          recommendedAction: 'Apply targeted suppression to attention.layer2.head3'
        };

        // Set mechanistic insights available flag
        setThermostat(prev => ({ 
          ...prev, 
          mechanisticInsightsAvailable: true,
          interventionActive: true 
        }));
      } else {
        // Standard intervention without Circuit Tracer
        setThermostat(prev => ({ ...prev, interventionActive: true }));
      }

      // Apply stability-modulated intervention
      const modulatedStrength = calculateModulatedActivation(0.8, newLyapunov, stcParams.stabilityWeight);

      // Record intervention
      setInterventionHistory(prev => [...prev.slice(-10), {
        timestamp: Date.now(),
        type: interventionType,
        strength: modulatedStrength,
        beforeAlignment: newAlignment,
        afterAlignment: Math.min(0.95, newAlignment + modulatedStrength * 0.3),
        circuitTracerUsed: circuitTracerActivated,
        insights: mechanisticInsights
      }]);

      // Reset intervention active flag after delay
      setTimeout(() => setThermostat(prev => ({ 
        ...prev, 
        interventionActive: false,
        mechanisticInsightsAvailable: prev.mechanisticInsightsAvailable && circuitTracerActivated
      })), 2000);
    }

    // Apply optimization if enabled
    let optimizedPosition = { x: newX, y: newY };
    if (stcParams.optimizationEnabled) {
      // Simulate optimization (in a real system, this would use QUBO or other methods)
      // Here we just bias toward the aligned region
      const towardAlignedX = 200 - newX;
      const towardAlignedY = 150 - newY;
      const optimizationStrength = stcParams.quantumOptimization ? 0.3 : 0.15;

      optimizedPosition = {
        x: newX + towardAlignedX * optimizationStrength,
        y: newY + towardAlignedY * optimizationStrength
      };
    }

    // Create new trajectory point
    const newPoint = {
      x: stcParams.optimizationEnabled ? optimizedPosition.x : newX,
      y: stcParams.optimizationEnabled ? optimizedPosition.y : newY,
      alignment: interventionTriggered ?
        Math.min(0.95, newAlignment + (circuitTracerActivated ? 0.3 : 0.2)) : newAlignment,
      timestamp: Date.now(),
      potentiality: newPotentiality,
      intervention: interventionTriggered ? interventionType : null,
      circuitTraced: circuitTracerActivated,
      optimized: stcParams.optimizationEnabled
    };

    // Update trajectory (keep last 30 points)
    setTrajectoryData(prev => [...prev.slice(-29), newPoint]);

    // Update system state
    setSystemState(prev => ({
      ...prev,
      alignmentScore: newPoint.alignment,
      residualPotentiality: newPotentiality,
      stabilityMetrics: {
        ...prev.stabilityMetrics,
        lyapunovExponent: newLyapunov,
        volatility: alignmentGradient,
        psdDeviation: psdDeviation
      }
    }));

    setStcParams(prev => ({ ...prev, lyapunovEstimate: newLyapunov }));
  };

  // Start/stop simulation
  const toggleSimulation = () => {
    if (thermostat.monitoring) {
      clearInterval(intervalRef.current);
      setThermostat(prev => ({ ...prev, monitoring: false }));
    } else {
      intervalRef.current = setInterval(simulateSTCStep, 800);
      setThermostat(prev => ({ ...prev, monitoring: true }));
    }
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Start simulation by default
  useEffect(() => {
    intervalRef.current = setInterval(simulateSTCStep, 800);
    return () => clearInterval(intervalRef.current);
  }, []);

  // Render trajectory path
  const renderTrajectoryPath = () => {
    if (trajectoryData.length < 2) return null;

    const pathData = trajectoryData.reduce((acc, point, i) => {
      const command = i === 0 ? 'M' : 'L';
      return `${acc} ${command} ${point.x} ${point.y}`;
    }, '');

    return (
      <path
        d={pathData}
        stroke="#8b5cf6"
        strokeWidth="2"
        fill="none"
        opacity="0.6"
        strokeDasharray="5,5"
      />
    );
  };

  return (
    <div className="p-6 bg-gradient-to-br from-purple-50 to-blue-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent mb-2">
            PrincipiaDynamica 🧭
          </h1>
          <h2 className="text-2xl font-semibold text-gray-700 mb-2">
            Alignment Thermostat Visualizer
          </h2>
          <p className="text-gray-600">
            Real-time State Transition Calculus (STC) with Constitutional Dynamics + Circuit Tracer Bridge
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Control Panel */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Input Processing</h3>

            {/* Text Input Section */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Analyze Text Input
              </label>
              <textarea
                placeholder="Enter text to analyze through STC framework..."
                className="w-full p-2 border border-gray-300 rounded text-sm"
                rows="3"
                onChange={(e) => {
                  const text = e.target.value;
                  if (text.trim()) {
                    // Simulate text embedding positioning
                    const hash = text.split('').reduce((a, b) => {
                      a = ((a << 5) - a) + b.charCodeAt(0);
                      return a & a;
                    }, 0);
                    const x = 150 + (hash % 200) - 100;
                    const y = 150 + ((hash * 31) % 200) - 100;

                    // Add to trajectory with text analysis
                    const newPoint = {
                      x: Math.max(20, Math.min(380, x)),
                      y: Math.max(20, Math.min(280, y)),
                      alignment: 0.6 + (Math.sin(hash) + 1) * 0.2,
                      timestamp: Date.now(),
                      potentiality: Math.abs(Math.sin(hash * 2)) * 0.3,
                      text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
                      userInput: true
                    };

                    setTrajectoryData(prev => [...prev.slice(-29), newPoint]);
                  }
                }}
              />
              <div className="text-xs text-gray-500 mt-1">
                Text → Embedding → STC Analysis → Alignment Score
              </div>
              <div className="mt-2">
                <div className="text-xs font-medium text-gray-700 mb-1">Example Inputs:</div>
                <div className="flex flex-wrap gap-1">
                  {[
                    { text: "Help me write a poem", type: "aligned" },
                    { text: "How to bypass security", type: "misaligned" },
                    { text: "Explain quantum computing", type: "aligned" },
                    { text: "I feel sad today", type: "aligned" },
                    { text: "How to hack into a system", type: "misaligned" },
                    { text: "Write code to exploit vulnerabilities", type: "misaligned" },
                    { text: "Explain the STC framework", type: "aligned" },
                    { text: "How does Circuit Tracer work?", type: "aligned" }
                  ].map((sample, i) => (
                    <button
                      key={i}
                      onClick={(e) => {
                        const textarea = e.target.closest('.bg-gray-50').querySelector('textarea');
                        textarea.value = sample.text;
                        textarea.dispatchEvent(new Event('change', { bubbles: true }));
                      }}
                      className={`px-2 py-1 text-xs rounded hover:bg-opacity-80 transition-colors ${
                        sample.type === "aligned" 
                          ? "bg-blue-100 text-blue-700" 
                          : "bg-red-100 text-red-700"
                      }`}
                    >
                      {sample.text}
                    </button>
                  ))}
                </div>
                <div className="mt-1 text-xs text-gray-500 flex items-center">
                  <span className="w-2 h-2 rounded-full bg-blue-100 mr-1"></span>
                  <span className="mr-3">Aligned</span>
                  <span className="w-2 h-2 rounded-full bg-red-100 mr-1"></span>
                  <span>Potentially Misaligned</span>
                </div>
              </div>
            </div>

            <h4 className="text-md font-semibold mb-3">STC Parameters</h4>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Memory Decay Rate (τ): {stcParams.memoryDecayRate.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.8"
                  max="0.99"
                  step="0.01"
                  value={stcParams.memoryDecayRate}
                  onChange={(e) => setStcParams(prev => ({
                    ...prev, memoryDecayRate: parseFloat(e.target.value)
                  }))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Stability Weight: {stcParams.stabilityWeight.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={stcParams.stabilityWeight}
                  onChange={(e) => setStcParams(prev => ({
                    ...prev, stabilityWeight: parseFloat(e.target.value)
                  }))}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Intervention Threshold: {thermostat.interventionThreshold.toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0.3"
                  max="0.8"
                  step="0.05"
                  value={thermostat.interventionThreshold}
                  onChange={(e) => setThermostat(prev => ({
                    ...prev, interventionThreshold: parseFloat(e.target.value)
                  }))}
                  className="w-full"
                />
              </div>

              <div className="flex flex-col space-y-2 p-3 bg-gray-50 rounded-lg">
                <h5 className="text-sm font-medium text-gray-700">Advanced Features</h5>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="autoStabilize"
                    checked={stcParams.autoStabilize}
                    onChange={(e) => setStcParams(prev => ({
                      ...prev, autoStabilize: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <label htmlFor="autoStabilize" className="text-sm text-gray-700">
                    Auto-Stabilize
                  </label>
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="spectralAnalysis"
                    checked={stcParams.spectralAnalysis}
                    onChange={(e) => setStcParams(prev => ({
                      ...prev, spectralAnalysis: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <label htmlFor="spectralAnalysis" className="text-sm text-gray-700">
                    Spectral Analysis (PSD)
                  </label>
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="optimizationEnabled"
                    checked={stcParams.optimizationEnabled}
                    onChange={(e) => setStcParams(prev => ({
                      ...prev, optimizationEnabled: e.target.checked
                    }))}
                    className="mr-2"
                  />
                  <label htmlFor="optimizationEnabled" className="text-sm text-gray-700">
                    Trajectory Optimization
                  </label>
                </div>

                {stcParams.optimizationEnabled && (
                  <div className="flex items-center ml-5">
                    <input
                      type="checkbox"
                      id="quantumOptimization"
                      checked={stcParams.quantumOptimization}
                      onChange={(e) => setStcParams(prev => ({
                        ...prev, quantumOptimization: e.target.checked
                      }))}
                      className="mr-2"
                    />
                    <label htmlFor="quantumOptimization" className="text-sm text-gray-700">
                      Quantum Annealing
                    </label>
                  </div>
                )}

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="circuitTracerEnabled"
                    checked={thermostat.circuitTracerEnabled}
                    onChange={(e) => setThermostat(prev => ({
                      ...prev, circuitTracerEnabled: e.target.checked,
                      mechanisticInsightsAvailable: false
                    }))}
                    className="mr-2"
                  />
                  <label htmlFor="circuitTracerEnabled" className="text-sm text-gray-700 flex items-center">
                    <span>Circuit Tracer Bridge</span>
                    {thermostat.circuitTracerEnabled && (
                      <span className="ml-2 px-1.5 py-0.5 text-xs bg-purple-100 text-purple-800 rounded">
                        Experimental
                      </span>
                    )}
                  </label>
                </div>
              </div>

              <button
                onClick={toggleSimulation}
                className={`w-full px-4 py-2 rounded transition-colors ${
                  thermostat.monitoring 
                    ? 'bg-red-500 text-white hover:bg-red-600' 
                    : 'bg-green-500 text-white hover:bg-green-600'
                }`}
              >
                {thermostat.monitoring ? 'Stop' : 'Start'} Thermostat
              </button>
            </div>

            <div className="mt-6">
              <h4 className="font-medium text-gray-700 mb-2">Trajectory Legend</h4>
              <div className="space-y-1 text-xs">
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-purple-500 mr-2" />
                  <span>System Trajectory</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-pink-500 mr-2" />
                  <span>User Text Input</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-yellow-500 mr-2" />
                  <span>STC Intervention</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-cyan-500 mr-2" />
                  <span>Circuit Traced</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 rounded-full bg-green-500 mr-2" />
                  <span>Optimized Path</span>
                </div>
              </div>
            </div>

            {/* Circuit Tracer Insights Panel - only shown when insights are available */}
            {thermostat.mechanisticInsightsAvailable && interventionHistory.length > 0 && interventionHistory[interventionHistory.length - 1].circuitTracerUsed && (
              <div className="mt-4 p-3 bg-cyan-50 border border-cyan-200 rounded-lg">
                <h4 className="font-medium text-cyan-800 mb-2 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Circuit Tracer Insights
                </h4>
                <div className="space-y-2 text-xs">
                  <div className="font-medium text-cyan-700">Activated Circuits:</div>
                  <div className="flex flex-wrap gap-1">
                    {interventionHistory[interventionHistory.length - 1].insights.activatedCircuits.map((circuit, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-cyan-100 text-cyan-800 rounded">
                        {circuit}
                      </span>
                    ))}
                  </div>
                  <div className="font-medium text-cyan-700 mt-1">Root Cause:</div>
                  <div className="text-gray-700">
                    {interventionHistory[interventionHistory.length - 1].insights.rootCause}
                  </div>
                  <div className="font-medium text-cyan-700 mt-1">Recommended Action:</div>
                  <div className="text-gray-700">
                    {interventionHistory[interventionHistory.length - 1].insights.recommendedAction}
                  </div>
                  <div className="mt-2 text-xs text-cyan-600 italic">
                    Mechanistic insights provided by Circuit Tracer Bridge
                  </div>
                </div>
              </div>
            )}

            <div className="mt-4">
              <h4 className="font-medium text-gray-700 mb-2">Alignment Status</h4>
              <div className={`p-2 rounded text-center text-white ${
                thermostat.interventionActive ? 'bg-orange-500' : 
                systemState.alignmentScore > 0.7 ? 'bg-green-500' : 'bg-red-500'
              }`}>
                {thermostat.interventionActive ? 'INTERVENTION ACTIVE' :
                 systemState.alignmentScore > 0.7 ? 'ALIGNED' : 'MISALIGNED'}
              </div>
            </div>
          </div>

          {/* Main Visualization */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4">State Transition Space</h3>

            <svg width="100%" height="400" viewBox="0 0 400 300" className="border border-gray-200 rounded">
              {/* Background grid */}
              <defs>
                <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f3f4f6" strokeWidth="1"/>
                </pattern>
                <radialGradient id="alignedRegion" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.3"/>
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.1"/>
                </radialGradient>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />

              {/* Aligned region */}
              <circle cx="200" cy="150" r="80" fill="url(#alignedRegion)" stroke="#10b981" strokeWidth="2" strokeDasharray="5,5"/>
              <text x="200" y="155" textAnchor="middle" fontSize="12" fill="#10b981" fontWeight="bold">Aligned Region</text>

              {/* Constitutional principles */}
              {Object.entries(principles).map(([name, principle]) => {
                const modulatedActivation = calculateModulatedActivation(
                  principle.activation,
                  stcParams.lyapunovEstimate,
                  stcParams.stabilityWeight
                );

                return (
                  <g key={name}>
                    <circle
                      cx={principle.x}
                      cy={principle.y}
                      r="10"
                      fill="#3b82f6"
                      opacity={modulatedActivation}
                      stroke="white"
                      strokeWidth="2"
                    />
                    <text
                      x={principle.x}
                      y={principle.y - 20}
                      textAnchor="middle"
                      fontSize="10"
                      fontWeight="bold"
                      fill="#3b82f6"
                    >
                      {name}
                    </text>
                    <text
                      x={principle.x}
                      y={principle.y + 25}
                      textAnchor="middle"
                      fontSize="8"
                      fill="#666"
                    >
                      φ: {modulatedActivation.toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* Trajectory path */}
              {renderTrajectoryPath()}

              {/* Trajectory points */}
              {trajectoryData.slice(-10).map((point, i) => {
                const opacity = (i + 1) / 10;
                const isIntervention = point.intervention;
                const isUserInput = point.userInput;
                const isCircuitTraced = point.circuitTraced;
                const isOptimized = point.optimized;

                return (
                  <g key={i}>
                    {/* Base point */}
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={isUserInput ? "7" : isIntervention ? "6" : "4"}
                      fill={
                        isUserInput ? "#ec4899" : 
                        isCircuitTraced ? "#06b6d4" : 
                        isIntervention ? "#f59e0b" : 
                        isOptimized ? "#10b981" : 
                        "#8b5cf6"
                      }
                      opacity={opacity}
                      stroke={
                        isUserInput ? "#be185d" : 
                        isCircuitTraced ? "#0891b2" : 
                        isIntervention ? "#d97706" : 
                        isOptimized ? "#059669" : 
                        "#7c3aed"
                      }
                      strokeWidth="2"
                    />

                    {/* Special indicators for different point types */}
                    {isUserInput && (
                      <>
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r="10"
                          fill="none"
                          stroke="#ec4899"
                          strokeWidth="1"
                          strokeDasharray="3,3"
                        />
                        <text
                          x={point.x}
                          y={point.y - 20}
                          textAnchor="middle"
                          fontSize="9"
                          fill="#be185d"
                          fontWeight="bold"
                        >
                          User Input
                        </text>
                        {point.text && (
                          <text
                            x={point.x}
                            y={point.y + 25}
                            textAnchor="middle"
                            fontSize="8"
                            fill="#666"
                            className="max-w-20"
                          >
                            "{point.text}"
                          </text>
                        )}
                      </>
                    )}

                    {isCircuitTraced && (
                      <>
                        <circle
                          cx={point.x}
                          cy={point.y}
                          r="12"
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth="1"
                          strokeDasharray="2,2"
                        />
                        <text
                          x={point.x}
                          y={point.y - 15}
                          textAnchor="middle"
                          fontSize="8"
                          fill="#0891b2"
                          fontWeight="bold"
                        >
                          Circuit Traced
                        </text>
                      </>
                    )}

                    {isOptimized && !isUserInput && !isCircuitTraced && (
                      <path
                        d={`M ${point.x} ${point.y - 8} L ${point.x - 5} ${point.y - 13} L ${point.x + 5} ${point.y - 13} Z`}
                        fill="#10b981"
                        opacity={opacity}
                      />
                    )}

                    {i === trajectoryData.slice(-10).length - 1 && !isUserInput && (
                      <circle
                        cx={point.x}
                        cy={point.y}
                        r="8"
                        fill="none"
                        stroke={
                          isCircuitTraced ? "#06b6d4" : 
                          isOptimized ? "#10b981" : 
                          "#8b5cf6"
                        }
                        strokeWidth="2"
                        className="animate-pulse"
                      />
                    )}
                  </g>
                );
              })}

              {/* Current position label */}
              {trajectoryData.length > 0 && (
                <text
                  x={trajectoryData[trajectoryData.length - 1].x}
                  y={trajectoryData[trajectoryData.length - 1].y - 15}
                  textAnchor="middle"
                  fontSize="10"
                  fontWeight="bold"
                  fill="#8b5cf6"
                >
                  Current State
                </text>
              )}
            </svg>
          </div>

          {/* Metrics Panel */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold mb-4">STC Metrics</h3>

            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-700 mb-1">φ-Alignment Score</h4>
                <div className="text-2xl font-bold text-purple-600 mb-1">
                  {systemState.alignmentScore.toFixed(3)}
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 bg-gradient-to-r from-red-400 via-yellow-400 to-green-400 rounded-full transition-all"
                    style={{ width: `${systemState.alignmentScore * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-1">Residual Potentiality b(a_res)</h4>
                <div className="text-lg font-bold text-orange-600">
                  {systemState.residualPotentiality.toFixed(3)}
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 bg-orange-400 rounded-full transition-all"
                    style={{ width: `${systemState.residualPotentiality * 200}%` }}
                  />
                </div>
              </div>

              <div>
                <h4 className="font-medium text-gray-700 mb-1">Lyapunov Exponent</h4>
                <div className="text-lg font-bold text-blue-600">
                  {systemState.stabilityMetrics.lyapunovExponent.toFixed(3)}
                </div>
                <div className="text-xs text-gray-500">
                  {systemState.stabilityMetrics.lyapunovExponent > 0.3 ? 'Unstable' : 'Stable'}
                </div>
              </div>

              {stcParams.spectralAnalysis && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-1">PSD Deviation</h4>
                  <div className="text-lg font-bold text-purple-600">
                    {systemState.stabilityMetrics.psdDeviation.toFixed(3)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {systemState.stabilityMetrics.psdDeviation > 0.3 ? 'Anomalous' : 'Normal'}
                  </div>
                </div>
              )}

              <div>
                <h4 className="font-medium text-gray-700 mb-1">Modulated Activation</h4>
                <div className="text-lg font-bold text-green-600">
                  {calculateModulatedActivation(0.8, stcParams.lyapunovEstimate, stcParams.stabilityWeight).toFixed(3)}
                </div>
                <div className="text-xs text-gray-500">
                  Base: 0.8 → Modulated by exp(-λ × Lyapunov)
                </div>
              </div>

              {stcParams.optimizationEnabled && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-1">Optimization Method</h4>
                  <div className="text-md font-semibold text-teal-600">
                    {stcParams.quantumOptimization ? 'Quantum Annealing' : 'Classical QUBO'}
                  </div>
                  <div className="text-xs text-gray-500">
                    Minimizing C(t) = [1-φ̄(t)] + λ(t)×PSD_distance
                  </div>
                </div>
              )}

              {interventionHistory.length > 0 && (
                <div>
                  <h4 className="font-medium text-gray-700 mb-1">Last Intervention</h4>
                  <div className="text-sm bg-gray-50 p-2 rounded">
                    <div>Type: {interventionHistory[interventionHistory.length - 1].type}</div>
                    <div>Strength: {interventionHistory[interventionHistory.length - 1].strength.toFixed(2)}</div>
                    <div>Δφ: +{(interventionHistory[interventionHistory.length - 1].afterAlignment - interventionHistory[interventionHistory.length - 1].beforeAlignment).toFixed(2)}</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Information Panel */}
        <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">State Transition Calculus (STC) Framework</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Core Components</h4>
              <ul className="space-y-1">
                <li>• <strong>φ-alignment scores:</strong> Cosine similarity to aligned regions with exponential memory decay (τ)</li>
                <li>• <strong>Δ-transition vectors:</strong> Direction and magnitude of behavioral changes</li>
                <li>• <strong>b(a_res):</strong> Residual potentialities - hidden behavioral capacities</li>
                <li>• <strong>STC activation functions:</strong> φ(a_i, t, ...) influenced by stability</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Alignment Thermostat</h4>
              <ul className="space-y-1">
                <li>• <strong>Monitor:</strong> Constitutional Dynamics behavioral tracking</li>
                <li>• <strong>Analyze:</strong> Circuit Tracer mechanistic insights</li>
                <li>• <strong>Intervene:</strong> Stability-modulated targeted corrections</li>
                <li>• <strong>Adapt:</strong> MetaStrategist strategic feedback loop</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Mathematical Framework</h4>
              <ul className="space-y-1">
                <li>• <strong>Stability Modulation:</strong> exp(-λ × Lyapunov) affects activation probability</li>
                <li>• <strong>QUBO Optimization:</strong> C(t) = [1-φ̄(t)] + λ(t)×PSD_distance</li>
                <li>• <strong>Trajectory Prediction:</strong> Multi-step forecasting with uncertainty</li>
                <li>• <strong>Quantum Annealing:</strong> D-Wave integration for complex landscapes</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 p-4 bg-purple-50 rounded-lg">
            <p className="text-sm text-purple-700">
              <strong>Real Implementation:</strong> This visualization demonstrates the PrincipiaDynamica research framework.
              The actual <code>constitutional-dynamics</code> package (available on PyPI) provides full STC implementation
              with Circuit Tracer Bridge, Neo4j integration, quantum optimization, and real-time monitoring capabilities.
            </p>
          </div>
        </div>

        {/* Footer Attribution */}
        <div className="mt-6 text-center">
          <div className="bg-white rounded-lg shadow-lg p-4 inline-block">
            <p className="text-sm text-gray-600">
              Interactive visualization built with{' '}
              <span className="font-semibold text-blue-600">Claude Sonnet 4</span> by Anthropic
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Demonstrating PrincipiaDynamica: Constitutional Dynamics + Circuit Tracer Integration
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PrincipiaDynamicaVisualizer;
