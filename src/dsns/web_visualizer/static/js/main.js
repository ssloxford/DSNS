// Main thread: spawns the Web Worker, handles state interpolation,
// and runs the 60 FPS render loop independently of simulation compute.

import WebRenderer from "./renderer.js";

const PRESETS = [
    "ground", "gps", "iridium", "starlink", "cubesat-tle",
    "multi-layer", "leo-leo", "leo-meo", "leo-geo",
    "leo-meo-geo", "earth-moon", "earth-mars", "walker",
];

const DEFAULT_PRESET = "walker";
const DEFAULT_TIME_SCALE = 100.0;
const DEFAULT_SPACE_SCALE = 1e-6;
const DEFAULT_INTERPLANETARY_SCALE = 1e-10;
const DEFAULT_STEP_INTERVAL = 333; // ms: ~3 steps/sec default
let currentStepInterval = DEFAULT_STEP_INTERVAL;

let worker = null;
let renderer = null;
let stateA = null;       // older state
let stateB = null;       // newer state
let stateBReceivedAt = 0;
let isAnimating = false; // render loop running
let simulationActive = false; // worker producing new states
let lastFrameTime = 0;
let animFrameId = null;
let frameCount = 0;
let lastFpsTime = 0;
let isFirstStart = true; // true until first init completes
let pendingRebuildConfig = null;

// UI elements
let presetSelect;
let timeScaleInput;
let spaceScaleInput;
let interplanetaryScaleInput;
let stepIntervalInput;
let startBtn;
let pauseBtn;
let statusText;
let loadingOverlay;
let loadingText;
let loadingDetail;
let presetNameLabel;

function getStructuralConfig() {
    return {
        preset: presetSelect.value,
        spaceScale: parseFloat(spaceScaleInput.value) || DEFAULT_SPACE_SCALE,
        interplanetaryScale: parseFloat(interplanetaryScaleInput.value) || DEFAULT_INTERPLANETARY_SCALE,
    };
}

function getRuntimeConfig() {
    const parsed = parseInt(stepIntervalInput.value, 10);
    return {
        timeScale: parseFloat(timeScaleInput.value) || DEFAULT_TIME_SCALE,
        stepInterval: Number.isNaN(parsed) ? currentStepInterval : parsed,
    };
}

function getConfigFromUI() {
    return {
        ...getStructuralConfig(),
        ...getRuntimeConfig(),
    };
}

function setLoading(text, detail) {
    loadingText.textContent = text;
    loadingDetail.textContent = detail || "";
    loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
    loadingOverlay.classList.add("hidden");
}

function setPauseEnabled(enabled) {
    pauseBtn.disabled = !enabled;
}

function spawnWorker(config) {
    if (worker) {
        worker.terminate();
        worker = null;
    }
    isFirstStart = true;

    worker = new Worker("js/worker.js");
    setLoading("Initializing Pyodide...", "");
    statusText.textContent = "Initializing...";
    statusText.style.color = "#a5a5a5";
    setPauseEnabled(false);

    worker.onmessage = async (event) => {
        const msg = event.data;
        switch (msg.type) {
            case "ready":
                setLoading("Pyodide ready, building simulation...", `Preset: ${config.preset}`);
                statusText.textContent = "Building simulation...";
                worker.postMessage({ type: "init", ...config });
                break;
            case "initialized": {
                const initConfig = pendingRebuildConfig || config;
                pendingRebuildConfig = null;
                await onSimulationInitialized(msg.data, initConfig);
                break;
            }
            case "state":
                onWorkerState(msg.data);
                break;
            case "error":
                console.error("Worker error:", msg.message);
                setLoading("Error", msg.message);
                statusText.textContent = `Error: ${msg.message}`;
                statusText.style.color = "#ff6b6b";
                break;
            default:
                console.warn("Unknown message from worker:", msg);
        }
    };

    worker.onerror = (err) => {
        console.error("Worker uncaught error:", err);
        setLoading("Worker error", err.message);
        statusText.textContent = `Worker error: ${err.message}`;
        statusText.style.color = "#ff6b6b";
    };
}

function rebuildSimulation(config) {
    if (!worker) {
        spawnWorker(config);
        return;
    }
    // Keep old state visible; send rebuild to existing worker
    pendingRebuildConfig = config;
    statusText.innerHTML = 'Rebuilding... <span class="status-spinner"></span>';
    statusText.style.color = "#a5a5a5";
    simulationActive = false;
    stateA = null;
    stateB = null;
    setPauseEnabled(false);
    worker.postMessage({ type: "rebuild", ...config });
}

function updateWorkerConfig(runtimeConfig) {
    if (!worker) return;
    worker.postMessage({ type: "config", ...runtimeConfig });
    if (runtimeConfig.stepInterval !== undefined) {
        currentStepInterval = runtimeConfig.stepInterval;
    }
}

async function onSimulationInitialized(state, config) {
    if (isFirstStart) {
        hideLoading();
        isFirstStart = false;

        const canvas = document.getElementById("gl-canvas");
        renderer = new WebRenderer(canvas, state, config.spaceScale);
        await renderer.loadControls();
    } else {
        // Rebuild: update renderer with new geometry counts
        renderer.rebuildFromState(state);
    }

    statusText.textContent = `Running: ${config.preset}`;
    statusText.style.color = "#4dabf7";
    presetNameLabel.textContent = `Preset: ${config.preset}`;
    simulationActive = true;
    currentStepInterval = config.stepInterval || DEFAULT_STEP_INTERVAL;
    pauseBtn.textContent = "Pause";
    setPauseEnabled(true);

    // Auto-frame camera based on scene bounds
    autoFrameCamera(state);

    // Prime the state buffer with the first two identical states
    stateA = state;
    stateB = state;
    stateBReceivedAt = performance.now();

    startRenderLoop();
}

function autoFrameCamera(state) {
    let maxDist = 0;
    // Consider planets
    if (state.planets) {
        for (const p of state.planets) {
            const d = Math.sqrt(p.position[0]**2 + p.position[1]**2 + p.position[2]**2) + p.radius;
            if (d > maxDist) maxDist = d;
        }
    }
    // Always consider satellites too (GPS etc. have high orbits)
    const satPos = state.satellite_positions;
    if (state.num_sats > 0) {
        for (let i = 0; i < state.num_sats; i++) {
            const d = Math.sqrt(
                satPos[i*3]**2 + satPos[i*3+1]**2 + satPos[i*3+2]**2
            );
            if (d > maxDist) maxDist = d;
        }
    }

    if (maxDist > 0 && renderer && renderer.camera && renderer.controls) {
        // Place camera so bounding sphere fills most of the viewport
        const dist = Math.max(10, Math.min(500, maxDist * 1.9));
        renderer.camera.position.set(dist, dist * 0.6, dist);
        renderer.camera.lookAt(0, 0, 0);
        renderer.controls.target.set(0, 0, 0);
        renderer.controls.update();
    }
}

function onWorkerState(state) {
    if (!renderer) return;

    if (!stateB) {
        // First worker state after init/rebuild
        stateA = state;
        stateB = state;
        stateBReceivedAt = performance.now();
        return;
    }

    // Shift states: A <- B, B <- new
    stateA = stateB;
    stateB = state;
    stateBReceivedAt = performance.now();
}

function startRenderLoop() {
    if (isAnimating) return;
    isAnimating = true;
    lastFrameTime = performance.now();
    lastFpsTime = performance.now();
    frameCount = 0;
    animFrameId = requestAnimationFrame(renderLoop);
}

function stopRenderLoop() {
    isAnimating = false;
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
}

function pauseSimulation() {
    simulationActive = false;
    if (worker) worker.postMessage({ type: "pause" });
}

function resumeSimulation() {
    simulationActive = true;
    if (worker) worker.postMessage({ type: "resume" });
    // Reset time bookkeeping so dt doesn't jump after a long pause
    lastFrameTime = performance.now();
    lastFpsTime = performance.now();
    frameCount = 0;
}

function renderLoop(timestamp) {
    if (!isAnimating) return;

    if (lastFrameTime === 0) lastFrameTime = timestamp;
    const dt = (timestamp - lastFrameTime) / 1000;
    lastFrameTime = timestamp;

    frameCount++;
    if (timestamp - lastFpsTime >= 1000) {
        frameCount = 0;
        lastFpsTime = timestamp;
    }

    if (renderer && stateB) {
        const elapsed = performance.now() - stateBReceivedAt;
        let alpha = elapsed / currentStepInterval;
        alpha = Math.max(0, Math.min(alpha, 1.0));

        const canInterpolate = simulationActive && stateA && stateA.satellite_positions && stateB.satellite_positions;

        if (canInterpolate) {
            renderer.lerpSatellites(stateA.satellite_positions, stateB.satellite_positions, alpha);
            renderer.updateLinks(stateB, stateA.satellite_positions, stateB.satellite_positions, alpha);
            renderer.lerpPlanets(stateA, stateB, alpha);
        } else {
            renderer.updateSatellites(stateB.satellite_positions);
            renderer.updateLinks(stateB);
            renderer.updatePlanets(stateB);
        }

        renderer.updateUI(stateB, dt, frameCount);
        renderer.render(dt);
    }

    animFrameId = requestAnimationFrame(renderLoop);
}

function initUI() {
    presetSelect = document.getElementById("preset-select");
    timeScaleInput = document.getElementById("time-scale");
    spaceScaleInput = document.getElementById("space-scale");
    interplanetaryScaleInput = document.getElementById("interplanetary-scale");
    stepIntervalInput = document.getElementById("step-interval");
    startBtn = document.getElementById("start-btn");
    pauseBtn = document.getElementById("pause-btn");
    statusText = document.getElementById("status-text");
    loadingOverlay = document.getElementById("loading-overlay");
    loadingText = document.getElementById("loading-text");
    loadingDetail = document.getElementById("loading-detail");
    presetNameLabel = document.getElementById("preset-name");

    // Parse URL parameters as defaults
    const params = new URLSearchParams(window.location.search);
    const urlPreset = params.get("preset");
    const urlTimeScale = params.get("timeScale");
    const urlSpaceScale = params.get("spaceScale");
    const urlInterplanetaryScale = params.get("interplanetaryScale");
    const urlStepInterval = params.get("stepInterval");

    // Populate preset dropdown
    PRESETS.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        presetSelect.appendChild(opt);
    });

    if (urlPreset && PRESETS.includes(urlPreset)) {
        presetSelect.value = urlPreset;
    } else {
        presetSelect.value = DEFAULT_PRESET;
    }

    timeScaleInput.value = urlTimeScale || DEFAULT_TIME_SCALE;
    spaceScaleInput.value = urlSpaceScale || DEFAULT_SPACE_SCALE;
    interplanetaryScaleInput.value = urlInterplanetaryScale || DEFAULT_INTERPLANETARY_SCALE;
    stepIntervalInput.value = parseInt(urlStepInterval, 10) || DEFAULT_STEP_INTERVAL;
    currentStepInterval = parseInt(urlStepInterval, 10) || DEFAULT_STEP_INTERVAL;

    // Start / Restart button
    startBtn.addEventListener("click", () => {
        const config = getConfigFromUI();
        if (!worker) {
            spawnWorker(config);
        } else {
            rebuildSimulation(config);
        }
    });

    // Structural changes (require rebuild)
    presetSelect.addEventListener("change", () => {
        if (worker) rebuildSimulation(getConfigFromUI());
    });
    spaceScaleInput.addEventListener("change", () => {
        if (worker) rebuildSimulation(getConfigFromUI());
    });
    interplanetaryScaleInput.addEventListener("change", () => {
        if (worker) rebuildSimulation(getConfigFromUI());
    });

    // Runtime changes (no rebuild needed)
    timeScaleInput.addEventListener("change", () => {
        if (worker) updateWorkerConfig(getRuntimeConfig());
    });
    stepIntervalInput.addEventListener("change", () => {
        if (worker) updateWorkerConfig(getRuntimeConfig());
    });

    pauseBtn.addEventListener("click", () => {
        if (simulationActive) {
            pauseSimulation();
            pauseBtn.textContent = "Resume";
        } else {
            resumeSimulation();
            pauseBtn.textContent = "Pause";
        }
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
        if (e.code === "Space") {
            e.preventDefault();
            pauseBtn.click();
        }
    });

    // Auto-start if preset is provided in URL
    if (urlPreset) {
        spawnWorker(getConfigFromUI());
    }
}

initUI();

/**
 * DSNS Visualizer Public API
 *
 * Exposed on window.DSNSVisualizer for external embedders.
 *
 * Methods:
 *   loadPreset(name: string)          – Switch to a named preset.
 *   setTimeScale(val: number)         – Update simulation speed.
 *   setSpaceScale(val: number)        – Update spatial scale.
 *   setInterplanetaryScale(val: number) – Update interplanetary scale.
 *   setStepInterval(val: number)      – Update step interval in ms.
 *   play()                              – Start or resume simulation.
 *   pause()                             – Pause simulation.
 *   isRunning(): boolean               – Whether the sim is active.
 *
 * Example:
 *   DSNSVisualizer.loadPreset("earth-mars");
 *   DSNSVisualizer.setTimeScale(500);
 */
window.DSNSVisualizer = {
    loadPreset(name) {
        if (PRESETS.includes(name)) {
            presetSelect.value = name;
            if (worker) rebuildSimulation(getConfigFromUI());
            else spawnWorker(getConfigFromUI());
        } else {
            console.warn(`Unknown preset: ${name}`);
        }
    },
    setTimeScale(val) {
        timeScaleInput.value = val;
        if (worker) updateWorkerConfig(getRuntimeConfig());
    },
    setSpaceScale(val) {
        spaceScaleInput.value = val;
        if (worker) rebuildSimulation(getConfigFromUI());
    },
    setInterplanetaryScale(val) {
        interplanetaryScaleInput.value = val;
        if (worker) rebuildSimulation(getConfigFromUI());
    },
    setStepInterval(val) {
        stepIntervalInput.value = val;
        if (worker) updateWorkerConfig(getRuntimeConfig());
    },
    play() {
        if (!worker) {
            spawnWorker(getConfigFromUI());
        } else if (!simulationActive) {
            resumeSimulation();
            pauseBtn.textContent = "Pause";
        }
    },
    pause() {
        if (simulationActive) {
            pauseSimulation();
            pauseBtn.textContent = "Resume";
        }
    },
    isRunning() {
        return simulationActive;
    },
};
