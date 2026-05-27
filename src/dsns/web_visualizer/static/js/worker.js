// Web Worker for running the Pyodide simulation off the main thread.
// Loaded as a classic worker (not ES module) so we use importScripts().

importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js");

let pyodide = null;
let runner = null;
let simTime = 0.0;
let timeScale = 100.0;
let stepInterval = 100; // ms
let intervalId = null;
let running = false;

async function initPyodide() {
    pyodide = await loadPyodide({
        stdout: (text) => console.log("[Pyodide stdout]", text),
        stderr: (text) => console.error("[Pyodide stderr]", text),
    });

    await pyodide.loadPackage(["numpy", "micropip"]);

    // Install pure-Python deps that dsns needs but Pyodide doesn't bundle
    try {
        await pyodide.runPythonAsync(`
import micropip
await micropip.install("PyAstronomy", deps=False)
await micropip.install("six")
await micropip.install("quantities")
await micropip.install("sgp4")
        `);
        console.log("[Worker] Python dependencies installed successfully");
    } catch (err) {
        console.error("[Worker] Failed to install dependencies:", err);
        self.postMessage({ type: "error", message: `Failed to install deps: ${err.message}` });
        return;
    }

    // Fetch dsns.zip and mount it into the Pyodide FS
    // Resolve relative to this worker script (go up from js/ to site root)
    const dsnsZipUrl = new URL("../dsns.zip", self.location.href).href;
    try {
        const response = await fetch(dsnsZipUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch dsns.zip: ${response.status} ${response.statusText}`);
        }
        const zipData = await response.arrayBuffer();
        pyodide.FS.writeFile("/tmp/dsns.zip", new Uint8Array(zipData));
        pyodide.FS.mkdirTree("/tmp/dsns");
        pyodide.runPython(`
import zipfile, sys
zipfile.ZipFile("/tmp/dsns.zip", "r").extractall("/tmp/dsns")
if "/tmp/dsns" not in sys.path:
    sys.path.insert(0, "/tmp/dsns")
        `);
    } catch (err) {
        console.error("Failed to load dsns.zip:", err);
        self.postMessage({ type: "error", message: `Failed to load dsns.zip: ${err.message}` });
        return;
    }

    self.postMessage({ type: "ready" });
}

function _createRunner(data) {
    const preset = data.preset || "walker";
    timeScale = data.timeScale || 100.0;
    const spaceScale = data.spaceScale || 1e-6;
    const interplanetaryScale = data.interplanetaryScale || 1e-10;
    stepInterval = data.stepInterval || 333;

    pyodide.runPython(`
import sys, types
if "scipy" not in sys.modules:
    scipy = types.ModuleType("scipy")
    scipy.__version__ = "1.0.0"
    scipy.special = types.ModuleType("scipy.special")
    sys.modules["scipy"] = scipy
    sys.modules["scipy.special"] = scipy.special
from dsns.web_visualizer.runner import WebRunner, PRESETS
runner = WebRunner(
    preset="${preset}",
    time_scale=${timeScale},
    space_scale=${spaceScale},
    interplanetary_scale=${interplanetaryScale},
)
    `);
    runner = pyodide.globals.get("runner");
    simTime = 0.0;

    // Get initial state
    const state = runner.step(0.0);
    const jsState = state.toJs({ depth: 4, dict_converter: Object.fromEntries });
    state.destroy();
    self.postMessage({ type: "initialized", data: jsState });
}

function _startLoop() {
    running = true;
    intervalId = setInterval(() => {
        if (!running) return;
        simTime += (stepInterval / 1000.0) * timeScale;
        try {
            const nextState = runner.step(simTime);
            const jsNext = nextState.toJs({ depth: 4, dict_converter: Object.fromEntries });
            nextState.destroy();
            self.postMessage({ type: "state", data: jsNext });
        } catch (err) {
            console.error("Simulation step error:", err);
            self.postMessage({ type: "error", message: err.message });
        }
    }, stepInterval);
}

function handleInit(data) {
    if (!pyodide) {
        self.postMessage({ type: "error", message: "Pyodide not initialized" });
        return;
    }

    try {
        _createRunner(data);
        _startLoop();
    } catch (err) {
        console.error("Failed to initialize runner:", err);
        self.postMessage({ type: "error", message: err.message });
    }
}

function handleRebuild(data) {
    if (!pyodide || !runner) {
        self.postMessage({ type: "error", message: "Pyodide or runner not initialized" });
        return;
    }

    // Stop existing loop
    running = false;
    if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
    }

    try {
        _createRunner(data);
        _startLoop();
    } catch (err) {
        console.error("Failed to rebuild runner:", err);
        self.postMessage({ type: "error", message: err.message });
    }
}

self.onmessage = function (event) {
    const msg = event.data;
    switch (msg.type) {
        case "init":
            handleInit(msg);
            break;
        case "rebuild":
            handleRebuild(msg);
            break;
        case "config":
            if (msg.timeScale !== undefined) timeScale = msg.timeScale;
            if (msg.stepInterval !== undefined) {
                const newInterval = msg.stepInterval;
                if (newInterval !== stepInterval && intervalId) {
                    clearInterval(intervalId);
                    stepInterval = newInterval;
                    intervalId = setInterval(() => {
                        if (!running) return;
                        simTime += (stepInterval / 1000.0) * timeScale;
                        try {
                            const nextState = runner.step(simTime);
                            const jsNext = nextState.toJs({ depth: 4, dict_converter: Object.fromEntries });
                            nextState.destroy();
                            self.postMessage({ type: "state", data: jsNext });
                        } catch (err) {
                            console.error("Simulation step error:", err);
                            self.postMessage({ type: "error", message: err.message });
                        }
                    }, stepInterval);
                } else {
                    stepInterval = newInterval;
                }
            }
            break;
        case "pause":
            running = false;
            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
            }
            break;
        case "resume":
            if (!running && runner) {
                running = true;
                intervalId = setInterval(() => {
                    if (!running) return;
                    simTime += (stepInterval / 1000.0) * timeScale;
                    try {
                        const nextState = runner.step(simTime);
                        const jsNext = nextState.toJs({ depth: 4, dict_converter: Object.fromEntries });
                        nextState.destroy();
                        self.postMessage({ type: "state", data: jsNext });
                    } catch (err) {
                        console.error("Simulation step error:", err);
                        self.postMessage({ type: "error", message: err.message });
                    }
                }, stepInterval);
            }
            break;
        default:
            console.warn("Unknown worker message type:", msg.type);
    }
};

// Bootstrap
initPyodide().catch((err) => {
    console.error("Pyodide init failed:", err);
    self.postMessage({ type: "error", message: `Pyodide init failed: ${err.message}` });
});
