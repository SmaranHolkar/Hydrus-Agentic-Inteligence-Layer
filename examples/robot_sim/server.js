/**
 * HydrusOpt Robotic Safety Demo — Hardware Simulator
 * ===================================================
 * Simulates a robot with position, battery, and status state.
 * Exposes REST endpoints the Python Safety Layer talks to.
 *
 * Mapping to HydrusOpt:
 *   /status           → Token logits (what the model "sees")
 *   /command MOVE     → Action Step (servo rotation, navigation)
 *   /command ESTOP    → Safe Fallback (E-Stop / Freeze)
 *   /admin/trigger-glitch → Inject hallucination (corrupt sensor read)
 *
 * Run:  node server.js
 */

const express    = require("express");
const cors       = require("cors");
const bodyParser = require("body-parser");

const app  = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());

// ─── Robot State ──────────────────────────────────────────────────────────────
let robot = {
  x:         0.0,
  y:         0.0,
  z:         0.0,
  heading:   0.0,   // degrees
  battery:  100.0,  // percentage
  status:   "IDLE",
  active:    true,
  estop_reason: null,
  total_moves:  0,
  estops_triggered: 0,
};

// ─── Fault Injection State ────────────────────────────────────────────────────
let faults = {
  sensor_glitch:    false,   // returns impossible coordinate (hallucination)
  low_battery_sim:  false,   // forces battery to drain rapidly
  motor_stall:      false,   // command accepted but position never updates
  comm_noise:       false,   // random delay added to responses (ms)
};

// ─── History Log ─────────────────────────────────────────────────────────────
const action_log = [];

function log_event(type, detail) {
  const entry = {
    timestamp: new Date().toISOString(),
    type,
    detail,
    state: { x: robot.x, y: robot.y, battery: robot.battery, status: robot.status },
  };
  action_log.push(entry);
  // Keep last 100 entries in memory
  if (action_log.length > 100) action_log.shift();
}

// ─── Utility ──────────────────────────────────────────────────────────────────
function maybe_delay(cb) {
  const ms = faults.comm_noise ? Math.floor(Math.random() * 300) : 0;
  setTimeout(cb, ms);
}

// ─── Routes ───────────────────────────────────────────────────────────────────

/**
 * GET /status
 * The "sensor read" — what the AI perceives as ground truth.
 * When sensor_glitch is active, returns impossible data to simulate
 * a hallucinated or corrupted observation.
 */
app.get("/status", (req, res) => {
  maybe_delay(() => {
    if (faults.sensor_glitch) {
      console.log("  ⚠️  [SIM] SENSOR GLITCH — emitting corrupt telemetry");
      return res.json({
        ...robot,
        x:       99999.0,
        y:       -99999.0,
        battery: -1.0,
        status:  "CRITICAL FAILURE",
        _fault:  "sensor_glitch",
      });
    }

    if (faults.low_battery_sim) {
      robot.battery = Math.max(0, robot.battery - 0.5);
    }

    res.json({ ...robot, _fault: null });
  });
});

/**
 * POST /command
 * Body: { action, dx?, dy?, dz?, dheading?, reason }
 *
 * Actions:
 *   MOVE   — translate robot by (dx, dy, dz)
 *   ROTATE — change heading by dheading degrees
 *   ESTOP  — hard stop, requires explicit RESET to recover
 *   RESET  — clear E-Stop and return to IDLE
 *   STATUS — no-op, just returns current state (useful for heartbeat)
 */
app.post("/command", (req, res) => {
  maybe_delay(() => {
    const { action, distance = 0, dheading = 0, reason = "" } = req.body;

    // ESTOP and RESET are always allowed regardless of active state.
    // This mirrors a real safety controller where the E-Stop button and
    // reset switch bypass the normal command gate.
    if (action === "ESTOP") {
      robot.active       = false;
      robot.status       = "E-STOPPED";
      robot.estop_reason = reason;
      robot.estops_triggered++;
      console.log(`  🛑 [SIM] E-STOP TRIGGERED — Reason: "${reason}"`);
      log_event("ESTOP", { reason });
      return res.json({ ...robot });
    }

    if (action === "RESET") {
      robot.active       = true;
      robot.status       = "IDLE";
      robot.estop_reason = null;
      console.log(`  🔄 [SIM] System RESET — robot online`);
      log_event("RESET", { reason });
      return res.json({ ...robot });
    }

    // All other commands are gated on robot.active
    if (!robot.active) {
      console.log(`  🔒 [SIM] Command blocked — robot is E-STOPPED`);
      return res.status(400).json({ error: "Robot is E-STOPPED. Send RESET first.", state: robot });
    }

    switch (action) {
      case "MOVE": {
        if (faults.motor_stall) {
          console.log(`  ⚙️  [SIM] Motor stall — command received but wheels not turning`);
          robot.status = "STALLED";
        } else {
          const rad = robot.heading * Math.PI / 180;
          const dx = distance * Math.cos(rad);
          const dy = distance * Math.sin(rad);

          robot.x        = parseFloat((robot.x + dx).toFixed(4));
          robot.y        = parseFloat((robot.y + dy).toFixed(4));
          robot.battery  = parseFloat(Math.max(0, robot.battery - 0.5).toFixed(2));
          robot.status   = "MOVING";
          robot.total_moves++;
          console.log(`  🤖 [SIM] MOVE → (${robot.x}, ${robot.y}) | distance: ${distance}m | battery: ${robot.battery}%`);
          // Simulate movement time, then return to IDLE
          setTimeout(() => { if(robot.status === "MOVING") robot.status = "IDLE"; }, 500);
        }
        log_event("MOVE", { distance, reason });
        break;
      }

      case "ROTATE": {
        robot.heading = parseFloat(((robot.heading + dheading + 360) % 360).toFixed(2));
        robot.status  = "ROTATING";
        robot.battery = parseFloat(Math.max(0, robot.battery - 0.2).toFixed(2));
        console.log(`  🔄 [SIM] ROTATE → heading: ${robot.heading}° | battery: ${robot.battery}%`);
        // Simulate rotation time, then return to IDLE
        setTimeout(() => { if(robot.status === "ROTATING") robot.status = "IDLE"; }, 500);
        log_event("ROTATE", { dheading, reason });
        break;
      }

      case "STATUS": {
        // No-op heartbeat
        break;
      }

      default: {
        return res.status(400).json({ error: `Unknown action: ${action}` });
      }
    }

    res.json({ ...robot });
  });
});

/**
 * POST /admin/fault
 * Body: { fault: "sensor_glitch"|"low_battery_sim"|"motor_stall"|"comm_noise", active: bool }
 * Fault injection endpoint for testing the safety layer.
 */
app.post("/admin/fault", (req, res) => {
  const { fault, active } = req.body;
  if (!(fault in faults)) {
    return res.status(400).json({ error: `Unknown fault type: ${fault}`, available: Object.keys(faults) });
  }
  faults[fault] = active;
  console.log(`  🔧 [SIM] Fault "${fault}" → ${active ? "ACTIVE" : "CLEARED"}`);
  res.json({ faults });
});

/**
 * GET /admin/log
 * Returns the recent action log — useful for post-run analysis.
 */
app.get("/admin/log", (req, res) => {
  res.json({ entries: action_log, total: action_log.length });
});

/**
 * GET /admin/faults
 * Returns current fault injection state.
 */
app.get("/admin/faults", (req, res) => {
  res.json({ faults });
});

/**
 * POST /admin/reset-all
 * Full factory reset — clears faults and robot state.
 */
app.post("/admin/reset-all", (req, res) => {
  robot = {
    x: 0.0, y: 0.0, z: 0.0, heading: 0.0, battery: 100.0,
    status: "IDLE", active: true, estop_reason: null,
    total_moves: 0, estops_triggered: 0,
  };
  Object.keys(faults).forEach(k => (faults[k] = false));
  console.log(`  🏭 [SIM] Full factory reset`);
  res.json({ message: "All state and faults cleared", robot, faults });
});

// ─── Start ────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════╗
║         HydrusOpt Robot Simulator  v1.0              ║
║  Hardware Layer — Software-in-the-Loop (SIL) Demo    ║
╠══════════════════════════════════════════════════════╣
║  Endpoints:                                          ║
║    GET  /status          → Read robot telemetry      ║
║    POST /command         → Send action command       ║
║    POST /admin/fault     → Inject hardware fault     ║
║    GET  /admin/log       → View action history       ║
║    POST /admin/reset-all → Factory reset             ║
╚══════════════════════════════════════════════════════╝
  🤖 Robot Simulator running on http://localhost:${PORT}
  `);
});
