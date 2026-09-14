# HydrusOpt Robotic Safety Demo

**Software-in-the-Loop (SIL) proof-of-concept** demonstrating how HydrusOpt's
MetacognitionPlugin maps directly to a physical robot safety system.

---

## Concept Mapping

| HydrusOpt Concept | Robotics Equivalent |
|:---|:---|
| Token Generation | Action Step (e.g., "Move 10cm forward") |
| Confidence Score | Proprioceptive Certainty |
| Hallucination Guard | Sensor Anomaly / Collision Detector |
| Recovery (Re-sample) | Correction Maneuver (micro-step, retry) |
| Safe Fallback | E-Stop / Freeze (prevent damage) |
| Dynamic Compute Allocation | Speed throttle tied to confidence score |

---

## Explain It To Me Like I'm 5 (ELI5)

Imagine you accidentally touch a hot stove. Your brain doesn't take the time
to process the pain, think about it, and decide to move your hand. That takes
too long, and you would get burned. Instead, the signal goes straight to your
spinal cord, which triggers an involuntary **reflex** to pull your hand back
instantly.

In this architecture, the main AI model (which is slow and expensive) is the
"Brain" of the robot, planning where to go. HydrusOpt acts as the **"Spinal Cord"**.
It sits between the brain and the motors. It evolves numerical reflexes to
instantly stop the robot or take cautious micro-steps if it senses danger,
without having to wait for the slow main AI to "think" about safety.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  PYTHON BRAIN (safety_layer.py)                          │
│                                                          │
│   Plan Action → Confidence Check → Sensor Guard → Act   │
│         ↓                ↓              ↓                │
│      Intention     Metacognition   Hallucination         │
│      (MOVE X)      (Score 0-1)     Guard (Range)         │
│                         ↓                                │
│                   Dynamic Speed Throttle                 │
│                   (Safety without the Tax)               │
└──────────────────────────────────────────────────────────┘
         ↕ HTTP REST  (simulates CAN bus / serial link)
┌──────────────────────────────────────────────────────────┐
│  NODE.JS SIMULATOR (server.js)                           │
│                                                          │
│   Robot State (x, y, z, battery, heading, status)       │
│   Fault Injection (sensor glitch, motor stall, etc.)    │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Terminal 1 — Start the hardware simulator:**
```powershell
cd robot_sim
npm install    # first time only
node server.js
```

**Terminal 2 — Run the metacognitive safety brain:**
```powershell
pip install requests
python safety_layer.py
```

---

## What the Demo Shows

### Scenario 1: Normal Operation
Confidence ≥ 72% → **Full-speed execution** with no delay. No safety overhead.
Equivalent to a token with 95% top-prob being accepted immediately.

### Scenario 2: Sensor Hallucination → E-Stop
A hardware fault is injected. The sensor returns `x=99999m` — physically
impossible. The safety layer:
1. Detects the anomaly (Hallucination Guard)
2. Retries `SENSOR_VERIFY_RETRIES` times (mirrors `resample_n`)
3. Triggers **E-Stop** — robot freezes instantly

Equivalent: model says "The capital of France is Berlin" but we catch and block it
*before* the robot acts on it.

### Scenario 3: Recovery
Fault cleared → ESTOP lifted → robot resumes at full confidence.
Equivalent to retrieval fallback resolving the uncertain fact.

### Scenario 4: Motor Stall
Sensors report nominal but actuators fail — demonstrates that the sensor guard
and actuator layer are separate, independent safety layers.

### Scenario 5: Rotation
Shows the same metacognitive pipeline applies equally to any action type
(translation *and* rotation), proving generality.

---

## Dynamic Compute Allocation (Solving the "Safety Tax")

```
Confidence ≥ 72%  →  0ms delay    (full throughput, no overhead)
Confidence 40-72% →  scaled delay (deliberate before acting)
Confidence < 40%  →  E-Stop       (system freeze)
```

The robot only "thinks harder" when the environment demands it — not as
a blanket penalty on every single movement.

---

## Simulator API

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/status` | GET | Read current robot telemetry |
| `/command` | POST | Send action (`MOVE`, `ROTATE`, `ESTOP`, `RESET`) |
| `/admin/fault` | POST | Inject hardware fault |
| `/admin/faults` | GET | View active faults |
| `/admin/log` | GET | View action history |
| `/admin/reset-all` | POST | Factory reset |

### Available Fault Types

| Fault | Simulates |
|:---|:---|
| `sensor_glitch` | Corrupt sensor read (hallucinated coordinate) |
| `low_battery_sim` | Accelerated battery drain |
| `motor_stall` | Actuator accepts command but doesn't move |
| `comm_noise` | Random latency on all responses |

---

## Output Files

- `robot_sim_report.json` — JSON annotation log of the full demo run,
  with per-action confidence scores and ESTOP decisions

---

## The Pitch Line

> *"This £0 simulation demonstrates a safety layer that, in production,
> prevents a £50,000 robot from destroying itself or injuring a warehouse
> operative. HydrusOpt is the reflex system for physical AI — safety
> without the 'Safety Tax'."*
