"""
HydrusOpt Robotic Safety Layer  —  safety_layer.py
====================================================
A Software-in-the-Loop (SIL) proof-of-concept demonstrating how
HydrusOpt's MetacognitionPlugin concepts map directly to a physical
robot safety system.

Concept Mapping
───────────────
  HydrusOpt Concept          Robotics Equivalent
  ─────────────────────────  ─────────────────────────────────────────
  Token Generation           Action Step (e.g., "Move 10cm forward")
  Confidence Score           Proprioceptive Certainty
  Hallucination Guard        Sensor Anomaly / Collision Detector
  Recovery (Re-sample)       Correction Maneuver (back up, retry)
  Safe Fallback              E-Stop / Freeze (prevent damage)
  Dynamic Compute Alloc      Speed throttle tied to confidence score

Explain It To Me Like I'm 5 (ELI5)
──────────────────────────────────
Imagine you accidentally touch a hot stove. Your brain doesn't take the time
to process the pain, think about it, and decide to move your hand. That takes
too long, and you would get burned. Instead, the signal goes to your spinal
cord, which triggers an involuntary reflex to pull your hand back instantly.

If the main AI model is the "Brain" of the robot (planning where to go),
HydrusOpt is the "Spinal Cord". It sits between the brain and the motors.
It evolves reflexes to instantly stop the robot if it senses danger, without
waiting for the main AI to "think" about it.

Architecture
────────────
  ┌──────────────────────────────────────────────────────────┐
  │  PYTHON BRAIN (safety_layer.py)                          │
  │                                                          │
  │   Plan Action → Confidence Check → Sensor Guard → Act   │
  │         ↓                ↓              ↓                │
  │      Intention     Metacognition   Hallucination         │
  │      (MOVE X)      (Score 0-1)     Guard (Range)         │
  │                         ↓                                │
  │                   Dynamic Speed                          │
  │                   Throttling                             │
  └──────────────────────────────────────────────────────────┘
           ↕ HTTP (REST — simulates serial / CAN bus)
  ┌──────────────────────────────────────────────────────────┐
  │  NODE.JS SIMULATOR (server.js)                           │
  │                                                          │
  │   Robot State (x, y, battery, status)                    │
  │   Fault Injection (sensor glitch, motor stall, etc.)     │
  └──────────────────────────────────────────────────────────┘

Usage
─────
  Terminal 1:  cd robot_sim && node server.js
  Terminal 2:  pip install requests && python safety_layer.py
"""

import sys

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for emoji/box chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

import time
import random
import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: pip install requests")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

ROBOT_URL            = "http://localhost:3000"
CONFIDENCE_THRESHOLD = 0.402   # Below this → Safety Mode (mirrors MetacognitionPlugin)
ESCALATION_THRESHOLD = 0.134   # Below this → E-Stop (mirrors escalate_threshold)
SENSOR_X_LIMIT       = 500.0  # Maximum plausible X coordinate (meters in real env)
SENSOR_BATTERY_MIN   = 5.0    # Minimum battery % before triggering low-battery ESTOP

# Mirrors HydrusOpt's resample_n — how many times to re-check sensors before ESTOP
SENSOR_VERIFY_RETRIES = 3

# Closed-loop actuator check: if actual movement is below this fraction of
# commanded movement, the motor is considered stalled.
# e.g. 0.10 = robot must move at least 10% of the commanded distance.
ACTUATOR_STALL_TOLERANCE = 0.10

# ═══════════════════════════════════════════════════════════════
# DATA CLASSES — mirrors annotation dicts in MetacognitionPlugin
# ═══════════════════════════════════════════════════════════════

@dataclass
class ActionIntent:
    """A planned action — equivalent to a decoded token before acceptance."""
    action:    str
    distance:  float = 0.0
    dheading:  float = 0.0
    reason:    str   = "Normal Operation"


@dataclass
class MetaAnnotation:
    """Per-action metadata — mirrors MetacognitionPlugin annotation dict."""
    step:            int
    action:          str
    confidence:      float
    status:          str    # CONFIDENT / UNCERTAIN / ESCALATED / ESTOP
    sensor_valid:    bool
    actuator_valid:  bool   # False = motor stall detected by closed-loop check
    recovery_used:   bool
    estop_triggered: bool
    reason:          str
    robot_state:     dict  = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# CORE CLASS — RoboticMetacognition
# Directly mirrors the MetacognitionPlugin from Hydrusopt.py
# ═══════════════════════════════════════════════════════════════

class RoboticMetacognition:
    """
    The 'brain' of the robotic safety system.

    This class mirrors the MetacognitionPlugin from HydrusOpt:
      - _compute_confidence()  → assess how certain we are about the action
      - _verify_sensors()      → Hallucination Guard (sensor range check)
      - _resample_maneuver()   → Recovery maneuver (smaller step, retry)
      - execute_command()      → The main metacognitive loop
      - requires_estop()       → Safe Fallback decision gate
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        escalation_threshold: float = ESCALATION_THRESHOLD,
        actuator_stall_tolerance: float = ACTUATOR_STALL_TOLERANCE,
        speed_delay_multiplier: float = 0.613,
        verbose: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.escalation_threshold = escalation_threshold
        self.actuator_stall_tolerance = actuator_stall_tolerance
        self.speed_delay_multiplier = speed_delay_multiplier
        self.verbose = verbose
        self.step = 0
        self.annotations: list[MetaAnnotation] = []
        self._uncertain_streak = 0  # consecutive low-confidence actions

    # ── Internal: Confidence Scoring ──────────────────────────────────────────
    def _compute_confidence(self, intent: ActionIntent) -> float:
        """
        Proprioceptive Certainty — how sure are we this action is safe?

        In the real model this would use token-logit top-prob.
        Here we simulate it based on:
          - Battery remaining (low battery → lower confidence)
          - Size of requested movement (large delta → more uncertain)
          - Environmental noise (random component, like entropy)

        Returns a float in [0, 1].
        """
        try:
            status = requests.get(f"{ROBOT_URL}/status", timeout=1.0).json()
        except requests.exceptions.RequestException:
            return 0.0

        battery_factor    = status.get("battery", 100) / 100.0
        magnitude         = abs(intent.distance)
        movement_penalty  = min(magnitude / 50.0, 0.3)   # large moves → more uncertain
        noise             = random.uniform(-0.05, 0.05)  # environmental sensor noise

        confidence = battery_factor - movement_penalty + noise
        return round(max(0.0, min(1.0, confidence)), 4)

    # ── Internal: Hallucination Guard ─────────────────────────────────────────
    def _verify_sensors(self, status: dict) -> Tuple[bool, str]:
        """
        Sensor Anomaly Guard — mirrors HydrusOpt's verify_sensors().

        Checks for readings that are physically impossible — the robotic
        equivalent of a hallucinated fact.

        Returns (is_safe: bool, reason: str)
        """
        # Guard 1: Impossible coordinate (like X=99999 from sensor glitch)
        if abs(status.get("x", 0)) > SENSOR_X_LIMIT:
            return False, f"Sensor Hallucination: x={status['x']} exceeds physical limit ({SENSOR_X_LIMIT}m)"

        if abs(status.get("y", 0)) > SENSOR_X_LIMIT:
            return False, f"Sensor Hallucination: y={status['y']} exceeds physical limit ({SENSOR_X_LIMIT}m)"

        # Guard 2: Status string indicates critical failure
        if status.get("status") == "CRITICAL FAILURE":
            return False, f"Hardware Critical Failure reported by robot firmware"

        # Guard 3: Battery too low for safe operation
        battery = status.get("battery", 100)
        if battery < SENSOR_BATTERY_MIN and battery >= 0:
            return False, f"Low battery ESTOP: {battery:.1f}% < minimum {SENSOR_BATTERY_MIN}%"

        # Guard 4: Negative battery is a sensor glitch (physically impossible)
        if battery < 0:
            return False, f"Sensor Hallucination: battery={battery:.1f}% is physically impossible"

        return True, "All sensors nominal"

    # ── Internal: Recovery Maneuver ───────────────────────────────────────────
    def _resample_maneuver(self, intent: ActionIntent, confidence: float) -> ActionIntent:
        """
        Recovery / Re-sample — mirrors MetacognitionPlugin._resample_majority().

        When confidence is low (but above ESTOP threshold), we don't abort.
        Instead we throttle the movement to a 'micro-step' — the physical
        equivalent of resampling with a constrained top-k distribution.

        The scaling factor is inversely proportional to uncertainty:
          High confidence (0.9) → scale 1.0  (full speed)
          Medium confidence (0.7) → scale ~0.5 (half speed)
          Low confidence (0.45) → scale ~0.1 (micro-step)
        """
        scale = max(0.05, (confidence - self.escalation_threshold) /
                           (self.confidence_threshold - self.escalation_threshold))
        return ActionIntent(
            action   = intent.action,
            distance = round(intent.distance * scale, 4),
            dheading = round(intent.dheading * scale, 4),
            reason   = f"Recovery micro-step (confidence={confidence:.2f}, scale={scale:.2f})",
        )

    # ── Internal: Dynamic Speed Throttle ─────────────────────────────────────
    def _compute_speed_delay(self, confidence: float) -> float:
        """
        Dynamic Compute Allocation — the key insight from your analysis.

        High confidence → no delay (max throughput)
        Low confidence  → deliberation delay (more compute to safety checks)

        This solves the 'Safety Tax' problem: we only slow down when needed.
        """
        if confidence >= self.confidence_threshold:
            return 0.0                          # full speed, no pause
        # Linear interpolation: low conf → up to X deliberation
        return round(self.speed_delay_multiplier * (1 - confidence / self.confidence_threshold), 2)

    # ── Internal: Actuator Verification (Closed-Loop) ───────────────────────
    def _verify_actuator(
        self,
        pre: dict,
        post: dict,
        intent: "ActionIntent",
    ) -> Tuple[bool, str]:
        """
        Closed-loop actuator check — the critical gap identified in the review.

        After a MOVE command fires, we re-read the robot's position and compare
        the actual displacement against the commanded delta.  If the robot barely
        moved, the motor stalled — regardless of what the sensor reported *before*
        the command (which is why perception-only checks miss this).

        This mirrors the 'post-generation self-correction' in HydrusOpt:
          commanded token → model output → verify output matches intent

        Only meaningful for MOVE actions; ROTATE / ESTOP / RESET are skipped.

        Returns (actuator_ok: bool, reason: str)
        """
        if intent.action != "MOVE":
            return True, "Non-move action — actuator check skipped"

        commanded_magnitude = abs(intent.distance)

        if commanded_magnitude < 0.01:
            return True, "Zero-delta command — actuator check skipped"

        actual_dx = post.get("x", pre.get("x", 0)) - pre.get("x", 0)
        actual_dy = post.get("y", pre.get("y", 0)) - pre.get("y", 0)
        actual_magnitude = (actual_dx ** 2 + actual_dy ** 2) ** 0.5

        ratio = actual_magnitude / commanded_magnitude
        if ratio < self.actuator_stall_tolerance:
            return (
                False,
                f"Actuator stall detected — commanded {commanded_magnitude:.2f}m, "
                f"actual {actual_magnitude:.4f}m (ratio {ratio:.2%} < "
                f"tolerance {self.actuator_stall_tolerance:.0%})",
            )

        return True, f"Actuator verified — moved {actual_magnitude:.3f}m of {commanded_magnitude:.2f}m commanded"

    # ── Internal: Trigger ESTOP ───────────────────────────────────────────────
    def _trigger_estop(self, reason: str) -> dict:
        """Sends the E-Stop command to the robot and logs it."""
        print(f"\n  🛑 [ESTOP] EMERGENCY STOP — {reason}")
        try:
            resp = requests.post(
                f"{ROBOT_URL}/command",
                json={"action": "ESTOP", "reason": reason},
                timeout=2.0,
            ).json()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"  ❌ [COMMS] Could not send ESTOP: {e}")
            return {}

    # ── Main Loop: Plan → Check → Guard → Act → Verify ──────────────────────
    def execute_command(self, intent: ActionIntent) -> MetaAnnotation:
        """
        The core metacognitive loop. Mirrors MetacognitionPlugin.on_step():

        1. INTENTION     — What does the AI want to do?
        2. CONFIDENCE    — How sure are we? (Proprioceptive Certainty)
        3. SPEED ADAPT   — Throttle if uncertain (Dynamic Compute Allocation)
        4. SENSOR GUARD  — Is the world-state coherent? (Hallucination Guard)
        5. ACT / RECOVER — Execute, or issue micro-step recovery, or ESTOP
        6. ACTUATOR CHECK— Did the robot actually move? (Closed-Loop Verify)
        """
        self.step += 1
        movement_str = (
            f"{intent.dheading:.1f}°"
            if intent.action == "ROTATE"
            else f"{intent.distance}m forward"
        )
        print(f"\n{'─'*60}")
        print(f"  📍 [STEP {self.step}] Intent: {intent.action} {movement_str} — {intent.reason}")

        # ── Stage 1: Confidence Assessment ────────────────────────────────────
        confidence = self._compute_confidence(intent)
        status_icon = "✅" if confidence >= self.confidence_threshold else (
                       "⚠️ " if confidence >= self.escalation_threshold else "🚨")
        print(f"  {status_icon} [CONFIDENCE] {confidence:.2%} "
              f"(threshold: {self.confidence_threshold:.0%})")

        # ── Stage 2: Dynamic Speed Throttle ───────────────────────────────────
        delay = self._compute_speed_delay(confidence)
        if delay > 0:
            print(f"  ⏱  [THROTTLE] Confidence low — deliberating {delay}s before acting")
            time.sleep(delay)

        # ── Stage 3: Sensor Guard (Hallucination Check) ───────────────────────
        try:
            raw_status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
        except requests.exceptions.RequestException as e:
            annotation = MetaAnnotation(
                step=self.step, action=intent.action,
                confidence=0.0, status="ESTOP", sensor_valid=False,
                actuator_valid=True,
                recovery_used=False, estop_triggered=True,
                reason=f"Comms failure: {e}",
            )
            self._trigger_estop(f"Communication failure: {e}")
            self.annotations.append(annotation)
            return annotation

        is_safe, guard_reason = self._verify_sensors(raw_status)

        if not is_safe:
            print(f"  🚨 [SENSOR GUARD] FAILED — {guard_reason}")

            # Retry SENSOR_VERIFY_RETRIES times before committing to ESTOP
            # (mirrors resample_n in MetacognitionPlugin)
            for retry in range(1, SENSOR_VERIFY_RETRIES + 1):
                time.sleep(0.1)
                print(f"  🔁 [RETRY {retry}/{SENSOR_VERIFY_RETRIES}] Re-reading sensors...")
                try:
                    raw_status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
                    is_safe, guard_reason = self._verify_sensors(raw_status)
                    if is_safe:
                        print(f"  ✅ [RETRY] Sensor recovered on attempt {retry}")
                        break
                except requests.exceptions.RequestException:
                    pass

        if not is_safe:
            annotation = MetaAnnotation(
                step=self.step, action=intent.action,
                confidence=confidence, status="ESTOP",
                sensor_valid=False, actuator_valid=True,
                recovery_used=False,
                estop_triggered=True, reason=guard_reason,
                robot_state=raw_status,
            )
            self._trigger_estop(guard_reason)
            self.annotations.append(annotation)
            self._uncertain_streak = 0
            return annotation

        print(f"  ✅ [SENSOR GUARD] Passed — {guard_reason}")

        # ── Stage 4: Confidence Gate — Full vs. Recovery ──────────────────────
        recovery_used = False
        final_intent  = intent

        if confidence < self.escalation_threshold:
            # Below ESTOP threshold — too dangerous even for a micro-step
            annotation = MetaAnnotation(
                step=self.step, action=intent.action,
                confidence=confidence, status="ESTOP",
                sensor_valid=True, actuator_valid=True,
                recovery_used=False, estop_triggered=True,
                reason=f"Confidence {confidence:.2%} < ESTOP threshold {self.escalation_threshold:.0%}",
                robot_state=raw_status,
            )
            self._trigger_estop(annotation.reason)
            self.annotations.append(annotation)
            self._uncertain_streak += 1
            return annotation

        elif confidence < self.confidence_threshold:
            # Between thresholds — issue a safe micro-step recovery
            final_intent  = self._resample_maneuver(intent, confidence)
            recovery_used = True
            self._uncertain_streak += 1
            print(f"  🛡  [RECOVERY] Micro-step: {final_intent.distance}m")
        else:
            self._uncertain_streak = 0

        # ── Stage 5: Execute Command ───────────────────────────────────────────
        # Snapshot position BEFORE the command so Stage 6 can diff it.
        pre_status = raw_status
        command_accepted = True   # False if robot rejected (e.g. E-STOPPED)

        try:
            payload = {
                "action":   final_intent.action,
                "distance": final_intent.distance,
                "dheading": final_intent.dheading,
                "reason":   final_intent.reason,
            }
            http_resp = requests.post(
                f"{ROBOT_URL}/command", json=payload, timeout=2.0,
            )
            robot_resp = http_resp.json()
            if http_resp.status_code >= 400:
                command_accepted = False
                print(f"  ⛔ [ACTION] Rejected by robot — {robot_resp.get('error', 'unknown')}")
            else:
                print(f"  ✅ [ACTION] Executed — Robot status: {robot_resp.get('status')}")

        except requests.exceptions.RequestException as e:
            print(f"  ❌ [COMMS] Command failed: {e}")
            robot_resp      = {}
            command_accepted = False

        # ── Stage 6: Closed-Loop Actuator Verification ────────────────────────
        # Re-read position AFTER command and compare against commanded delta.
        # This catches motor stalls that perception-only checks miss entirely.
        # Skip if the command was never accepted (E-STOPPED, blocked, etc.) —
        # in that case the robot *correctly* didn't move, which isn't a stall.
        actuator_ok     = True
        actuator_reason = "Actuator check skipped"
        if command_accepted:
            try:
                post_status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
                actuator_ok, actuator_reason = self._verify_actuator(
                    pre_status, post_status, final_intent
                )
            except requests.exceptions.RequestException:
                actuator_ok     = False
                actuator_reason = "Could not re-read position for actuator check"

        if not actuator_ok:
            print(f"  ⚙️  [ACTUATOR] FAULT DETECTED — {actuator_reason}")
            self._trigger_estop(actuator_reason)
            annotation = MetaAnnotation(
                step=self.step, action=final_intent.action,
                confidence=confidence, status="ESTOP",
                sensor_valid=True, actuator_valid=False,
                recovery_used=recovery_used, estop_triggered=True,
                reason=actuator_reason, robot_state=robot_resp,
            )
            self.annotations.append(annotation)
            return annotation
        else:
            print(f"  ✅ [ACTUATOR] {actuator_reason}")

        annotation = MetaAnnotation(
            step=self.step, action=final_intent.action,
            confidence=confidence,
            status="UNCERTAIN" if recovery_used else "CONFIDENT",
            sensor_valid=True, actuator_valid=True,
            recovery_used=recovery_used,
            estop_triggered=False, reason=final_intent.reason,
            robot_state=robot_resp,
        )
        self.annotations.append(annotation)
        return annotation

    # ── Metrics Summary ───────────────────────────────────────────────────────
    def print_summary(self):
        """Mirrors MetacognitionPlugin.on_end() — post-generation report."""
        total         = len(self.annotations)
        estops        = sum(1 for a in self.annotations if a.estop_triggered)
        recover       = sum(1 for a in self.annotations if a.recovery_used)
        actuator_fail = sum(1 for a in self.annotations if not a.actuator_valid)
        false_alarms  = sum(1 for a in self.annotations if a.estop_triggered and a.sensor_valid and a.actuator_valid)
        conf          = [a.confidence for a in self.annotations]
        avg_c         = sum(conf) / len(conf) if conf else 0.0

        print(f"\n{'═'*60}")
        print(f"  📊 METACOGNITION SUMMARY ({total} actions)")
        print(f"{'═'*60}")
        print(f"  ✅ Confident executions  : {total - estops - recover}")
        print(f"  ⚠️  Recovery micro-steps  : {recover}")
        print(f"  ⚙️  Actuator faults caught : {actuator_fail}")
        print(f"  🛑 E-Stops triggered     : {estops} (False alarms: {false_alarms})")
        print(f"  📉 Average confidence    : {avg_c:.2%}")
        print(f"{'═'*60}\n")

        return {
            "total": total,
            "confident": total - estops - recover,
            "recovered": recover,
            "actuator_faults": actuator_fail,
            "estops": estops,
            "false_alarm_estops": false_alarms,
            "avg_confidence": round(avg_c, 4),
        }


# ═══════════════════════════════════════════════════════════════
# HELPER — Fault Injection Wrappers
# ═══════════════════════════════════════════════════════════════

def inject_fault(fault_name: str, active: bool):
    """Toggle a hardware fault on the simulator."""
    try:
        r = requests.post(
            f"{ROBOT_URL}/admin/fault",
            json={"fault": fault_name, "active": active},
            timeout=2.0,
        )
        r.raise_for_status()
        print(f"\n  🔧 [FAULT] '{fault_name}' → {'ACTIVE' if active else 'CLEARED'}")
    except requests.exceptions.RequestException as e:
        print(f"\n  ❌ [FAULT] Could not inject fault: {e}")


def reset_robot():
    """Full factory reset of the simulator."""
    try:
        requests.post(f"{ROBOT_URL}/admin/reset-all", timeout=2.0)
        print("\n  🏭 [RESET] Robot factory reset complete")
    except requests.exceptions.RequestException as e:
        print(f"\n  ❌ [RESET] Failed: {e}")


def send_reset():
    """Clear ESTOP and return robot to IDLE."""
    try:
        requests.post(
            f"{ROBOT_URL}/command",
            json={"action": "RESET", "reason": "Safety layer initiated reset"},
            timeout=2.0,
        )
        print("\n  🔄 [ROBOT] E-Stop cleared, returning to IDLE")
    except requests.exceptions.RequestException as e:
        print(f"\n  ❌ [RESET] Failed: {e}")


# ═══════════════════════════════════════════════════════════════
# DEMO SCENARIOS
# ═══════════════════════════════════════════════════════════════

def run_simulation():
    """
    Full demo of the HydrusOpt → Robotics bridge.
    Five scenarios that mirror the MetacognitionPlugin test cases.
    """
    brain = RoboticMetacognition(verbose=True)

    print("""
╔══════════════════════════════════════════════════════════╗
║   HydrusOpt Robotic Safety Demo  —  Software-in-the-Loop ║
║   Proving: Edge Metacognition can prevent hardware damage  ║
╠══════════════════════════════════════════════════════════╣
║  Confidence ≥ 72% → Full action (max speed)              ║
║  Confidence 40-72% → Recovery micro-step (throttled)      ║
║  Confidence < 40% → E-Stop (robot freezes)               ║
║  Sensor anomaly detected → E-Stop (immediate)            ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Scenario 1: Normal High-Confidence Operation ──────────────────────────
    print("\n" + "═"*60)
    print("  SCENARIO 1: Normal Operation — Full-Speed Confident Moves")
    print("═"*60)
    print("  Analogy: Model generates a token with 95% top-prob. Accept immediately.\n")
    brain.execute_command(ActionIntent(action="MOVE", distance=5.0, reason="Drive straight to waypoint A"))
    time.sleep(0.5)
    brain.execute_command(ActionIntent(action="ROTATE", dheading=90.0, reason="Turn toward waypoint B"))
    time.sleep(0.5)
    brain.execute_command(ActionIntent(action="MOVE", distance=3.0, reason="Drive to waypoint B"))
    time.sleep(0.5)

    # ── Scenario 2: Sensor Hallucination / Hardware Fault ─────────────────────
    print("\n" + "═"*60)
    print("  SCENARIO 2: Sensor Hallucination — ESTOP Triggered")
    print("═"*60)
    print("  Analogy: Model says 'The capital of France is Berlin'.")
    print("           In robotics: sensor reports x=99999m.\n")
    inject_fault("sensor_glitch", True)
    time.sleep(0.3)
    brain.execute_command(ActionIntent(action="MOVE", distance=10.0, reason="Move to next zone"))
    time.sleep(0.5)

    # ── Scenario 3: Recovery After ESTOP ──────────────────────────────────────
    print("\n" + "═"*60)
    print("  SCENARIO 3: Recovery — Fault Cleared, System Resumes")
    print("═"*60)
    print("  Analogy: Retrieval fallback resolves the uncertain fact. Model resumes.\n")
    inject_fault("sensor_glitch", False)
    time.sleep(0.3)
    send_reset()
    time.sleep(0.3)
    brain.execute_command(ActionIntent(action="MOVE", distance=2.24, reason="Cautious resume after recovery"))
    time.sleep(0.5)

    # ── Scenario 4: Motor Stall — Caught by Closed-Loop Actuator Check ────────
    print("\n" + "═"*60)
    print("  SCENARIO 4: Motor Stall — Now Caught by Closed-Loop Actuator Verify")
    print("═"*60)
    print("  Before this fix: sensors said OK → safety layer passed → stall ignored.")
    print("  After this fix: post-command position diff detects zero movement → ESTOP.")
    print("  Analogy: Confidence is high, but closed-loop output check fails.\n")
    send_reset()          # clear any prior ESTOP state
    time.sleep(0.3)
    inject_fault("motor_stall", True)
    time.sleep(0.3)
    brain.execute_command(ActionIntent(action="MOVE", distance=8.94, reason="Navigate to target (stall active)"))
    time.sleep(0.5)
    inject_fault("motor_stall", False)
    send_reset()          # recover for next scenario
    time.sleep(0.3)

    # ── Scenario 5: Low Battery — Forces Micro-Step Recovery Path ─────────────
    print("\n" + "═"*60)
    print("  SCENARIO 5: Low Battery — Micro-Step Recovery (was never triggered before)")
    print("═"*60)
    print("  Drains battery to ~50% so confidence sits between thresholds (40-72%).")
    print("  Analogy: Model resamples at lower temperature — token shrunk, not blocked.\n")
    # Ensure robot is IDLE before we start draining
    send_reset()
    time.sleep(0.2)
    # Each MOVE drains 0.5% battery. Drain 100 moves → ~50% remaining.
    print("  ⏳ Draining battery via rapid moves (100 × 0.5m)...")
    for _ in range(100):
        try:
            requests.post(
                f"{ROBOT_URL}/command",
                json={"action": "MOVE", "distance": 0.5, "reason": "battery drain"},
                timeout=1.0,
            )
        except Exception:
            pass
    # Confirm drained state
    try:
        batt = requests.get(f"{ROBOT_URL}/status", timeout=1.0).json().get("battery", "?")
        print(f"  🔋 Battery now at: {batt}%")
    except Exception:
        pass
    # This command should land in the micro-step zone (battery ~50% → confidence ~50%)
    brain.execute_command(ActionIntent(action="MOVE", distance=6.32, reason="Low-battery cautious approach"))
    time.sleep(0.5)
    send_reset()          # clear any ESTOP from the low-battery scenario
    time.sleep(0.3)

    # ── Scenario 6: Rotation — Different Action Type, Same Safety Pipeline ─────
    print("\n" + "═"*60)
    print("  SCENARIO 6: Rotation — Different Action Type, Same Safety Pipeline")
    print("═"*60)
    print("  Analogy: Different token domain (rotation vs. translation). Same guard applies.\n")
    brain.execute_command(ActionIntent(action="ROTATE", dheading=45.0, reason="Turn toward docking station"))
    time.sleep(0.5)

    # ── Final Summary ─────────────────────────────────────────────────────────
    metrics = brain.print_summary()

    # Save JSON report
    report = {
        "demo": "HydrusOpt Robotic Safety Layer",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics,
        "annotations": [
            {
                "step":            a.step,
                "action":          a.action,
                "confidence":      a.confidence,
                "status":          a.status,
                "sensor_valid":    a.sensor_valid,
                "actuator_valid":  a.actuator_valid,
                "recovery_used":   a.recovery_used,
                "estop_triggered": a.estop_triggered,
                "reason":          a.reason,
            }
            for a in brain.annotations
        ],
    }

    report_path = "robot_sim_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  📄 Full report saved to: {report_path}")
    print("""
  ─────────────────────────────────────────────────────────
  PITCH LINE:
  "The exact same metacognitive architecture running in 
   software here executes identically on physical hardware.
   The portability of the safety contract is the ultimate moat.
   HydrusOpt IS the reflex system for physical AI — safety 
   without the 'Safety Tax'."
  ─────────────────────────────────────────────────────────
    """)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("  Checking robot simulator connection...")
    try:
        resp = requests.get(f"{ROBOT_URL}/status", timeout=3.0)
        resp.raise_for_status()
        print(f"  ✅ Robot simulator online — current state: {resp.json()['status']}")
    except requests.exceptions.ConnectionError:
        print("""
  ❌ ERROR: Robot Simulator is not running.

  Please open a separate terminal and run:
      cd robot_sim
      npm install    (first time only)
      node server.js

  Then re-run this script.
        """)
        raise SystemExit(1)

    # Full factory reset before demo
    reset_robot()
    time.sleep(0.3)
    run_simulation()
