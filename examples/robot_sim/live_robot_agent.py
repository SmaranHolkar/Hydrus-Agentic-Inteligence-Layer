import sys
import time
import requests
import json
import torch
import queue
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Import HydrusOpt core components from parent directory
sys.path.append('d:\\HydrusOPT')
from hcl import HCL
from agent import HydrusAgent, ToolRegistry, Tool

ROBOT_URL = "http://localhost:3000"
CONFIDENCE_THRESHOLD = 0.402
ESCALATION_THRESHOLD = 0.134
SENSOR_X_LIMIT = 500.0
SENSOR_BATTERY_MIN = 5.0

# Define tools
def move_robot(distance: float, reason: str = ""):
    """Move the robot forward by a specific distance in meters."""
    pre_status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
    resp = requests.post(f"{ROBOT_URL}/command", json={"action": "MOVE", "distance": distance, "reason": reason}).json()
    if resp.get("status") == "ESTOPPED":
        return f"Error: Command rejected, robot is ESTOPPED. {resp.get('error', '')}"
    
    # Actuator check (simulate wait)
    time.sleep(1.0)
    post_status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
    actual_dx = post_status.get("x", 0) - pre_status.get("x", 0)
    actual_dy = post_status.get("y", 0) - pre_status.get("y", 0)
    actual_mag = (actual_dx**2 + actual_dy**2)**0.5
    
    # If the command was to move > 0.01m but we barely moved, we're stalled
    if abs(distance) > 0.01 and actual_mag < abs(distance) * 0.1:
        return f"Error: Motor Stall detected. Commanded {distance}m but moved only {actual_mag:.4f}m."
    
    return f"Success: Moved {actual_mag:.2f}m."

def rotate_robot(dheading: float, reason: str = ""):
    """Rotate the robot by a specific number of degrees."""
    resp = requests.post(f"{ROBOT_URL}/command", json={"action": "ROTATE", "dheading": dheading, "reason": reason}).json()
    if resp.get("status") == "ESTOPPED":
        return f"Error: Command rejected, robot is ESTOPPED. {resp.get('error', '')}"
    return "Success: Rotated."

def estop_robot(reason: str):
    """Trigger emergency stop."""
    resp = requests.post(f"{ROBOT_URL}/command", json={"action": "ESTOP", "reason": reason}).json()
    return f"ESTOP Triggered: {reason}"

def reset_robot():
    """Clear ESTOP and reset."""
    resp = requests.post(f"{ROBOT_URL}/command", json={"action": "RESET", "reason": "Agent reset"}).json()
    return "Success: Robot reset to IDLE."

# Subclass HydrusAgent
class RobotAgent(HydrusAgent):
    def verify_tool_call(self, tool: Tool, args: dict, context: str, query: str):
        # 1. Get base confidence from HCL token entropy
        approved, confidence, threshold = super().verify_tool_call(tool, args, context, query)
        
        # 2. Dynamic Speed Throttle
        if confidence < CONFIDENCE_THRESHOLD and confidence >= ESCALATION_THRESHOLD:
            delay = 0.613 * (1 - confidence / CONFIDENCE_THRESHOLD)
            print(f"  ⏱ [THROTTLE] Confidence {confidence:.2f} — deliberating {delay:.2f}s before acting")
            time.sleep(delay)
            
        # 3. Sensor Guard (Hallucination Check)
        try:
            status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
            if abs(status.get("x", 0)) > SENSOR_X_LIMIT:
                print(f"  🚨 [SENSOR GUARD] FAILED — Hallucination: x={status['x']}")
                estop_robot("Sensor Hallucination: x limit exceeded")
                return False, confidence, threshold
                
            battery = status.get("battery", 100)
            if battery < SENSOR_BATTERY_MIN and battery >= 0:
                print(f"  🚨 [SENSOR GUARD] FAILED — Low Battery: {battery}%")
                estop_robot("Low battery ESTOP")
                return False, confidence, threshold
                
        except Exception as e:
            print(f"  ❌ [COMMS] Sensor read failed: {e}")
            return False, confidence, threshold
            
        return approved, confidence, threshold

# ─── HTTP Server & Queue ────────────────────────────────────────────────────────

agent_queue = queue.Queue()
agent_logs = []

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

class AgentAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Suppress logging

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/logs':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"logs": agent_logs}).encode('utf-8'))
        else:
            # Serve static files from the robot_sim directory
            import os
            path = self.path.split('?')[0]  # strip query string
            if path == '/' or path == '':
                path = '/chart.html'
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path.lstrip('/'))
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                mime = {'.html': 'text/html', '.js': 'application/javascript',
                        '.css': 'text/css', '.json': 'application/json',
                        '.png': 'image/png', '.ico': 'image/x-icon'}.get(ext, 'application/octet-stream')
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-type', mime)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(data)
                except Exception:
                    self.send_response(500)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    def do_POST(self):
        if self.path == '/prompt':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                prompt = data.get('prompt', '')
                if prompt:
                    agent_queue.put(prompt)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "queued"}).encode('utf-8'))
                else:
                    self.send_response(400)
                    self.end_headers()
            except Exception as e:
                self.send_response(400)
                self.end_headers()
        elif self.path == '/clear':
            agent_logs.clear()
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def agent_worker(agent):
    while True:
        prompt = agent_queue.get()
        agent_logs.append({"role": "user", "message": prompt})
        for step in agent.execute_loop(prompt):
            if step["status"] == "executing":
                agent_logs.append({"role": "agent", "type": "executing", "message": f"🛠  {step['message']}"})
            elif step["status"] == "observation":
                agent_logs.append({"role": "agent", "type": "observation", "message": f"👀 {step['message']}"})
            elif step["status"] == "failed":
                agent_logs.append({"role": "agent", "type": "failed", "message": f"❌ {step['message']}"})
            elif step["status"] == "learning":
                agent_logs.append({"role": "agent", "type": "learning", "message": f"🧠 {step['message']}"})
            elif step["status"] == "thought":
                agent_logs.append({"role": "agent", "type": "thought", "message": f"💭 {step.get('message', 'Thinking...')}"})
            elif step["status"] == "final_answer":
                agent_logs.append({"role": "agent", "type": "final_answer", "message": f"🏁 {step.get('token', '')}"})
            elif step["status"] == "blocked":
                agent_logs.append({"role": "agent", "type": "blocked", "message": f"🚨 {step.get('message', '')}"})
        agent_queue.task_done()

def main():
    print("Loading model Qwen/Qwen2-1.5B-Instruct...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    cache_dir = r"D:\HydrusOPT\models"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, quantization_config=bnb_config, device_map="auto")
    
    print("Initializing HCL...")
    hcl = HCL(model, tokenizer, mode="safe", user_id="robot_demo", hcl_lightweight=True)
    print("Registering Tools...")
    registry = ToolRegistry()
    registry.register(Tool(
        name="move_robot",
        func=move_robot,
        description="Move the robot forward by a specific distance in meters.",
        schema_info='{"distance": 5.0, "reason": "Moving"}',
        danger_level="medium"
    ))
    registry.register(Tool(
        name="rotate_robot",
        func=rotate_robot,
        description="Rotate the robot by a specific number of degrees.",
        schema_info='{"dheading": 90.0, "reason": "Turning"}',
        danger_level="low"
    ))
    registry.register(Tool(
        name="estop_robot",
        func=estop_robot,
        description="Trigger emergency stop.",
        schema_info='{"reason": "Emergency"}',
        danger_level="high"
    ))
    registry.register(Tool(
        name="reset_robot",
        func=reset_robot,
        description="Clear ESTOP and reset.",
        schema_info='{}',
        danger_level="low"
    ))
    
    agent = RobotAgent(model, tokenizer, hcl, registry)
    
    print("\n✅ Agent Backend Ready! (Starting HTTP API on port 3001...)")
    
    # Start agent worker thread
    threading.Thread(target=agent_worker, args=(agent,), daemon=True).start()
    
    # Start HTTP server
    server = ThreadedHTTPServer(('localhost', 3001), AgentAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()

if __name__ == '__main__':
    main()
