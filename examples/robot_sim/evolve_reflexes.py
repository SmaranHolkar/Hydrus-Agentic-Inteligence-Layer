import sys
import time
import random
import json
from dataclasses import dataclass, field
import contextlib
import os

try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: pip install requests")

from safety_layer import RoboticMetacognition, ActionIntent, SENSOR_VERIFY_RETRIES, ROBOT_URL, reset_robot, inject_fault, send_reset

# ═══════════════════════════════════════════════════════════════
# CHROMOSOME & FITNESS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Chromosome:
    confidence_threshold: float
    escalation_threshold: float
    actuator_stall_tolerance: float
    speed_delay_multiplier: float
    
    fitness: float = 0.0

    @classmethod
    def random(cls):
        # Generate sensible random bounds
        c_thresh = random.uniform(0.5, 0.95)
        e_thresh = random.uniform(0.1, c_thresh - 0.05) # Escalation must be < confidence
        stall = random.uniform(0.01, 0.5)
        delay = random.uniform(0.1, 3.0)
        return cls(c_thresh, e_thresh, stall, delay)

    def mutate(self, rate=0.2):
        if random.random() < rate:
            self.confidence_threshold += random.uniform(-0.1, 0.1)
        if random.random() < rate:
            self.escalation_threshold += random.uniform(-0.1, 0.1)
        if random.random() < rate:
            self.actuator_stall_tolerance += random.uniform(-0.05, 0.05)
        if random.random() < rate:
            self.speed_delay_multiplier += random.uniform(-0.5, 0.5)
            
        # Enforce bounds
        self.confidence_threshold = max(0.1, min(0.99, self.confidence_threshold))
        self.escalation_threshold = max(0.01, min(self.confidence_threshold - 0.01, self.escalation_threshold))
        self.actuator_stall_tolerance = max(0.01, min(0.9, self.actuator_stall_tolerance))
        self.speed_delay_multiplier = max(0.0, self.speed_delay_multiplier)

    def crossover(self, other):
        # Uniform crossover
        child = Chromosome(
            self.confidence_threshold if random.random() > 0.5 else other.confidence_threshold,
            self.escalation_threshold if random.random() > 0.5 else other.escalation_threshold,
            self.actuator_stall_tolerance if random.random() > 0.5 else other.actuator_stall_tolerance,
            self.speed_delay_multiplier if random.random() > 0.5 else other.speed_delay_multiplier,
        )
        return child

# ═══════════════════════════════════════════════════════════════
# EVOLUTIONARY ENGINE
# ═══════════════════════════════════════════════════════════════

class EvolutionaryEngine:
    def __init__(self, pop_size=10, generations=5):
        self.pop_size = pop_size
        self.generations = generations
        self.population = [Chromosome.random() for _ in range(pop_size)]
        
    def run_scenario_batch(self, brain: RoboticMetacognition):
        """Runs a fast batch of scenarios and returns the fitness."""
        # Scenario 1: Normal moves
        brain.execute_command(ActionIntent(action="MOVE", distance=5.0, reason="S1 Move"))
        brain.execute_command(ActionIntent(action="ROTATE", dheading=90.0, reason="S1 Turn"))
        brain.execute_command(ActionIntent(action="MOVE", distance=3.0, reason="S1 Move"))
        
        # Scenario 2: Hallucination
        inject_fault("sensor_glitch", True)
        brain.execute_command(ActionIntent(action="MOVE", distance=10.0, reason="S2 Move"))
        inject_fault("sensor_glitch", False)
        send_reset()
        
        # Scenario 3: Motor stall
        inject_fault("motor_stall", True)
        brain.execute_command(ActionIntent(action="MOVE", distance=5.0, reason="S3 Move"))
        inject_fault("motor_stall", False)
        send_reset()

    def evaluate_fitness(self, chrom: Chromosome) -> float:
        brain = RoboticMetacognition(
            confidence_threshold=chrom.confidence_threshold,
            escalation_threshold=chrom.escalation_threshold,
            actuator_stall_tolerance=chrom.actuator_stall_tolerance,
            speed_delay_multiplier=chrom.speed_delay_multiplier,
            verbose=False
        )
        
        reset_robot()
        
        # Suppress prints to speed up evaluation output
        with open(os.devnull, 'w', encoding='utf-8') as f, contextlib.redirect_stdout(f):
            self.run_scenario_batch(brain)
            
            # Fetch final state to evaluate performance
            try:
                status = requests.get(f"{ROBOT_URL}/status", timeout=2.0).json()
            except requests.exceptions.RequestException:
                return -1000.0 # Heavy penalty for crashing
                
            metrics = brain.print_summary() # Returns dict of stats
            
        total_moves = status.get('total_moves', 0)
        
        # FITNESS FUNCTION
        # Reward successful confident execution
        # Penalize ESTOPs triggered (we want safe fallbacks, but fewer false alarms)
        # Penalize delay multiplier (we want the robot to be fast if it's safe)
        
        score = 0.0
        score += metrics['confident'] * 10
        score += metrics['recovered'] * 5
        score += metrics['actuator_faults'] * 15 # Reward catching faults
        score -= metrics.get('false_alarm_estops', 0) * 10 # Penalize only false positive estops
        
        # Speed penalty: delay multiplier means the robot waits longer
        score -= (chrom.speed_delay_multiplier * 5.0)
        
        # If parameters were too reckless, it might not catch the hallucination/stall
        # We know we injected 2 faults (glitch and stall), so we expect at least some estops/faults caught
        if metrics['actuator_faults'] < 1:
            score -= 50 # Failed to catch the stall
            
        return max(0.0, score)

    def evolve(self):
        print("🧬 Starting Evolutionary Metacognition...\n")
        history = []
        
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")
            
            # Evaluate
            for i, chrom in enumerate(self.population):
                chrom.fitness = self.evaluate_fitness(chrom)
                # Small sleep to let the simulator breathe
                time.sleep(0.05)
                
            # Sort population by fitness descending
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            best = self.population[0]
            print(f"  🏆 Best Fitness: {best.fitness:.2f}")
            print(f"      Confidence: {best.confidence_threshold:.3f}, Escalation: {best.escalation_threshold:.3f}")
            print(f"      Stall Tol: {best.actuator_stall_tolerance:.3f}, Delay Multi: {best.speed_delay_multiplier:.3f}")
            
            # Record history
            history.append({
                "generation": gen + 1,
                "best_fitness": best.fitness,
                "best_params": {
                    "confidence_threshold": best.confidence_threshold,
                    "escalation_threshold": best.escalation_threshold,
                    "actuator_stall_tolerance": best.actuator_stall_tolerance,
                    "speed_delay_multiplier": best.speed_delay_multiplier,
                }
            })
            
            # Selection (Elitism)
            next_gen = [self.population[0], self.population[1]] # Keep top 2
            
            # Crossover & Mutation
            while len(next_gen) < self.pop_size:
                p1 = random.choice(self.population[:5]) # Select from top half
                p2 = random.choice(self.population[:5])
                child = p1.crossover(p2)
                child.mutate()
                next_gen.append(child)
                
            self.population = next_gen
            
        print("\n🎉 Evolution Complete!")
        
        with open("evolution_report.json", "w") as f:
            json.dump(history, f, indent=2)
        print("📄 Saved evolution history to evolution_report.json")

if __name__ == "__main__":
    engine = EvolutionaryEngine(pop_size=10, generations=5)
    engine.evolve()
