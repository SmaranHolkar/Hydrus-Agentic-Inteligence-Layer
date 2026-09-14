import random
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════
# 1. THE GENOME (The Reflex Parameters)
# ═══════════════════════════════════════════════════════════════
class ReflexGenome:
    def __init__(self, conf_thresh, esc_thresh, delay_mult, step_scale):
        # e.g., 0.72 (Full speed above this)
        self.conf_thresh = conf_thresh 
        # e.g., 0.40 (Hard E-Stop below this)
        self.esc_thresh = esc_thresh   
        # Multiplier for the deliberation pause when uncertain
        self.delay_mult = delay_mult   
        # How much to shrink the movement vector during recovery
        self.step_scale = step_scale   
        
        self.fitness_score = 0.0

    @classmethod
    def random(cls):
        """Spawn a completely random set of reflexes."""
        conf = random.uniform(0.50, 0.95)
        esc = random.uniform(0.10, conf - 0.05) # Must be lower than conf
        delay = random.uniform(0.5, 3.0)
        scale = random.uniform(0.05, 0.5)
        return cls(conf, esc, delay, scale)

# ═══════════════════════════════════════════════════════════════
# 2. THE SIMULATOR (The Fitness Function)
# ═══════════════════════════════════════════════════════════════
def evaluate_fitness(genome: ReflexGenome) -> float:
    """
    Simulates the robot trying to cross a warehouse.
    This replaces your manual testing with an automated score.
    """
    distance_traveled = 0.0
    time_spent = 0.0
    crashed = False

    # Simulate 50 steps in a chaotic environment (noisy sensors)
    for step in range(50):
        # The environment generates a random confidence score for this move
        env_confidence = random.uniform(0.2, 1.0)
        
        if env_confidence >= genome.conf_thresh:
            # Full speed ahead
            distance_traveled += 1.0
            time_spent += 0.1 
            
            # --- CRASH LOGIC EXPLANATION ---
            # If the robot's reflex decides to move fast (env_confidence >= genome.conf_thresh)
            # BUT the environment was actually highly hazardous (env_confidence < 0.4), it crashes.
            # This correctly penalizes genomes with recklessly low confidence_thresholds.
            # A safe robot would have hesitated (triggering the elif branch below) and survived.
            if env_confidence < 0.4:
                crashed = True
                break

        elif env_confidence >= genome.esc_thresh:
            # Micro-step recovery
            distance_traveled += (1.0 * genome.step_scale)
            # Add deliberation delay based on genome's multiplier
            time_spent += (0.1 + (1.0 - env_confidence) * genome.delay_mult)
            
        else:
            # Hardware E-Stop (Survives, but loses time)
            time_spent += 2.0 

    # Calculate final fitness score
    if crashed:
        return -1000.0  # Death penalty for crashing
    
    # Reward distance, penalize time (forces the AI to be both safe AND fast)
    score = (distance_traveled * 100) - (time_spent * 10)
    return score

# ═══════════════════════════════════════════════════════════════
# 3. EVOLUTION (Selection, Crossover, and Mutation)
# ═══════════════════════════════════════════════════════════════
def breed(parent1: ReflexGenome, parent2: ReflexGenome) -> ReflexGenome:
    """Combine traits from two successful reflexes."""
    child_conf = random.choice([parent1.conf_thresh, parent2.conf_thresh])
    child_esc  = random.choice([parent1.esc_thresh,  parent2.esc_thresh])
    child_dly  = random.choice([parent1.delay_mult,  parent2.delay_mult])
    child_step = random.choice([parent1.step_scale,  parent2.step_scale])
    
    # Ensure escalation is always below confidence
    if child_esc >= child_conf:
        child_esc = child_conf - 0.05
        
    return ReflexGenome(child_conf, child_esc, child_dly, child_step)

def mutate(genome: ReflexGenome):
    """Randomly tweak a parameter (Evolutionary mutation)."""
    if random.random() < 0.2: # 20% mutation chance
        genome.conf_thresh += random.uniform(-0.05, 0.05)
    if random.random() < 0.2:
        genome.esc_thresh += random.uniform(-0.05, 0.05)

# ═══════════════════════════════════════════════════════════════
# 4. THE MAIN LOOP
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    GENERATIONS = 50
    POPULATION_SIZE = 100

    print("🧬 Starting HydrusOpt Reflex Evolution...\n")

    # Initialize random population
    population = [ReflexGenome.random() for _ in range(POPULATION_SIZE)]

    for gen in range(GENERATIONS):
        # 1. Evaluate all genomes
        for genome in population:
            genome.fitness_score = evaluate_fitness(genome)
            
        # 2. Sort by highest score (survival of the fittest)
        population.sort(key=lambda g: g.fitness_score, reverse=True)
        
        # 3. Keep the top 20% (The Elites)
        elites = population[:int(POPULATION_SIZE * 0.2)]
        
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            best = elites[0]
            print(f"Gen {gen:02d} | Best Score: {best.fitness_score:7.2f} | "
                  f"Conf: {best.conf_thresh:.2f} | Esc: {best.esc_thresh:.2f} | "
                  f"Delay Mult: {best.delay_mult:.2f}")

        # 4. Breed the next generation from the elites
        next_generation = list(elites) # Elites live on
        while len(next_generation) < POPULATION_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = breed(p1, p2)
            mutate(child)
            next_generation.append(child)
            
        population = next_generation

    print("\n✅ Evolution Complete. Optimal Reflex Parameters Found.")
    
    best = population[0]
    print("\n🏆 Fittest Genome (Deploy to safety_layer.py):")
    print(f"CONFIDENCE_THRESHOLD     = {best.conf_thresh:.3f}")
    print(f"ESCALATION_THRESHOLD     = {best.esc_thresh:.3f}")
    print(f"SPEED_DELAY_MULTIPLIER   = {best.delay_mult:.3f}")
    print(f"RECOVERY_STEP_SCALE      = {best.step_scale:.3f}")
