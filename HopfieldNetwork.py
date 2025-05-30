import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        self.patterns = []

    def train(self, patterns):
        """Train the Hopfield network with given patterns"""
        self.patterns = [np.array(p) for p in patterns]
        self.weights = np.zeros((self.size, self.size))
        
        # Correct Hebbian learning rule - DON'T divide by number of patterns
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        
        # Zero diagonal (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        print(f"Training complete with {len(patterns)} patterns.")
        print(f"Weight matrix range: [{self.weights.min():.2f}, {self.weights.max():.2f}]")

    def recall(self, pattern, max_steps=50, convergence_threshold=0):
        """Recall a pattern using asynchronous updates with better initialization"""
        state = np.array(pattern, dtype=np.float64)
        
        # Add small random noise to avoid getting stuck
        if np.random.random() < 0.3:  # 30% chance to add noise
            noise_indices = np.random.choice(len(state), size=max(1, int(0.02 * len(state))), replace=False)
            state[noise_indices] *= -1
        
        for step in range(max_steps):
            old_state = state.copy()
            
            # Asynchronous updates in random order
            indices = np.random.permutation(len(state))
            changes = 0
            
            for i in indices:
                new_val = np.sign(np.dot(self.weights[i], state))
                if new_val != state[i]:
                    state[i] = new_val
                    changes += 1
            
            # Check for convergence
            if changes <= convergence_threshold:
                print(f"Converged after {step + 1} steps with {changes} changes")
                break
        else:
            print(f"Max steps ({max_steps}) reached, {changes} changes in last step")
            
        return state.astype(int)

    def energy(self, state):
        """Calculate the energy of a state"""
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    
    def pattern_similarity(self):
        """Check similarity between stored patterns"""
        print("\nPattern similarities (dot products):")
        for i in range(len(self.patterns)):
            for j in range(i + 1, len(self.patterns)):
                similarity = np.dot(self.patterns[i], self.patterns[j])
                print(f"Pattern {i} vs Pattern {j}: {similarity}")
                
    def verify_patterns(self):
        """Verify that stored patterns are stable"""
        print("\nPattern verification:")
        for i, pattern in enumerate(self.patterns):
            recalled = self.recall(pattern.copy(), max_steps=10)
            errors = np.sum(pattern != recalled)
            stability = (self.size - errors) / self.size * 100
            print(f"Pattern {i}: {errors} errors, {stability:.1f}% stable")
            
        return all(np.array_equal(p, self.recall(p.copy(), max_steps=10)) for p in self.patterns)