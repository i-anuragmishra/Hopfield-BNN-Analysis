import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.W = np.zeros((size, size))
        self.patterns = []
    
    def store_patterns(self, patterns):
        self.patterns = patterns
        self.W = np.zeros((self.size, self.size))
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)  # no self-connections
        self.W /= self.size
    
    def compute_energy(self, state):
        return -0.5 * state @ self.W @ state
    
    def compute_overlap(self, state1, state2):
        return np.sum(state1 * state2) / self.size
    
    def update(self, state, method='synchronous'):
        if method == 'synchronous':
            return np.sign(self.W @ state)
        else:  # asynchronous
            new_state = state.copy()
            i = np.random.randint(0, self.size)
            new_state[i] = np.sign(self.W[i] @ state)
            return new_state
    
    def recall(self, noisy_state, max_steps=100, method='synchronous', track_energy=False):
        state = noisy_state.copy()
        energies = [self.compute_energy(state)] if track_energy else None
        
        for _ in range(max_steps):
            new_state = self.update(state, method)
            if track_energy:
                energies.append(self.compute_energy(new_state))
            if np.array_equal(new_state, state):
                break
            state = new_state
            
        if track_energy:
            return state, np.array(energies)
        return state
    
    def compute_basin_size(self, pattern, n_samples=100, flip_range=(1, 10)):
        results = []
        for n_flips in range(flip_range[0], flip_range[1] + 1):
            successes = 0
            for _ in range(n_samples):
                noisy = pattern.copy()
                flip_indices = np.random.choice(len(noisy), n_flips, replace=False)
                noisy[flip_indices] *= -1
                recalled = self.recall(noisy)
                if np.array_equal(recalled, pattern):
                    successes += 1
            results.append(successes / n_samples)
        return np.array(results)
    
    def theoretical_capacity(self):
        return int(0.138 * self.size)