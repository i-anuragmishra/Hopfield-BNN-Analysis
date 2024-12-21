# Main script to run Hopfeld network experments and visulize results
import numpy as np
import matplotlib.pyplot as plt
from hopfield_network import HopfieldNetwork
import os
import seaborn as sns

def pattern_to_image(vec, shape=(8,8)):
    return vec.reshape(shape)

def show_pattern(pattern, title="", ax=None):
    if ax is None:
        plt.imshow(pattern.reshape(8,8), cmap='binary')
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(pattern.reshape(8,8), cmap='binary')
        ax.set_title(title)
        ax.axis('off')

def create_random_pattern(size=64):
    return np.where(np.random.rand(size) > 0.5, 1, -1)

def analyze_convergence(hn, pattern, noise_levels=[0.1, 0.3, 0.5]):
    plt.figure(figsize=(15, 5))
    for idx, noise in enumerate(noise_levels):
        noisy = pattern.copy()
        n_flips = int(noise * len(pattern))
        flip_indices = np.random.choice(len(noisy), n_flips, replace=False)
        noisy[flip_indices] *= -1
        
        # Track energy during recall
        recalled, energies = hn.recall(noisy, track_energy=True)
        
        plt.subplot(1, 3, idx+1)
        plt.plot(energies, '-o')
        plt.title(f'Energy Convergence (Noise={noise:.1f})')
        plt.xlabel('Update Step')
        plt.ylabel('Energy')
    plt.tight_layout()
    plt.savefig("results/hopfield_results/energy_convergence.png")
    plt.close()

def analyze_basin_of_attraction(hn, patterns):
    plt.figure(figsize=(10, 6))
    for i, pattern in enumerate(patterns):
        basin_sizes = hn.compute_basin_size(pattern, n_samples=50)
        plt.plot(range(1, len(basin_sizes) + 1), basin_sizes, '-o', label=f'Pattern {i+1}')
    
    plt.xlabel('Number of Flipped Bits')
    plt.ylabel('Recovery Success Rate')
    plt.title('Basin of Attraction Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/hopfield_results/basin_of_attraction.png")
    plt.close()

def analyze_pattern_overlap(patterns):
    n_patterns = len(patterns)
    overlap_matrix = np.zeros((n_patterns, n_patterns))
    
    for i in range(n_patterns):
        for j in range(n_patterns):
            overlap_matrix[i,j] = np.sum(patterns[i] * patterns[j]) / len(patterns[i])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Pattern Overlap Matrix')
    plt.xlabel('Pattern Index')
    plt.ylabel('Pattern Index')
    plt.savefig("results/hopfield_results/pattern_overlap.png")
    plt.close()

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("results/hopfield_results", exist_ok=True)
    
    # Create patterns
    p1 = np.ones((64,))
    p1[32:] = -1  # half top white, half bottom black
    
    p2 = np.ones((64,))
    p2[::8] = -1  # a vertical line of -1 on the left
    
    p3 = np.array([1 if (i//8 + i%8)%2==0 else -1 for i in range(64)])
    
    # Add more complex patterns
    p4 = create_random_pattern()  # random pattern
    p5 = np.ones((64,))
    p5[::2] = -1  # alternating pattern
    
    patterns = [p1, p2, p3, p4, p5]

    # Initialize Hopfield network
    hn = HopfieldNetwork(size=64)
    hn.store_patterns(patterns)

    # 1. Basic Pattern Recovery Demonstration
    test_pattern = p1
    noisy_pattern = test_pattern.copy()
    flip_indices = np.random.choice(len(noisy_pattern), 10, replace=False)
    noisy_pattern[flip_indices] *= -1
    retrieved = hn.recall(noisy_pattern)

    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1)
    show_pattern(test_pattern, "Original")
    plt.subplot(1,3,2)
    show_pattern(noisy_pattern, "Noisy")
    plt.subplot(1,3,3)
    show_pattern(retrieved, "Retrieved")
    plt.tight_layout()
    plt.savefig("results/hopfield_results/basic_retrieval.png")
    plt.close()

    # 2. Energy Landscape Analysis
    analyze_convergence(hn, test_pattern)

    # 3. Basin of Attraction Analysis
    analyze_basin_of_attraction(hn, patterns)

    # 4. Pattern Overlap Analysis
    analyze_pattern_overlap(patterns)

    # 5. Capacity Analysis
    theoretical_capacity = hn.theoretical_capacity()
    print(f"\nTheoretical Storage Capacity: {theoretical_capacity} patterns")
    
    # Test actual capacity
    success_rates = []
    n_patterns_range = range(1, 15)
    for n in n_patterns_range:
        test_patterns = [create_random_pattern() for _ in range(n)]
        hn.store_patterns(test_patterns)
        
        successes = 0
        trials = 20
        for pattern in test_patterns:
            noisy = pattern.copy()
            flip_indices = np.random.choice(len(noisy), 5, replace=False)
            noisy[flip_indices] *= -1
            retrieved = hn.recall(noisy)
            if np.array_equal(retrieved, pattern):
                successes += 1
        success_rates.append(successes / (n * trials))

    plt.figure(figsize=(10, 6))
    plt.plot(n_patterns_range, success_rates, '-o')
    plt.axvline(x=theoretical_capacity, color='r', linestyle='--', 
                label=f'Theoretical Capacity ({theoretical_capacity})')
    plt.xlabel('Number of Stored Patterns')
    plt.ylabel('Recovery Success Rate')
    plt.title('Network Capacity Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/hopfield_results/capacity_analysis.png")
    plt.close()

    print("\nAnalysis Results:")
    print("1. Basic pattern retrieval demonstration saved as 'basic_retrieval.png'")
    print("2. Energy convergence analysis saved as 'energy_convergence.png'")
    print("3. Basin of attraction analysis saved as 'basin_of_attraction.png'")
    print("4. Pattern overlap analysis saved as 'pattern_overlap.png'")
    print("5. Capacity analysis saved as 'capacity_analysis.png'")