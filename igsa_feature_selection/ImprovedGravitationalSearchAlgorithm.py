import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class ImprovedGravitationalSearchAlgorithm:
    def __init__(self, n_features, population_size=20, max_iterations=100, w=0.8):
        """
        Initialize the Improved Gravitational Search Algorithm for Feature Selection
        
        Parameters:
        - n_features: Total number of features in the dataset
        - population_size: Number of particles in the population
        - max_iterations: Maximum number of iterations
        - w: Weighting factor for fitness function (balance between accuracy and feature reduction)
        """
        np.random.seed(42)
        self.n_features = n_features
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.w = w
        
        # Initialize population
        self.population = np.random.randint(2, size=(population_size, n_features))
        self.velocities = np.zeros((population_size, n_features))
        
        # Global best solution
        self.gbest = None
        self.gbest_fitness = float('-inf')
        
        # Tracking performance metrics for visualization
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.feature_count_history = []
        self.exploration_coefficient_history = []
        self.kbest_history = []
    
    def calculate_mass(self, fitness_values):
        """
        Calculate the mass of each particle based on fitness values
        
        Parameters:
        - fitness_values: Fitness values of particles
        
        Returns:
        - Masses of particles
        """
        best = np.max(fitness_values)
        worst = np.min(fitness_values)
        
        # Prevent division by zero
        if best == worst:
            return np.ones(len(fitness_values)) / len(fitness_values)
        
        masses = (fitness_values - worst) / (best - worst)
        return masses / np.sum(masses)
    
    def calculate_fitness(self, X, y, subset):
        """
        Calculate fitness of a feature subset
        
        Parameters:
        - X: Input features
        - y: Target labels
        - subset: Binary vector representing selected features
        
        Returns:
        - Fitness value
        """
        # Select features
        X_subset = X[:, subset == 1]
        
        if X_subset.shape[1] == 0:
            return 0
        
        # Use K-NN classifier with 10-fold cross-validation
        classifier = KNeighborsClassifier(n_neighbors=1)
        accuracies = cross_val_score(classifier, X_subset, y, cv=10)
        accuracy = np.mean(accuracies)
        
        # Calculate feature reduction proportion
        feature_reduction = 1 - (np.sum(subset) / self.n_features)
        
        # Combined fitness function
        fitness = self.w * accuracy + (1 - self.w) * feature_reduction
        
        return fitness
    
    def exponential_kbest(self, iteration):
        """
        Exponentially reduce Kbest with iterations
        
        Parameters:
        - iteration: Current iteration
        
        Returns:
        - Number of particles to consider for force calculation
        """
        per = 2  # Percent of particles at the end
        return int(self.population_size * (per/100) ** (iteration / self.max_iterations))
    
    def update_velocity_and_position(self, X, y, iteration):
        """
        Update velocities and positions of particles
        
        Parameters:
        - X: Input features
        - y: Target labels
        - iteration: Current iteration
        """
        # Calculate fitness for current population
        fitness_values = np.array([
            self.calculate_fitness(X, y, particle) 
            for particle in self.population
        ])
        
        # Update global best
        max_fitness_idx = np.argmax(fitness_values)
        if fitness_values[max_fitness_idx] > self.gbest_fitness:
            self.gbest = self.population[max_fitness_idx].copy()
            self.gbest_fitness = fitness_values[max_fitness_idx]
        
        # Calculate masses
        masses = self.calculate_mass(fitness_values)
        
        # Adaptive exploration-exploitation coefficient
        c1 = 2 - 2 * (iteration / self.max_iterations)**3
        
        # Exponential Kbest
        kbest = self.exponential_kbest(iteration)
        
        # Sort particles by fitness to get top Kbest
        top_indices = np.argsort(fitness_values)[-kbest:]
        
        # Update velocities and positions
        for i in range(self.population_size):
            # Gravitational force component
            force = np.zeros(self.n_features)
            for j in top_indices:
                if i != j:
                    r = np.random.rand()
                    force += r * (masses[j] * (self.population[j] - self.population[i]) / 
                                  (np.linalg.norm(self.population[j] - self.population[i]) + 1e-10))
            
            # Velocity update with global memory
            self.velocities[i] = (np.random.rand() * self.velocities[i] + 
                                  c1 * force + 
                                  (2 - c1) * (self.gbest - self.population[i]))
            
            # Binary position update
            for d in range(self.n_features):
                prob = np.abs(np.tanh(self.velocities[i][d]))
                if np.random.rand() < prob:
                    self.population[i][d] = 1 - self.population[i][d]
        
        # Store metrics for visualization
        self.best_fitness_history.append(self.gbest_fitness)
        self.avg_fitness_history.append(np.mean(fitness_values))
        self.feature_count_history.append(np.sum(self.gbest))
        self.exploration_coefficient_history.append(c1)
        self.kbest_history.append(kbest)
    
    def feature_selection(self, X, y):
        """
        Perform feature selection using Improved Gravitational Search Algorithm
        
        Parameters:
        - X: Input features
        - y: Target labels
        
        Returns:
        - Selected feature indices
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run IGSA
        for iteration in range(self.max_iterations):
            self.update_velocity_and_position(X_scaled, y, iteration)
        
        # Return selected features
        return self.gbest == 1
    
    def plot_convergence(self, dataset_name=""):
        """
        Plot the convergence of the algorithm for various metrics
        
        Parameters:
        - dataset_name: Name of the dataset for the plot titles
        """
        # Create a figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle(f'IGSA Convergence Metrics - {dataset_name}', fontsize=16)
        
        # Plot 1: Fitness values over iterations
        iterations = range(1, len(self.best_fitness_history) + 1)
        axs[0].plot(iterations, self.best_fitness_history, 'b-', label='Best Fitness')
        axs[0].plot(iterations, self.avg_fitness_history, 'r--', label='Average Fitness')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Fitness Value')
        axs[0].set_title('Fitness Convergence')
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot 2: Feature count over iterations
        axs[1].plot(iterations, self.feature_count_history, 'g-')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Number of Selected Features')
        axs[1].set_title('Feature Selection Convergence')
        axs[1].grid(True)
        
        # Plot 3: Exploration coefficient and Kbest
        ax3_twin = axs[2].twinx()
        axs[2].plot(iterations, self.exploration_coefficient_history, 'b-', label='Exploration Coefficient (c1)')
        ax3_twin.plot(iterations, self.kbest_history, 'r--', label='Kbest')
        
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Exploration Coefficient')
        ax3_twin.set_ylabel('Kbest Value')
        axs[2].set_title('Algorithm Parameters Over Time')
        
        # Add legends to both y-axes
        lines1, labels1 = axs[2].get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        return fig