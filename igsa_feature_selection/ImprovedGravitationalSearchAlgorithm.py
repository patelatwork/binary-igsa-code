import numpy as np
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

