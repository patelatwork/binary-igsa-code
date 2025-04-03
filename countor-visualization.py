import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
from igsa_feature_selection.ImprovedGravitationalSearchAlgorithm import ImprovedGravitationalSearchAlgorithm

class DataCollector:
    """Collect data during IGSA execution for visualization"""
    def __init__(self):
        self.iterations = []
        self.feature_counts = []
        self.accuracies = []
        self.dataset_names = []
        
    def add_data_point(self, iteration, feature_count, accuracy, dataset_name):
        self.iterations.append(iteration)
        self.feature_counts.append(feature_count)
        self.accuracies.append(accuracy)
        self.dataset_names.append(dataset_name)

def modified_feature_selection(igsa, X, y, dataset_name, collector):
    """Modified feature selection to collect data at each iteration"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Run IGSA with data collection
    for iteration in range(igsa.max_iterations):
        # Update algorithm
        igsa.update_velocity_and_position(X_train, y_train, iteration)
        
        # Calculate accuracy on validation set
        selected_features = igsa.gbest == 1
        if np.sum(selected_features) > 0:  # Ensure at least one feature is selected
            X_val_selected = X_val[:, selected_features]
            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(X_train[:, selected_features], y_train)
            y_pred = clf.predict(X_val_selected)
            accuracy = accuracy_score(y_val, y_pred)
        else:
            accuracy = 0
        
        # Add data point to collector
        collector.add_data_point(
            iteration + 1,  # 1-indexed for better visualization
            np.sum(selected_features),
            accuracy,
            dataset_name
        )
    
    # Return selected features
    return igsa.gbest == 1

def create_3d_animation(collector, output_path="output/3d_animation.gif"):
    """Create animated 3D plot showing the relationship between iterations, features, and accuracy"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Unique dataset names
    dataset_names = np.unique(collector.dataset_names)
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle('3D Feature Selection Visualization', fontsize=16)
    
    # Max iterations
    max_iter = max(collector.iterations)
    max_features = max(collector.feature_counts)
    
    # Create a grid for the surface plot
    feature_grid = np.linspace(0, max_features, 50)
    iter_grid = np.linspace(1, max_iter, 50)
    X_grid, Y_grid = np.meshgrid(feature_grid, iter_grid)
    
    # Setup 3D plot
    ax.set_xlabel('Number of Features Selected')
    ax.set_ylabel('Iterations')
    ax.set_zlabel('Accuracy')
    ax.set_xlim(0, max_features)
    ax.set_ylim(0, max_iter)
    ax.set_zlim(0, 1.05)
    
    # Store all plot elements
    scatter_plots = []
    path_lines = []
    surface = None  # Will hold the surface plot
    
    # Initialize interpolation grid
    Z_grid = np.zeros_like(X_grid)
    
    # Function to update the animation frame
    def update(frame):
        nonlocal surface, scatter_plots, path_lines, Z_grid
        
        # Clear previous surface if it exists
        if surface is not None:
            surface.remove()
        
        # Clear previous scatter plots and path lines
        for scatter in scatter_plots:
            scatter.remove()
        scatter_plots = []
        
        for line in path_lines:
            line.remove()
        path_lines = []
        
        # Get data up to current frame (iteration)
        mask = [i <= frame for i in collector.iterations]
        current_iterations = [collector.iterations[i] for i, flag in enumerate(mask) if flag]
        current_features = [collector.feature_counts[i] for i, flag in enumerate(mask) if flag]
        current_accuracies = [collector.accuracies[i] for i, flag in enumerate(mask) if flag]
        current_datasets = [collector.dataset_names[i] for i, flag in enumerate(mask) if flag]
        
        # If we have data points, create the surface plot
        Z_grid.fill(0)  # Reset Z_grid to zeros
        
        if current_iterations:
            # Simple inverse distance weighting for interpolation
            for i in range(len(current_iterations)):
                distances = np.sqrt((X_grid - current_features[i])**2 + (Y_grid - current_iterations[i])**2)
                weights = 1.0 / (distances + 0.1)  # Add small value to avoid division by zero
                Z_grid += weights * current_accuracies[i]
            
            # Normalize weights - sum along both dimensions
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                Z_grid = Z_grid / weights_sum
            
            # Create 3D surface plot
            surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.7, 
                                      linewidth=0, antialiased=True)
        else:
            # Create an empty surface if no data points yet
            surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.7, 
                                      linewidth=0, antialiased=True)
        
        # Dataset-specific scatter plots and trajectory paths
        for i, dataset in enumerate(dataset_names):
            # Filter data for this dataset
            idx = [j for j, name in enumerate(current_datasets) if name == dataset]
            if idx:
                x = [current_features[j] for j in idx]
                y = [current_iterations[j] for j in idx]
                z = [current_accuracies[j] for j in idx]
                
                # 3D scatter plot for this dataset
                scatter = ax.scatter(x, y, z, c=colors[i], marker=markers[i], s=70, 
                                    edgecolor='black', label=dataset)
                scatter_plots.append(scatter)
                
                # Plot the optimization path
                line, = ax.plot(x, y, z, color=colors[i], linestyle='-', linewidth=2, alpha=0.8)
                path_lines.append(line)
        
        # Update title with current iteration
        ax.set_title(f'Iteration {frame}/{max_iter}')
        
        # Add legend if not yet added
        if frame == 1 and dataset_names.size > 0:
            ax.legend(loc='upper right')
        
        # Adjust view angle to show progress better (optional rotating view)
        ax.view_init(elev=30, azim=frame * (360 / max_iter) % 360)
        
        # Return a list of artists that need to be redrawn
        return [surface] + scatter_plots + path_lines
    
    # Create initial frame (important to ensure at least one frame is created)
    update(1)
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=range(1, max_iter + 1),
                        blit=False, repeat=True, interval=200)
    
    # Save animation
    ani.save(output_path, writer='pillow', fps=5, dpi=100)
    print(f"Animation saved to {output_path}")
    
    # Save a static final view
    plt.tight_layout()
    plt.savefig(output_path.replace('.gif', '.png'))
    
    return fig, ani

def run_igsa_with_data_collection():
    """Run IGSA on multiple datasets and collect data for visualization"""
    # Load datasets
    datasets = [
        (load_breast_cancer(return_X_y=True), 'Breast Cancer'),
        (load_wine(return_X_y=True), 'Wine'),
        (load_iris(return_X_y=True), 'Iris')
    ]
    
    # Create data collector
    collector = DataCollector()
    
    # Run IGSA on each dataset
    for (X, y), dataset_name in datasets:
        print(f"Processing {dataset_name} dataset...")
        
        # Initialize IGSA
        igsa = ImprovedGravitationalSearchAlgorithm(
            n_features=X.shape[1],
            population_size=20,
            max_iterations=50,  # Reduced for faster execution
            w=0.8
        )
        
        # Run modified feature selection
        selected_features = modified_feature_selection(igsa, X, y, dataset_name, collector)
        
        # Print summary
        print(f"Selected {np.sum(selected_features)} features out of {X.shape[1]}")
    
    return collector

def main():
    """Main function to run the visualization"""
    print("Starting feature selection with data collection...")
    collector = run_igsa_with_data_collection()
    
    print("Creating 3D animation...")
    fig, ani = create_3d_animation(collector)
    
    print("Done! Animation and final frame have been saved to the output directory.")
    print("You can also run this file with the --show flag to display the plots interactively.")
    
    # Show plots if requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--show':
        plt.show()

if __name__ == "__main__":
    main()