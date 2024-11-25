import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_data(n_samples=100):
        np.random.seed(0)
        
        # Generate random points in 2D space
        X = np.random.randn(n_samples, 2) * 2
        
        # Create circular decision boundary
        # Points inside the circle are class 1, outside are class -1
        # Calculate distance from origin for each point
        distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        
        # Assign labels based on distance (inside or outside unit circle)
        y = (distances < 1.5).astype(int) * 2 - 1
        y = y.reshape(-1, 1)
        
        return X, y

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation):
        np.random.seed(0)
        self.lr = lr
        self.hidden_activation = activation  # Store activation type for hidden layer
        self.output_activation = 'tanh'  # Fixed output activation
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        # Store activations and gradients for visualization
        self.hidden_features = None
        self.gradients = {'W1': None, 'W2': None}

    def activation(self, x, activation_type):
        if activation_type == 'tanh':
            return np.tanh(x)
        elif activation_type == 'relu':
            return np.maximum(0, x)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -700, 700)))  # Clip to prevent overflow
        
    def activation_derivative(self, x, activation_type):
        if activation_type == 'tanh':
            return 1 - np.tanh(x)**2
        elif activation_type == 'relu':
            return (x > 0).astype(float)
        elif activation_type == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(x, -700, 700)))
            return s * (1 - s)

    def forward(self, X):
        # First layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1, self.hidden_activation)
        self.hidden_features = self.a1  # Store for visualization
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation(self.z2, self.output_activation)
        
        return self.a2

    def backward(self, X, y):
        """
        Compute gradients and update weights using backpropagation
        """
        m = X.shape[0]
        
        # Output layer gradients
        delta2 = (self.a2 - y) * self.activation_derivative(self.z2, self.output_activation)
        dW2 = (1/m) * np.dot(self.a1.T, delta2)
        db2 = (1/m) * np.sum(delta2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1, self.hidden_activation)
        dW1 = (1/m) * np.dot(X.T, delta1)
        db1 = (1/m) * np.sum(delta1, axis=0, keepdims=True)
        
        # Store gradients for visualization
        self.gradients['W1'] = dW1
        self.gradients['W2'] = dW2
        
        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Clear all axes for new frame
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    # Set titles for all three visualizations
    ax_hidden.set_title(f'Hidden Space Transform (3D) at Step {frame * 10}')
    ax_input.set_title(f'Decision Boundary at Step {frame * 10}')
    ax_gradient.set_title(f'Network Gradients at Step {frame * 10}')
    
    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Create grid for visualizations
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 30),
                        np.linspace(y_min, y_max, 30))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions and hidden features for grid points
    Z = mlp.forward(grid_points).reshape(xx.shape)
    grid_hidden_features = mlp.hidden_features
    
    ###########################################
    # VISUALIZATION 1: HIDDEN SPACE (3D)      #
    ###########################################
    
    # Forward pass on actual data points to get their hidden representations
    mlp.forward(X)
    data_hidden_features = mlp.hidden_features
    
    # Calculate the bounds for the 3D plot based on actual data points
    margin = 0.1  # Add some margin around the data
    x_min, x_max = data_hidden_features[:, 0].min(), data_hidden_features[:, 0].max()
    y_min, y_max = data_hidden_features[:, 1].min(), data_hidden_features[:, 1].max()
    z_min, z_max = data_hidden_features[:, 2].min(), data_hidden_features[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_min -= x_range * margin
    x_max += x_range * margin
    y_min -= y_range * margin
    y_max += y_range * margin
    z_min -= z_range * margin
    z_max += z_range * margin
    
    # Create grid for surface plot
    xx_plane, yy_plane = np.meshgrid(
        np.linspace(x_min, x_max, 20),
        np.linspace(y_min, y_max, 20)
    )
    
    # Plot decision boundary plane
    w = mlp.W2.flatten()
    b = mlp.b2.flatten()
    zz_plane = -(w[0] * xx_plane + w[1] * yy_plane + b[0]) / (w[2] + 1e-8)
    
    # Plot the decision boundary plane
    ax_hidden.plot_surface(xx_plane, yy_plane, zz_plane,
                          alpha=0.3, color='yellow')
    
    # Plot the actual data points in hidden space with reversed color scheme
    scatter = ax_hidden.scatter(data_hidden_features[:, 0],
                              data_hidden_features[:, 1],
                              data_hidden_features[:, 2],
                              c=y.ravel(), cmap='bwr_r',
                              s=50, alpha=1.0)
    
    # Set 3D plot properties with dynamic bounds
    ax_hidden.set_xlim(x_min, x_max)
    ax_hidden.set_ylim(y_min, y_max)
    ax_hidden.set_zlim(z_min, z_max)
    
    # Adjust view angle for better visibility
    ax_hidden.view_init(elev=30, azim=45)
    
    # Make axis labels more readable
    ax_hidden.set_xlabel('Hidden 1', labelpad=10)
    ax_hidden.set_ylabel('Hidden 2', labelpad=10)
    ax_hidden.set_zlabel('Hidden 3', labelpad=10)
    
    # Adjust grid properties for better visibility
    ax_hidden.grid(True, alpha=0.3)
    
    ############################################
    # VISUALIZATION 2: DECISION BOUNDARY       #
    ############################################
    
    # Create grid for decision boundary
    x_min, x_max = -4, 4  # Fixed range to match data generation
    y_min, y_max = -4, 4
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions for grid points
    Z = mlp.forward(grid_points).reshape(xx.shape)
    
    # Create filled contour plot for decision regions
    colors = ['#FF6B6B', '#4ECDC4']  # Red for outside (class -1), Blue for inside (class 1)
    levels = [-1, 0, 1]
    cf = ax_input.contourf(xx, yy, Z, levels=levels, colors=colors, alpha=0.5)
    
    # Add decision boundary line
    ax_input.contour(xx, yy, Z, levels=[0], colors='white', linewidths=2)
    
    # Plot original data points
    scatter = ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                             cmap='bwr_r',  # Use regular blue-white-red colormap
                             edgecolors='black', linewidth=0.5, s=50)
    
    # Set input space plot limits and labels
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)
    
    ##############################################
    # VISUALIZATION 3: GRADIENT FLOW            #
    ##############################################
    
    # Define neuron positions
    input_neurons = [(0, 0), (0, 1)]
    hidden_neurons = [(0.4, 0), (0.4, 0.5), (0.4, 1)]
    output_neurons = [(0.8, 0.5)]
    
    # Plot neurons
    node_size = 1000
    for pos in input_neurons:
        ax_gradient.scatter(pos[0], pos[1], s=node_size, c='lightblue', 
                          edgecolors='black', linewidth=2)
    for pos in hidden_neurons:
        ax_gradient.scatter(pos[0], pos[1], s=node_size, c='lightblue',
                          edgecolors='black', linewidth=2)
    for pos in output_neurons:
        ax_gradient.scatter(pos[0], pos[1], s=node_size, c='lightblue',
                          edgecolors='black', linewidth=2)
    
    # Plot connections with gradient-based thickness
    if mlp.gradients['W1'] is not None and mlp.gradients['W2'] is not None:
        max_gradient = max(np.abs(mlp.gradients['W1']).max(),
                         np.abs(mlp.gradients['W2']).max())
        
        if max_gradient > 0:
            # Input to hidden connections
            for i, input_pos in enumerate(input_neurons):
                for j, hidden_pos in enumerate(hidden_neurons):
                    gradient = np.abs(mlp.gradients['W1'][i, j]) / max_gradient
                    ax_gradient.plot([input_pos[0], hidden_pos[0]],
                                   [input_pos[1], hidden_pos[1]],
                                   color='purple', linewidth=1 + 8 * gradient,
                                   alpha=0.6)
            
            # Hidden to output connections
            for i, hidden_pos in enumerate(hidden_neurons):
                for output_pos in output_neurons:
                    gradient = np.abs(mlp.gradients['W2'][i, 0]) / max_gradient
                    ax_gradient.plot([hidden_pos[0], output_pos[0]],
                                   [hidden_pos[1], output_pos[1]],
                                   color='purple', linewidth=1 + 8 * gradient,
                                   alpha=0.6)
    
    # Add labels
    for i, pos in enumerate(input_neurons):
        ax_gradient.annotate(f'x{i+1}', xy=(pos[0], pos[1] + 0.1), 
                           ha='center', fontsize=12)
    for i, pos in enumerate(hidden_neurons):
        ax_gradient.annotate(f'h{i+1}', xy=(pos[0], pos[1] + 0.1),
                           ha='center', fontsize=12)
    ax_gradient.annotate('y', xy=(output_neurons[0][0], output_neurons[0][1] + 0.1),
                        ha='center', fontsize=12)
    
    ax_gradient.set_xlim(-0.2, 1.0)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.axis('off')
    
    return ax_hidden, ax_input, ax_gradient

def visualize(activation, lr, step_num):
    # Generate circular dataset
    X, y = generate_data(n_samples=100)
    
    # Initialize MLP
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    
    # Set up the visualization
    matplotlib.use('agg')
    plt.style.use('default')
    fig = plt.figure(figsize=(21, 7))
    
    # Add spacing between subplots
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    # Create subplots (now with 3D plot for hidden space)
    ax_hidden = fig.add_subplot(gs[0], projection='3d')
    ax_input = fig.add_subplot(gs[1])
    ax_gradient = fig.add_subplot(gs[2])
    
    # Create animation
    frames = max(1, step_num // 10)
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, 
                                   ax_hidden=ax_hidden, ax_gradient=ax_gradient,
                                   X=X, y=y), 
                       frames=frames, 
                       interval=100,
                       blit=False,
                       repeat=False)
    
    # Save animation
    ani.save(os.path.join(result_dir, "visualize.gif"), 
            writer='pillow', fps=10, dpi=150)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)