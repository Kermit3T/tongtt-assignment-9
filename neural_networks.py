import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation):
        np.random.seed(0)
        self.lr = lr 
        self.activation_fn = activation # activation function from user

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
            return 1 / (1 + np.exp(-x))
        
    def activation_derivative(self, x, activation_type):
        if activation_type == 'tanh':
            return 1 - np.tanh(x)**2
        elif activation_type == 'relu':
            return (x > 0).astype(float)
        elif activation_type == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
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
        m = X.shape[0]
        
        # Output layer gradients
        if self.output_activation == 'tanh':
            delta2 = (self.a2 - y) * (1 - self.a2**2)
        else:
            delta2 = (self.a2 - y) * self.activation_derivative(self.z2, self.output_activation)
            
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.z1, self.hidden_activation)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        # Store gradients for visualization
        self.gradients['W1'] = dW1
        self.gradients['W2'] = dW2
        
        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Clear all axes for new frame
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    
    # Set titles for all three visualizations
    ax_input.set_title(f'Decision Boundary Learning at Step {frame * 10}')
    ax_hidden.set_title(f'Feature Space Transformation at Step {frame * 10}')
    ax_gradient.set_title(f'Network Architecture & Gradients at Step {frame * 10}')
    
    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Create grid for visualization
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    ###########################################
    # VISUALIZATION 1: INPUT SPACE           #
    ###########################################
    
    # Get predictions for the grid points
    Z = mlp.forward(grid_points).reshape(xx.shape)
    
    # Plot decision boundary
    ax_input.contourf(xx, yy, Z, levels=[-1, 0, 1], 
                     colors=['#FF9999', '#9999FF'], alpha=0.5)
    ax_input.contour(xx, yy, Z, levels=[0], colors='white', linewidths=2)
    
    # Plot training points
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr',
                    edgecolors='black', linewidth=1, s=50)
    
    # Draw the true circular decision boundary
    circle = plt.Circle((0, 0), 1, color='gray', 
                       fill=False, linestyle='--', linewidth=2)
    ax_input.add_artist(circle)
    
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)
    ax_input.set_xlabel('X₁')
    ax_input.set_ylabel('X₂')
    
    ###########################################
    # VISUALIZATION 2: HIDDEN SPACE          #
    ###########################################
    
    # Get hidden layer activations for grid points
    mlp.forward(grid_points)
    hidden_features_grid = mlp.hidden_features.reshape(-1, 3)
    
    # Get hidden layer activations for data points
    mlp.forward(X)
    hidden_features_data = mlp.hidden_features
    
    # Create 3D scatter plot of hidden features
    scatter = ax_hidden.scatter(hidden_features_data[:, 0],
                              hidden_features_data[:, 1],
                              hidden_features_data[:, 2],
                              c=y.ravel(), cmap='bwr',
                              s=50, alpha=1.0)
    
    # Plot decision boundary plane in hidden space
    w = mlp.W2.flatten()
    b = mlp.b2.flatten()
    xx_plane, yy_plane = np.meshgrid(np.linspace(-2, 2, 20),
                                    np.linspace(-2, 2, 20))
    zz_plane = -(w[0] * xx_plane + w[1] * yy_plane + b[0]) / (w[2] + 1e-8)
    
    ax_hidden.plot_surface(xx_plane, yy_plane, zz_plane,
                          alpha=0.3, color='gray')
    
    ax_hidden.set_xlim(-2, 2)
    ax_hidden.set_ylim(-2, 2)
    ax_hidden.set_zlim(-2, 2)
    ax_hidden.set_xlabel('h₁')
    ax_hidden.set_ylabel('h₂')
    ax_hidden.set_zlabel('h₃')
    ax_hidden.view_init(elev=20, azim=45)

    ##############################################
    # VISUALIZATION 3: NETWORK ARCHITECTURE      #
    ##############################################
    
    # Define neuron positions
    input_neurons = [(0, 0), (0, 1)]
    hidden_neurons = [(0.4, 0), (0.4, 0.5), (0.4, 1)]
    output_neurons = [(0.8, 0.5)]
    
    # Set node appearance
    node_size = 1200
    node_color = 'royalblue'
    edge_color = 'black'
    
    # Plot all neurons
    for pos in input_neurons:
        ax_gradient.scatter(pos[0], pos[1], s=node_size, c=node_color, 
                          edgecolors=edge_color, linewidth=2, zorder=3)
    for pos in hidden_neurons:
        ax_gradient.scatter(pos[0], pos[1], s=node_size, c=node_color, 
                          edgecolors=edge_color, linewidth=2, zorder=3)
    for pos in output_neurons:
        ax_gradient.scatter(pos[0], pos[1], s=node_size, c=node_color, 
                          edgecolors=edge_color, linewidth=2, zorder=3)
    
    # Plot connections with gradient-based thickness
    if mlp.gradients['W1'] is not None and mlp.gradients['W2'] is not None:
        max_gradient = max(np.abs(mlp.gradients['W1']).max(), 
                         np.abs(mlp.gradients['W2']).max())
        
        if max_gradient > 0:
            # Plot input to hidden connections
            for i, input_pos in enumerate(input_neurons):
                for j, hidden_pos in enumerate(hidden_neurons):
                    gradient = np.abs(mlp.gradients['W1'][i, j]) / max_gradient
                    ax_gradient.plot([input_pos[0], hidden_pos[0]], 
                                   [input_pos[1], hidden_pos[1]],
                                   color='purple', linewidth=1 + 8 * gradient,
                                   alpha=0.7, zorder=2)
            
            # Plot hidden to output connections
            for i, hidden_pos in enumerate(hidden_neurons):
                for j, output_pos in enumerate(output_neurons):
                    gradient = np.abs(mlp.gradients['W2'][i, j]) / max_gradient
                    ax_gradient.plot([hidden_pos[0], output_pos[0]],
                                   [hidden_pos[1], output_pos[1]],
                                   color='purple', linewidth=1 + 8 * gradient,
                                   alpha=0.7, zorder=2)
    
    # Add labels to neurons
    label_offset = 0.15
    for i, pos in enumerate(input_neurons):
        ax_gradient.annotate(f'x{i+1}', xy=(pos[0], pos[1] + label_offset), 
                           fontsize=12, ha='center', va='bottom')
    
    for i, pos in enumerate(hidden_neurons):
        ax_gradient.annotate(f'h{i+1}', xy=(pos[0], pos[1] + label_offset), 
                           fontsize=12, ha='center', va='bottom')
    
    ax_gradient.annotate('y', xy=(output_neurons[0][0], output_neurons[0][1] + label_offset), 
                        fontsize=12, ha='center', va='bottom')
    
    # Set network visualization plot properties
    ax_gradient.set_xlim(-0.2, 1.0)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.axis('off')
    
    return ax_hidden, ax_input, ax_gradient

def visualize(activation, lr, step_num):
    X, y = generate_data(n_samples=100)
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    
    matplotlib.use('agg')
    plt.style.use('default')
    fig = plt.figure(figsize=(21, 7))
    
    # Add spacing between subplots
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    
    ax_hidden = fig.add_subplot(gs[0], projection='3d')
    ax_input = fig.add_subplot(gs[1])
    ax_gradient = fig.add_subplot(gs[2])
    
    frames = max(1, step_num // 10)
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, 
                                   ax_hidden=ax_hidden, ax_gradient=ax_gradient,
                                   X=X, y=y), 
                       frames=frames, 
                       interval=100,
                       blit=False,
                       repeat=False)
    
    ani.save(os.path.join(result_dir, "visualize.gif"), 
            writer='pillow', fps=10, dpi=150)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)