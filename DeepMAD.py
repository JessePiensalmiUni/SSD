import numpy as np
from scipy.optimize import minimize

def calculateFLOPs(w, Li, image_size=(320, 320)):
    flops = 0
    # Example: iterate through layers, assuming w influences output channels and Li influences depth
    for layer_idx, depth in enumerate(Li):
        for d in range(int(depth)):  # Repeat calculations based on depth
            # Simplified FLOPs calculation for a conv layer
            in_channels = w[layer_idx-1] if layer_idx > 0 else 3  # Assume 3 input channels for the first layer
            out_channels = w[layer_idx]
            kernel_size = 3  # Example kernel size
            flops += 2 * image_size[0] * image_size[1] * in_channels * out_channels * kernel_size ** 2
            image_size = (image_size[0] // 2, image_size[1] // 2)  # Example: halve image size for simplicity
    return flops

def calculateParams(w, Li):
    params = 0
    # Similar loop to calculateFLOPs
    for layer_idx, depth in enumerate(Li):
        for d in range(int(depth)):
            in_channels = w[layer_idx-1] if layer_idx > 0 else 3
            out_channels = w[layer_idx]
            kernel_size = 3
            params += in_channels * out_channels * kernel_size ** 2 + out_channels
    return params

# Hyperparameters and problem definition
alpha = np.array([1, 1, 1, 1, 1,1,1,8])
beta = 10
gamma=5
rho0 = 0.1
L = 8
M = 3
w_initial = np.array([256, 512, 128, 256, 128, 256 ,64 , 128])
Li_initial = np.array([2, 3, 2])
flops_budget = 360000
params_budget = 300000
lambda1, lambda2 = 1e-3, 1e-3  # Regularization factors, adjust as needed


def objective(x, alpha, beta, flops_budget, params_budget, lambda1, lambda2):
    w = x[:L]
    Li = x[L:]
    entropy = -np.sum(alpha * np.log(w))
    variance_penalty = beta * np.exp(np.var(Li))
    flops_penalty = lambda1 * max(0, calculateFLOPs(w, Li, (320, 320)) - flops_budget)
    params_penalty = lambda2 * max(0, calculateParams(w, Li) - params_budget)
    return entropy + variance_penalty + flops_penalty + params_penalty

def objective_for_ssd(x, alpha, beta, gamma, flops_budget, params_budget, lambda1, lambda2, aspect_ratios):
    # x could represent a combination of channel width categories and stage depths
    w_categories = x[:L]  # Categorical representation of width
    Li = x[L:]  # Stage depths
    
    # Adjust the entropy calculation for categorical widths
    entropy = -np.sum(alpha * np.log(w_categories))
    
    # Variance penalty on stage depths to encourage efficient multiscale feature extraction
    variance_penalty = beta * np.exp(np.var(Li))
    
    # Additional penalty on the deviation from preferred aspect ratios
    aspect_ratio_penalty = gamma * np.sum(np.abs(aspect_ratios - calculate_aspect_ratios_from_model_params(w_categories, Li)))
    
    # FLOPs and parameters penalties remain the same
    flops_penalty = lambda1 * max(0, calculateFLOPs(w_categories, Li, aspect_ratios) - flops_budget)
    params_penalty = lambda2 * max(0, calculateParams(w_categories, Li) - params_budget)
    
    return entropy + variance_penalty + aspect_ratio_penalty + flops_penalty + params_penalty

def calculate_aspect_ratios(default_boxes):
    """
    Calculates the aspect ratios of default boxes.

    Args:
        default_boxes (list of tuples): Each tuple contains (width, height) of default boxes at each layer.

    Returns:
        np.array: Array containing the aspect ratios of the default boxes.
    """
    aspect_ratios = []
    for w, h in default_boxes:
        # Calculate the aspect ratio as width / height
        # Ensure division by zero is handled, assuming minimum dimension to be very small
        aspect_ratio = w / max(h, 1e-6)
        aspect_ratios.append(aspect_ratio)
    
    return np.array(aspect_ratios)

def calculate_aspect_ratios_from_model_params(w_categories, Li):
    """
    Calculate aspect ratios from model parameters.

    Args:
        w_categories (np.array): Array representing width categories.
        Li (np.array): Array representing layer depths.

    Returns:
        np.array: Array containing calculated aspect ratios.
    """

    # Placeholder: Convert w_categories and Li into default box sizes
    # This is model-specific and depends on how your w_categories and Li translate into box dimensions
    default_box_sizes = []  # Replace with actual calculation

    for w, h in default_box_sizes:
        # Calculate the aspect ratio as width / height
        # Ensure division by zero is handled by adding a small epsilon to the denominator
        aspect_ratio = w / (h + 1e-6)
        aspect_ratios.append(aspect_ratio)

    return np.array(aspect_ratios)

# Example usage
default_boxes = [(10, 20), (20, 40), (40, 40), (80, 40), (100, 100)]  # Example default box sizes (w, h)
aspect_ratios = calculate_aspect_ratios(default_boxes)
print("Aspect Ratios:", aspect_ratios)

# Bounds
lb = np.concatenate((np.ones(L) * rho0, np.ones(M)))
ub = np.concatenate((np.full(L, 256), np.full(M, 10)))
bounds = [(l, u) for l, u in zip(lb, ub)]

# Initial guess
x0 = np.concatenate((w_initial, Li_initial))

# Define constraints (if any) as functions that return 0 when satisfied
constraints = []

# Optimization call
result = minimize(objective_for_ssd, x0, args=(alpha, beta, gamma,flops_budget, params_budget, lambda1, lambda2,aspect_ratios), bounds=bounds, constraints=constraints, method='SLSQP')

# Extract and display optimized parameters
w_opt = result.x[:L]
Li_opt = result.x[L:]
print('Optimized channel widths:', w_opt)
print('Optimized stage depths:', Li_opt)