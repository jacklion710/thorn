# perturbation_calc.py
import torch

def calculate_perturbations_pytorch(features, model, target_features=None, epsilon=0.01):
    """
    Calculate perturbations to apply to audio for cloaking using PyTorch.

    Parameters:
    - features: np.ndarray or torch.Tensor, the original features of the audio to cloak.
    - model: The PyTorch model used to compute the loss and gradients.
    - target_features: np.ndarray, torch.Tensor or None, the target features to mimic, if any.
    - epsilon: float, scaling factor to ensure perturbations are small.

    Returns:
    - perturbations: torch.Tensor, calculated perturbations to apply to the original audio features.
    """
    # Ensure input is a PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32, requires_grad=True)
    
    # If no target features specified, aim to minimize the model's confidence
    if target_features is None:
        # Predictions from the model
        predictions = model(features_tensor)
        # Assume the model outputs logits and we use the max logit as the loss to maximize
        loss = -torch.max(predictions)
    else:
        # Ensure target features are also a tensor
        target_features_tensor = torch.tensor(target_features, dtype=torch.float32)
        # Calculate the L2 distance between the features and the target features
        loss = torch.norm(features_tensor - target_features_tensor)

    # Calculate gradients of the loss with respect to the input features
    loss.backward()  # This populates the .grad attribute of features_tensor
    gradients = features_tensor.grad

    # Apply epsilon to scale the gradients to small perturbations
    perturbations = epsilon * torch.sign(gradients)

    return perturbations.detach()  # Detach the perturbations from the computation graph

