# optimization.py
import torch

def optimize_perturbations_pytorch(initial_perturbations, audio_features, model, perceptual_model, max_iterations=100, alpha=0.01, beta=1.0, epsilon=0.001):
    """
    Optimize the perturbations using PyTorch to ensure they are effective and imperceptible.

    Parameters are similar to the original function but adapted for use with PyTorch tensors.
    """
    # Ensure inputs are PyTorch tensors
    perturbations = torch.tensor(initial_perturbations, dtype=torch.float32, requires_grad=True)
    audio_features = torch.tensor(audio_features, dtype=torch.float32)

    # Define an optimizer for the perturbations
    optimizer = torch.optim.Adam([perturbations], lr=alpha)

    for iteration in range(max_iterations):
        optimizer.zero_grad()  # Reset gradient accumulation

        # Apply current perturbations to the audio features
        cloaked_features = audio_features + perturbations
        
        # Compute loss related to the effectiveness of cloaking
        cloaking_loss = compute_cloaking_loss_pytorch(cloaked_features, model)
        
        # Compute loss related to the perceptual quality of the cloaked audio
        perceptual_loss = compute_perceptual_loss_pytorch(cloaked_features, audio_features, perceptual_model)
        
        # Total loss is a weighted sum of the cloaking loss and perceptual loss
        total_loss = cloaking_loss + beta * perceptual_loss
        
        # Backpropagate the total loss
        total_loss.backward()
        
        # Update the perturbations based on the gradients
        optimizer.step()
        
        # Check for convergence
        with torch.no_grad():  # Temporarily disable gradient tracking
            if torch.mean(torch.abs(perturbations.grad)) < epsilon:
                break

    return perturbations.detach().numpy()  # Convert back to numpy array if necessary
