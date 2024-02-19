# load_model.py
import os
import torch

def get_pretrained_model(model_name, model_class=None):
    """
    Loads a pre-trained model from the /models/store directory using PyTorch.

    Parameters:
    - model_name: The name of the model to load. This should match the filename (without extension) of the model stored in /models/store.
    - model_class: The class of the model to load. This is necessary for PyTorch models as the architecture needs to be defined before loading the state dictionary.

    Returns:
    - The loaded pre-trained model.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'store')
    model_path = os.path.join(model_dir, f'{model_name}.pt')  # PyTorch models are often saved with .pt or .pth extension

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No pre-trained model found with the name '{model_name}' in {model_dir}")
    
    # Instantiate the model class if provided
    if model_class is not None:
        model = model_class()
    else:
        raise ValueError("Model class must be provided for PyTorch models.")
    
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    return model

# Example usage
if __name__ == "__main__":
    # Define your model class here. This should match the architecture of the saved model.
    # For example, if your model is a ResNet18, you should define/import the ResNet18 class and pass it here.
    class YourModelClass:
        pass  # Replace this with your model's class definition
    
    model_name = 'your_pretrained_model_name_here'
    try:
        model = get_pretrained_model(model_name, YourModelClass)
        print(f"Successfully loaded model: {model_name}")
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
