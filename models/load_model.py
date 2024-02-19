# load_model.py
import os
from tensorflow.keras.models import load_model

def get_pretrained_model(model_name):
    """
    Loads a pre-trained model from the /models/store directory.

    Parameters:
    - model_name: The name of the model to load. This should match the filename (without extension) of the model stored in /models/store.

    Returns:
    - The loaded pre-trained model.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'store')
    model_path = os.path.join(model_dir, f'{model_name}.h5')  # Assuming models are saved in HDF5 format

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No pre-trained model found with the name '{model_name}' in {model_dir}")
    
    # Load and return the model
    loaded_model = load_model(model_path)
    return loaded_model

# Example usage
if __name__ == "__main__":
    # Load a specific pre-trained model by name
    model_name = 'your_pretrained_model_name_here'
    try:
        model = get_pretrained_model(model_name)
        print(f"Successfully loaded model: {model_name}")
    except FileNotFoundError as e:
        print(e)
