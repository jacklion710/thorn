# perturbation_calc.py
import tensorflow as tf

def calculate_perturbations(features, model, target_features=None, epsilon=0.01):
    """
    Calculate perturbations to apply to audio for cloaking.

    Parameters:
    - features: np.ndarray, the original features of the audio to cloak.
    - model: The model used to compute the loss and gradients.
    - target_features: np.ndarray or None, the target features to mimic, if any.
    - epsilon: float, scaling factor to ensure perturbations are small.

    Returns:
    - perturbations: np.ndarray, calculated perturbations to apply to the original audio features.
    """
    # If target features are not specified, we aim to minimize the model's confidence.
    if target_features is None:
        # Create a tensor from the original features
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(features_tensor)
            # Predictions from the model
            predictions = model(features_tensor)
            # Assume the model outputs logits and we use the max logit as the loss to maximize
            loss = -tf.reduce_max(predictions)
        
        # Calculate gradients of the loss with respect to the input features
        gradients = tape.gradient(loss, features_tensor)
    else:
        # Mimicking a target implies minimizing the distance between the original and target features
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        target_features_tensor = tf.convert_to_tensor(target_features, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(features_tensor)
            # Calculate the L2 distance between the features and the target features
            loss = tf.norm(features_tensor - target_features_tensor)
        
        # Calculate gradients of the loss with respect to the input features
        gradients = tape.gradient(loss, features_tensor)
    
    # Apply epsilon to scale the gradients to small perturbations
    perturbations = epsilon * tf.sign(gradients)
    
    return perturbations.numpy()
