import os

from functools import partial

import jax
import jax.numpy as jnp
import optax

@jax.jit
def loss_fn_cnn10(predictions, target_data):
    # Compute softmax cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions,target_data)
    return jnp.mean(loss)

@jax.jit
def loss_fn_wine(predictions, target_data):
    # Compute softmax cross-entropy loss
    loss = optax.squared_error(predictions,target_data)
    return jnp.mean(loss)

@partial(jax.jit, static_argnums=[1, 4])
@partial(jax.value_and_grad, argnums=0)
def forward(weights, model, input_data, target_data, loss_fn):
    prediction = model.apply(weights, input_data)
    return loss_fn(prediction, target_data)

def prediction(weights, model, input_data):
    prediction = model.apply(weights, input_data)
    return prediction

def validate_cifar(validation_data, model, weights) -> float:
    """Calculate validation loss for model and weights.
    
    :param validation_data: validation data batches
    :type validation_data:
    :param model: model structure
    :type model:
    :param weights: weights to be applied to model
    :type weights:
    :returns: validation data loss for model with weights
    :rtype: float
    """
    total_loss = 0.0
    num_batches = 0
    for batch in validation_data:
        input_data, target_data = batch
        input_data = jnp.array(input_data)
        target_data = jnp.array(target_data)
        loss_v, _ = forward(weights, model, input_data, target_data, loss_fn_cnn10)
        total_loss += loss_v
        num_batches += 1
    return total_loss / num_batches

def validate_wine(validation_data, model, weights) -> float:
    """Calculate validation loss for model and weights.
    
    :param validation_data: validation data batches
    :type validation_data:
    :param model: model structure
    :type model:
    :param weights: weights to be applied to model
    :type weights:
    :returns: validation data loss for model with weights
    :rtype: float
    """
    total_loss = 0.0
    num_batches = 0
    for batch in validation_data:
        input_data, target_data = batch
        input_data = jnp.array(input_data)
        target_data = jnp.array(target_data)
        loss_v, _ = forward(weights, model, input_data, target_data, loss_fn_wine)
        total_loss += loss_v
        num_batches += 1
    return total_loss / num_batches