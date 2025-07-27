import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from functools import partial

def generate_params(m, n, k):
    stddev = jnp.sqrt(2.0 / m) 
    return random.normal(k, (m, n)) * stddev, jnp.zeros((n,))

@jit
def relu(x):
    return jnp.maximum(0, x)

@jit
def softmax(x):
    e = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return e / jnp.sum(e, axis=-1, keepdims=True)

@jit
def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

@jit
def tanh(x):
    return jnp.tanh(x)

activation_map = {
    "relu": relu,
    "softmax": softmax,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "identity": lambda x: x
}

@partial(jit, static_argnames=("activation", "dropout_rate",))
def feedforward(params, activation, x, dropout_rate, k):
    entropy = x
    keys = random.split(k, len(params))
    for act, (w, b), key in zip(activation, params, keys):
        entropy = entropy @ w + b
        if dropout_rate > 0.0:
            mask = random.bernoulli(key, p=1.0 - dropout_rate, shape=entropy.shape)
            entropy = entropy * mask / (1.0 - dropout_rate)
        entropy = act(entropy)
    return entropy
    
@partial(jit, static_argnames=("activation", "loss_name", "num_classes", "regularization", "lambda_", "dropout_rate",))
def compute_loss(params, activation, x, y, loss_name, num_classes, regularization, lambda_, dropout_rate, key):
    
    y_pred = feedforward(params=params, activation=activation, x=x, dropout_rate=dropout_rate, k=key)

    if loss_name == "mse":
        loss_scalar = jnp.mean((y_pred - y) ** 2)

    elif loss_name == "mae":
        loss_scalar = jnp.mean(jnp.abs(y_pred - y))

    elif loss_name == "huber":
        delta = 1.0
        error = y_pred - y
        abs_error = jnp.abs(error)
        quadratic = jnp.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss_scalar = jnp.mean(0.5 * quadratic**2 + delta * linear)

    elif loss_name == "cross_entropy":
        one_hot = jax.nn.one_hot(y, num_classes)
        log_probs = jax.nn.log_softmax(y_pred)
        loss_scalar = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    elif loss_name == "sparse_cross_entropy":
        log_probs = jax.nn.log_softmax(y_pred)
        loss_scalar = -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])

    elif loss_name == "binary_cross_entropy":
        loss_scalar = -jnp.mean(y * jax.nn.log_sigmoid(y_pred) + (1 - y) * jax.nn.log_sigmoid(-y_pred))

    elif loss_name == "focal":
        gamma = 2.0
        alpha = 0.25
        probs = jax.nn.sigmoid(y_pred)
        loss = -alpha * (1 - probs) ** gamma * y * jnp.log(probs + 1e-8) - (1 - alpha) * probs ** gamma * (1 - y) * jnp.log(1 - probs + 1e-8)
        loss_scalar = jnp.mean(loss)

    elif loss_name == "log_cosh":
        loss_scalar = jnp.mean(jnp.log(jnp.cosh(y_pred - y + 1e-12)))
    
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    if regularization == "":
        return loss_scalar
    
    if regularization == "l2":
        l2_penalty = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
        return loss_scalar + lambda_ * l2_penalty
    
    if regularization == "l1":
        l1_penalty = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
        return loss_scalar + lambda_ * l1_penalty
    
    if regularization == "l1_l2":
        l1 = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
        l2 = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
        return loss_scalar + lambda_ * (0.5 * l1 + 0.5 * l2)
    
    raise ValueError(f"Unknown regularization: {regularization}")

@jit
def update_params(params, loss_grad, learning_rate):
    return jax.tree_util.tree_map(lambda p,g: p - learning_rate * g, params, loss_grad)

class MLP():
    def __init__(self, layer_sizes, activation, dropout_rate=0.0, seed=0):
        self.key = random.key(seed)
        keys = random.split(self.key, len(layer_sizes) - 1)
        self.params = [generate_params(m=m, n=n, k=k) for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
        self.activation = tuple(activation_map[a] for a in activation)
        self.dropout = dropout_rate
        self.step = grad(compute_loss, argnums=0)
    
    def predict(self, x):
        self.key, subkey = random.split(self.key)
        return feedforward(params=self.params, activation=self.activation, x=x, dropout_rate=0.0, k=subkey)

    def train(self, xs, ys, loss_name, regularization, num_classes=10, lambda_reg=1e-4, lr=0.001):
        self.key, subkey = random.split(self.key)
        dloss_dparams = self.step(self.params, self.activation, xs, ys, loss_name, num_classes, regularization, lambda_reg, self.dropout, subkey)
        self.params = update_params(params=self.params, loss_grad=dloss_dparams, learning_rate=lr)