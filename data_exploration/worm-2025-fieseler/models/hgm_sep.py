import jax
import jax.numpy as jnp
import optax


def init_params(seed, shape_X, shape_X_man):
	return {
		'W1': 1E-2 * jax.random.normal(seed, shape_X),	# randomly initialize weights
		'W2': 1E-2 * jax.random.normal(seed, shape_X_man)
	}


def predict(W_1, W_2, X, X_man):
	return X @ W_1 * jax.nn.sigmoid(X_man @ W_2)	# (n, p) @ (p, k) → (n, k)


def loss_fn(params, X, X_man, Y):
	Y_pred = predict(params['W1'], params['W2'], X, X_man)
	return jnp.mean((Y_pred - Y) ** 2)  # MSE

loss_fn_grad = jax.jit(jax.grad(loss_fn))


def update(opt_update, opt_state, params, X, X_man, Y):
	grads = loss_fn_grad(params, X, X_man, Y)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state

update = jax.jit(update, static_argnums=(0,))
