import jax
import jax.numpy as jnp
import optax


def init_params(seed, shape):
	return {
		'W': 1E-2 * jax.random.normal(seed, shape),  # randomly initialize weights
	}


def predict(W, X):
	return X @ W	# (n, p) @ (p, k) → (n, k)


def loss_fn(params, X, Y):
	Y_pred = predict(params['W'], X)
	return jnp.mean((Y_pred - Y) ** 2)  # MSE

loss_fn_grad = jax.jit(jax.grad(loss_fn))


def update(opt_update, opt_state, params, X, Y):
	grads = loss_fn_grad(params, X, Y)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state

update = jax.jit(update, static_argnums=(0,))
