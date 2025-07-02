import jax
import jax.numpy as jnp
import optax


def get_type():
	return False


def init_params(seed, shape):
	return {
		'W': 1E-2 * jax.random.normal(seed, shape),  # randomly initialize weights
	}


def predict(X, W):
	return X @ W	# (n, p) @ (p, k) → (n, k)


def loss_fn(params, X, Y, reg_lambdas):
	W = params['W']
	reg_l = reg_lambdas[0]

	Y_pred = predict(X, W)

	mse_loss = jnp.mean((Y_pred - Y) ** 2)  # MSE
	w_loss = reg_l * jnp.linalg.norm(W) ** 2

	res_loss = mse_loss + w_loss

	return res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(3,))


def update(opt_update, opt_state, params, X, Y, reg_lambdas):
	grads = loss_fn_grad(params, X, Y, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 5))
