import jax
import jax.numpy as jnp
import optax


def get_type():
	return True


def init_params(seed, shape_X, shape_X_man):
	return {
		'W1': 1E-2 * jax.random.normal(seed, shape_X),  # randomly initialize weights
		'W2': 1E-2 * jax.random.normal(seed, shape_X_man)
	}


def predict(X, X_man, W1, W2):
	return X @ W1 * jax.nn.sigmoid(X_man @ W2)	# (n, p) @ (p, k) → (n, k)


def loss_fn(params, X, X_man, Y, reg_lambdas):
	W1, W2 = params['W1'], params['W2']
	reg_l1, reg_l2 = reg_lambdas[0], reg_lambdas[1]

	Y_pred = predict(X, X_man, W1, W2)

	mse_loss = jnp.mean((Y_pred - Y) ** 2)  # MSE
	w1_loss = reg_l1 * jnp.linalg.norm(W1) ** 2
	w2_loss = reg_l2 * jnp.linalg.norm(W2) ** 2  # todo: correct norm and squaring?

	res_loss = mse_loss + w1_loss + w2_loss

	return res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4,))


def update(opt_update, opt_state, params, X, X_man, Y, reg_lambdas):
	grads = loss_fn_grad(params, X, X_man, Y, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6))
