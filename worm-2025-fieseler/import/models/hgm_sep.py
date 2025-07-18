import jax
import jax.numpy as jnp
import optax
import numpy as np


def get_type():
	return True


def init_params(seeds, shape_X_bll, shape_X_man):
	# return {
	# 	'W1': 0.5 * jnp.tile(jax.random.normal(seeds[0], shape_X_bll[1]), (shape_X_bll[0], 1)),  # + 1E-3 * jax.random.normal(jax.random.PRNGKey(69), shape_X_bll),
	# 	'W2': 0.5 * jnp.tile(jnp.abs(jax.random.normal(seeds[1], shape_X_bll[1])), (shape_X_man[0], 1)) + 1.5 * jax.random.normal(jax.random.PRNGKey(420), shape_X_man)
	# }
	return {
		'W1': jax.random.normal(seeds[0], shape_X_bll),
		'W2': jax.random.normal(seeds[1], shape_X_man)
	}


@jax.jit
def predict(X_bll, X_man, W1, W2):
	return X_bll @ W1 * jax.nn.sigmoid(X_man @ W2)


# temp
def mse_loss_fn(params, X_bll, X_man, Y):
	W1, W2 = params['W1'], params['W2']
	Y_pred = predict(X_bll, X_man, W1, W2)
	mse_loss = jnp.mean((Y_pred - Y) ** 2)
	return mse_loss


def loss_fn(params, X_bll, X_man, Y, reg_lambdas):
	W1, W2 = params['W1'], params['W2']
	w1_reg_lambda, w2_reg_lambda = reg_lambdas[0], reg_lambdas[1]

	Y_pred = predict(X_bll, X_man, W1, W2)

	mse_loss = jnp.mean((Y_pred - Y) ** 2)
	w1_reg_loss = jnp.linalg.norm(W1) ** 2
	w2_reg_loss = jnp.linalg.norm(W2) ** 2  # todo: correct norm and squaring?

	res_loss = mse_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda

	return res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4,))


def update(opt_update, opt_state, params, X_bll, X_man, Y, reg_lambdas):
	grads = loss_fn_grad(params, X_bll, X_man, Y, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6))
