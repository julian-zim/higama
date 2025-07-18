import jax
import jax.numpy as jnp
import optax


def get_type():
	return False


def init_params(seeds, shape_X_bll):
	# return {
	# 	'W': 0.4 * jnp.tile(jax.random.normal(seeds[0], shape_X_bll[1]), (shape_X_bll[0], 1))
	# }
	return {
		'W': jax.random.normal(seeds[0], shape_X_bll),
	}


@jax.jit
def predict(X_bll, W):
	return X_bll @ W


# temp
def mse_loss_fn(params, X_bll, Y):
	W = params['W']
	Y_pred = predict(X_bll, W)
	mse_loss = jnp.mean((Y_pred - Y) ** 2)
	return mse_loss


def loss_fn(params, X_bll, Y, reg_lambdas):
	W = params['W']
	w_reg_lambda = reg_lambdas[0]

	Y_pred = predict(X_bll, W)

	mse_loss = jnp.mean((Y_pred - Y) ** 2)
	w_reg_loss = jnp.linalg.norm(W) ** 2

	res_loss = mse_loss + w_reg_loss * w_reg_lambda

	return res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(3,))


def update(opt_update, opt_state, params, X_bll, Y, reg_lambdas):
	grads = loss_fn_grad(params, X_bll, Y, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 5))
