import jax
import jax.numpy as jnp
import optax


def get_type():
	return False


def init_params(seeds, shape_U, shape_V, shape_N):
	return {
	    'W': 1E-2 * jax.random.normal(seeds[0], shape_N),
	    'U': 1E-2 * jax.random.normal(seeds[1], shape_U),
	    'V': 1E-2 * jax.random.normal(seeds[2], shape_V)
	}


@jax.jit
def predict(B_ll, W):
	return B_ll @ W


# temp
def mse_loss_fn(params, N, B_ll):
	U, V, W = params['U'], params['V'], params['W']

	N_man = N @ U
	N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_res = N - N_rec
	Y_pred = predict(B_ll, W)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)

	return mse_loss


@jax.jit
def cca_loss(X_proj, Y_proj):
	eps = 1e-8

	X_centered = X_proj - jnp.mean(X_proj, axis=0, keepdims=True)
	Y_centered = Y_proj - jnp.mean(Y_proj, axis=0, keepdims=True)

	cov = jnp.mean(X_centered * Y_centered, axis=0)
	std_X = jnp.std(X_centered, axis=0) + eps
	std_Y = jnp.std(Y_centered, axis=0) + eps

	corrs = cov / (std_X * std_Y)

	return jnp.mean(1 - corrs)

def loss_fn(params, N, B_ll, B_hl, loss_lambdas, reg_lambda):
	loss1_lambda, loss2_lambda = loss_lambdas
	w_reg_lambda = reg_lambda

	U, V, W = params['U'], params['V'], params['W']

	# manifold ================================
	N_man = N @ U
	N_man_bhl = B_hl @ V

	N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_res = N - N_rec

	man_loss = cca_loss(N_man, N_man_bhl)
	# =========================================

	# residuals ===============================
	Y_pred = predict(B_ll, W)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)
	w_reg_loss = jnp.linalg.norm(W) ** 2

	res_loss = mse_loss + w_reg_loss * w_reg_lambda
	# =========================================

	return man_loss * loss1_lambda + res_loss * loss2_lambda

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5))


def update(opt_update, opt_state, params, N, B_ll, B_hl, loss_weights, reg_lambda):
	grads = loss_fn_grad(params, N, B_ll, B_hl, loss_weights, reg_lambda)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7))
