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
@jax.jit
def mse_loss_fn(params, N, B_ll):
	U, V, W = params['U'], params['V'], params['W']

	N_man = N @ U
	# N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N  # gottem
	N_res = N - N_rec
	Y_pred = predict(B_ll, W)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)

	return mse_loss


@jax.jit
def norming(X_proj):
	X_proj = X_proj - jnp.mean(X_proj, axis=0)
	X_proj = X_proj / (jnp.linalg.norm(X_proj, axis=0) + 1e-6)
	return X_proj

@jax.jit
def cca_loss(X_proj, Y_proj, cca_lambdas):
	ccaunit_lambda, ccadiag_lambda = cca_lambdas

	corr = norming(X_proj).T @ norming(Y_proj)

	corr_diag = jnp.diag(jnp.diag(corr))
	corr_nondiag = corr - corr_diag

	return (ccaunit_lambda * jnp.mean(jnp.abs(corr_diag - jnp.eye(*corr_diag.shape)))
			+ ccadiag_lambda * jnp.mean(jnp.abs(corr_nondiag)))

def loss_fn(params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambda):
	loss1_lambda, loss2_lambda = loss_lambdas
	w_reg_lambda = reg_lambda

	U, V, W = params['U'], params['V'], params['W']

	# manifold ================================
	N_man = N @ U
	N_man_bhl = B_hl @ V

	# N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N  # gottem
	N_res = N - N_rec

	man_loss = cca_loss(N_man, N_man_bhl, cca_lambdas)
	# =========================================

	# residuals ===============================
	Y_pred = predict(B_ll, W)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)
	w_reg_loss = jnp.linalg.norm(W) ** 2

	res_loss = mse_loss + w_reg_loss * w_reg_lambda
	# =========================================

	return man_loss * loss1_lambda + res_loss * loss2_lambda

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5, 6))


def update(opt_update, opt_state, params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambda):
	grads = loss_fn_grad(params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambda)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7, 8))
