import jax
import jax.numpy as jnp
import optax


def get_type():
	return True


def init_params(seeds, shape_U, shape_V, shape_N, shape_N_man, shape_N_man_b):
	return {
	    'W1': 1E-2 * jax.random.normal(seeds[0], shape_N),  # todo: rename shape_N
	    'W2': 1E-2 * jax.random.normal(seeds[1], shape_N_man),
		'b': 1E-2 * jax.random.normal(seeds[4], shape_N_man_b),
	    'U': 1E-2 * jax.random.normal(seeds[2], shape_U),
	    'V': 1E-2 * jax.random.normal(seeds[3], shape_V)
	}


@jax.jit
def predict(B_ll, N_man, W1, W2, b):
	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2 + b)


# temp
def mse_loss_fn(params, N, B_ll):
	U, V, W1, W2, b = params['U'], params['V'], params['W1'], params['W2'], params['b']

	N_man = N @ U
	N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_res = N - N_rec
	Y_pred = predict(B_ll, N_man, W1, W2, b)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)

	return mse_loss


@jax.jit
def cca_loss(X_proj, Y_proj, cca_lambdas):
	eps = 1e-6

	X_proj = X_proj - jnp.mean(X_proj, axis=0)
	Y_proj = Y_proj - jnp.mean(Y_proj, axis=0)

	# Compute covariance matrices
	n = X_proj.shape[0]
	s_xx = (X_proj.T @ X_proj) / (n - 1) + eps * jnp.eye(X_proj.shape[1])
	s_yy = (Y_proj.T @ Y_proj) / (n - 1) + eps * jnp.eye(Y_proj.shape[1])
	s_xy = (X_proj.T @ Y_proj) / (n - 1)

	# Whitening transformations
	s_xx_inv_sqrt = jax.scipy.linalg.eigh(s_xx)[1] @ jnp.diag(1.0 / jnp.sqrt(jax.scipy.linalg.eigh(s_xx)[0])) @ jax.scipy.linalg.eigh(s_xx)[1].T
	s_yy_inv_sqrt = jax.scipy.linalg.eigh(s_yy)[1] @ jnp.diag(1.0 / jnp.sqrt(jax.scipy.linalg.eigh(s_yy)[0])) @ jax.scipy.linalg.eigh(s_yy)[1].T

	# Canonical correlations matrix
	t = s_xx_inv_sqrt @ s_xy @ s_yy_inv_sqrt

	# Compute singular values (canonical correlations)
	corr = jnp.linalg.svd(t, compute_uv=False)

	# Positive loss: max correlation sum is rank
	max_corr = min(X_proj.shape[1], Y_proj.shape[1])
	cca_loss = max_corr - jnp.sum(corr)

	return cca_loss

	'''
	ccaunit_lambda, ccadiag_lambda = cca_lambdas

	corr = jax.numpy.corrcoef(X_proj, Y_proj, rowvar=False)
	corr_diag = jnp.diag(jnp.diag(corr))
	corr_nondiag = corr - corr_diag

	return (ccaunit_lambda * jnp.mean((corr_diag - jnp.eye(corr_diag.shape)) ** 2)
			+ ccadiag_lambda * jnp.mean(corr_nondiag ** 2))
	'''

	'''eps = 1e-8

	X_centered = X_proj - jnp.mean(X_proj, axis=0, keepdims=True)
	Y_centered = Y_proj - jnp.mean(Y_proj, axis=0, keepdims=True)

	cov = jnp.mean(X_centered * Y_centered, axis=0)
	std_X = jnp.std(X_centered, axis=0) + eps
	std_Y = jnp.std(Y_centered, axis=0) + eps

	corrs = cov / (std_X * std_Y)

	return jnp.mean(1 - corrs)  # COVARIANCE loss instead of correlation. also, signed correlation?'''


def loss_fn(params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambdas):
	loss1_lambda, loss2_lambda = loss_lambdas
	w1_reg_lambda, w2_reg_lambda = reg_lambdas[0], reg_lambdas[1]

	U, V, W1, W2, b = params['U'], params['V'], params['W1'], params['W2'], params['b']

	# manifold ================================
	N_man = N @ U
	B_hl_man = B_hl @ V

	N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_res = N - N_rec

	man_loss = cca_loss(N_man, B_hl_man, cca_lambdas)
	# =========================================

	# residuals ===============================
	Y_pred = predict(B_ll, N_man, W1, W2, b)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)
	w1_reg_loss = jnp.linalg.norm(W1) ** 2
	w2_reg_loss = jnp.linalg.norm(W2) ** 2

	res_loss = mse_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda
	# =========================================

	return man_loss * loss1_lambda + res_loss * loss2_lambda

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5))


def update(opt_update, opt_state, params, N, B_ll, B_hl, cca_lambdas, loss_weights, reg_lambdas):
	grads = loss_fn_grad(params, N, B_ll, B_hl, loss_weights, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7))
