import jax
import jax.numpy as jnp
import optax


def get_type():
	return True


def init_params(seeds, shape_U, shape_V, shape_N, shape_N_man):
	return {
	    'W1': 1E-2 * jax.random.normal(seeds[0], shape_N),
	    'W2': 1E-2 * jax.random.normal(seeds[1], shape_N_man),
	    'U': 1E-2 * jax.random.normal(seeds[2], shape_U),
	    'V': 1E-2 * jax.random.normal(seeds[3], shape_V)
	}


@jax.jit
def predict(B_ll, N_man, W1, W2):
	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)


# temp
@jax.jit
def mse_loss_fn(params, N, B_ll):
	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	N_man = N @ U
	# N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N  # got'em
	N_res = N - N_rec
	Y_pred = predict(B_ll, N_man, W1, W2)
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

def loss_fn(params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambdas):
	loss1_lambda, loss2_lambda = loss_lambdas
	w1_reg_lambda, w2_reg_lambda = reg_lambdas[0], reg_lambdas[1]

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	# manifold ================================
	N_man = N @ U
	B_hl_man = B_hl @ V

	# N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N  # got'em
	N_res = N - N_rec

	man_loss = cca_loss(N_man, B_hl_man, cca_lambdas)
	# =========================================

	# residuals ===============================
	N_res_pred = predict(B_ll, N_man, W1, W2)
	mse_loss = jnp.mean((N_res_pred - N_res) ** 2)
	w1_reg_loss = jnp.linalg.norm(W1) ** 2
	w2_reg_loss = jnp.linalg.norm(W2) ** 2

	res_loss = mse_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda
	# =========================================

	return man_loss * loss1_lambda + res_loss * loss2_lambda

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5, 6))


def update(opt_update, opt_state, params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambdas):
	grads = loss_fn_grad(params, N, B_ll, B_hl, cca_lambdas, loss_lambdas, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7, 8))
