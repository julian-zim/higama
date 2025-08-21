import jax
import jax.numpy as jnp
import optax

from sklearn.metrics import r2_score


def get_type():
	return True


def init_params(seeds, shape_U, shape_V, shape_bn, shape_dn):
	return {
	    'U': 1E-2 * jax.random.normal(seeds[0], shape_U),
	    'V': 1E-2 * jax.random.normal(seeds[1], shape_V),
	    'W1': 1E-2 * jax.random.normal(seeds[2], shape_bn),
	    'W2': 1E-2 * jax.random.normal(seeds[3], shape_dn)
	}


# EVALUATING

@jax.jit
def predict(B_ll, N_man, W1, W2):
	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)

def obtain(params, N, B_ll, prep_data):
	N_mean, N_std = jnp.asarray(prep_data.N_mean), jnp.asarray(prep_data.N_std)

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	N_zs = (N - N_mean) / N_std
	N_man = N_zs @ U
	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # got'em	N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	N_res_pred = predict(B_ll, N_man, W1, W2)

	return N_res_pred, N_res, N_rec

obtain = jax.jit(obtain, static_argnums=(3,))

def mse_loss_fn(params, N, B_ll, prep_data):
	N_res_pred, N_res, _ = obtain(params, N, B_ll, prep_data)
	mse_loss = jnp.mean((N_res_pred - N_res) ** 2)
	return mse_loss

mse_loss_fn = jax.jit(mse_loss_fn, static_argnums=(3,))

def score_fn(params, N, B_ll, prep_data):
	N_res_pred, N_res, _ = obtain(params, N, B_ll, prep_data)
	score_per_neuron = [r2_score(trace, trace_pred) for trace, trace_pred in zip(N_res.T, N_res_pred.T)]
	return score_per_neuron


# TRAINING

@jax.jit
def norming(X_proj):
	X_proj = X_proj - jnp.mean(X_proj, axis=0)
	X_proj = X_proj / (jnp.linalg.norm(X_proj, axis=0) + 1e-6)
	return X_proj

@jax.jit
def cca_loss(X_proj, Y_proj, ccaunit_lambda, ccadiag_lambda):
	corr = norming(X_proj).T @ norming(Y_proj)

	corr_diag = jnp.diag(jnp.diag(corr))
	corr_nondiag = corr - corr_diag

	return (ccaunit_lambda * jnp.mean(jnp.abs(corr_diag - jnp.eye(*corr_diag.shape)))
			+ ccadiag_lambda * jnp.mean(jnp.abs(corr_nondiag)))

def loss_fn(params, N, B_ll, B_hl, lambdas, prep_data):
	loss_lambdas, reg_lambdas, cca_lambdas = lambdas.loss_lambdas, lambdas.reg_lambdas, lambdas.cca_lambdas

	loss1_lambda, loss2_lambda = loss_lambdas
	w1_reg_lambda, w2_reg_lambda = reg_lambdas
	ccaunit_lambda, ccadiag_lambda = cca_lambdas

	N_mean, N_std, B_hl_mean, B_hl_std = (jnp.asarray(prep_data.N_mean),
										  jnp.asarray(prep_data.N_std),
										  jnp.asarray(prep_data.B_hl_mean),
										  jnp.asarray(prep_data.B_hl_std))

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	# manifold ================================
	N_zs = (N - N_mean) / N_std
	B_hl_zs = (B_hl - B_hl_mean) / B_hl_std

	N_man = N_zs @ U
	B_hl_man = B_hl_zs @ V

	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # got'em	N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	man_loss = cca_loss(N_man, B_hl_man, ccaunit_lambda, ccadiag_lambda)
	# =========================================

	# residuals ===============================
	N_res_pred = predict(B_ll, N_man, W1, W2)
	mse_loss = jnp.mean((N_res_pred - N_res) ** 2)
	w1_reg_loss, w2_reg_loss = jnp.linalg.norm(W1) ** 2, jnp.linalg.norm(W2) ** 2

	res_loss = mse_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda
	# =========================================

	return man_loss * loss1_lambda + res_loss * loss2_lambda

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5))

def update(opt_update, opt_state, params, N, B_ll, B_hl, lambdas, prep_data):
	grads = loss_fn_grad(params, N, B_ll, B_hl, lambdas, prep_data)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7))






# temp
def obtain2(params, N, B_ll, prep_data):
	N_mean, N_std = jnp.asarray(prep_data.N_mean), jnp.asarray(prep_data.N_std)

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	N_zs = (N - N_mean) / N_std
	N_man = N_zs @ U
	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # got'em	N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	N_res_pred = predict(B_ll, N_man, W1, W2)

	return N_res_pred, N_res, N_rec, (N_man, B_ll @ W1, jax.nn.sigmoid(N_man @ W2))