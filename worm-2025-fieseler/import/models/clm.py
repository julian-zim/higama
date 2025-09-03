import jax
import jax.numpy as jnp
import optax

from sklearn.metrics import r2_score


def get_type():
	return False


def init_params(seeds, shape_U, shape_V, shape_bn):
	return {
	    'U': 1E-2 * jax.random.normal(seeds[0], shape_U),
	    'V': 1E-2 * jax.random.normal(seeds[1], shape_V),
	    'W': 1E-2 * jax.random.normal(seeds[2], shape_bn)
	}


# EVALUATING

@jax.jit
def predict(B_ll, W):
	return B_ll @ W

def obtain(params, N, B_ll, prep_data):
	N_mean, N_std = jnp.asarray(prep_data.N_mean), jnp.asarray(prep_data.N_std)

	U, V, W = params['U'], params['V'], params['W']

	N_zs = (N - N_mean) / N_std
	N_man = N_zs @ U
	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # got'em	N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	N_res_pred = predict(B_ll, W)

	return N_res_pred, N_res, N_rec

obtain = jax.jit(obtain, static_argnums=(3,))

def mpe_loss_fn(params, N, B_ll, prep_data):
	N_res_pred, N_res, _ = obtain(params, N, B_ll, prep_data)
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** 1, axis=0))
	return mpe_loss

mpe_loss_fn = jax.jit(mpe_loss_fn, static_argnums=(3,))

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
def cca_loss(X_proj, Y_proj, ccadiag_lambda, ccaunit_lambda):
	corr = norming(X_proj).T @ norming(Y_proj)

	corr_diag = jnp.diag(jnp.diag(corr))
	corr_nondiag = corr - corr_diag

	return (ccadiag_lambda * jnp.mean(jnp.abs(corr_nondiag))
			+ ccaunit_lambda * jnp.mean(jnp.abs(corr_diag - jnp.eye(*corr_diag.shape))))

def loss_fn(params, N, B_ll, B_hl, lambdas, prep_data):
	loss_lambdas, reg_lambdas, cca_lambdas = lambdas.loss_lambdas, lambdas.reg_lambdas, lambdas.cca_lambdas

	loss1_lambda, loss2_lambda = loss_lambdas
	w_reg_lambda = reg_lambdas[0]
	ccadiag_lambda, ccaunit_lambda = cca_lambdas

	N_mean, N_std, B_hl_mean, B_hl_std = (jnp.asarray(prep_data.N_mean),
										  jnp.asarray(prep_data.N_std),
										  jnp.asarray(prep_data.B_hl_mean),
										  jnp.asarray(prep_data.B_hl_std))

	U, V, W = params['U'], params['V'], params['W']

	# manifold ================================
	N_zs = (N - N_mean) / N_std
	B_hl_zs = (B_hl - B_hl_mean) / B_hl_std

	N_man = N_zs @ U
	B_hl_man = B_hl_zs @ V

	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # got'em	N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	man_loss = cca_loss(N_man, B_hl_man, ccadiag_lambda, ccaunit_lambda)
	# =========================================

	# residuals ===============================
	N_res_pred = predict(B_ll, W)
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** 1, axis=0))
	w_reg_loss = jnp.linalg.norm(W) ** 2

	res_loss = mpe_loss + w_reg_loss * w_reg_lambda
	# =========================================

	return man_loss * loss1_lambda + res_loss * loss2_lambda

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5))


def update(opt_update, opt_state, params, N, B_ll, B_hl, lambdas, prep_data):
	grads = loss_fn_grad(params, N, B_ll, B_hl, lambdas, prep_data)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7))
