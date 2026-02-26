import jax
import jax.numpy as jnp
import optax

p = 1


def get_type():
	return False


def init_params(N, B, S, seeds, n_dimensions, n_neurons, n_behaviors, n_states):
	shape_UN = (n_neurons, n_dimensions)
	shape_US = (n_states, n_dimensions)
	shape_WB = (n_behaviors, n_neurons)

	return {
		'prep': {
			'N': {
				'mean': N.mean(axis=0),
				'std': jnp.where((N_std := jnp.std(N, axis=0)) == 0.0, 1.0, N_std),
			},
			'B': {
				'mean': B.mean(axis=0),
				'std': jnp.where((B_std := jnp.std(B, axis=0)) == 0.0, 1.0, B_std),
			},
			'S': {
				'mean': S.mean(axis=0),
				'std': jnp.where((S_std := jnp.std(S, axis=0)) == 0.0, 1.0, S_std),
			},
		},
		'model': {
			'UN': 1E-3 * jax.random.normal(seeds[0], shape_UN),
			# 'US': 1E-3 * jax.random.normal(seeds[1], shape_US),
			'WB': 1E-3 * jax.random.normal(seeds[3], shape_WB),
		}
	}


def get_reconstruction_matrix(Y, X):
	# best reconstruction matrix that reconstruct X using Y (NOT Y @ jax.numpy.linalg.pinv(UN))
	# return jnp.linalg.inv(Y.T @ Y) @ Y.T @ X
	# return jnp.linalg.solve(Y.T @ Y + 1E-3 * jnp.eye(Y.shape[1]), Y.T @ X)
	# return jnp.linalg.lstsq(Y, X, rcond=None)[0]

	# YTY = Y.T @ Y
	# lam = 1E2 * jnp.finfo(YTY.dtype).eps * jnp.maximum(jnp.trace(YTY) / YTY.shape[0], 1.0)
	# return jnp.linalg.solve(YTY + lam * jnp.eye(Y.shape[1], dtype=Y.dtype), Y.T @ X)

	return jnp.linalg.lstsq(Y, X, rcond=None)[0]


#region PREDICTING

def orthonormalize_qr(WB):
	Q, R = jnp.linalg.qr(WB)
	signs = jnp.sign(jnp.diag(R))  # fix sign ambiguity for a deterministic mapping
	return Q * jnp.where(signs == 0, 1.0, signs)


@jax.jit
def predict_res(prep_b, B, WB):
	B_mean, B_std = prep_b['mean'], prep_b['std']
	B_zs = (B - B_mean) / B_std

	return B_zs @ WB


@jax.jit
def predict_man(prep_n, N, UN):
	N_mean, N_std = prep_n['mean'], prep_n['std']

	N_zs = (N - N_mean) / N_std
	# N_zs = N - N_mean
	# N_man = N_zs @ UN
	N_man = N_zs @ orthonormalize_qr(UN)
	# N_zs_rec = N_man @ orthonormalize_qr(UN).T
	N_zs_rec = N_man @ get_reconstruction_matrix(N_man, N_zs)
	N_rec = N_zs_rec * N_std + N_mean
	# N_rec = N_zs_rec + N_mean
	N_res = N - N_rec

	return N_man, N_rec, N_res


@jax.jit
def predict(params, N, B):
	prep_n = jax.lax.stop_gradient(params['prep']['N'])
	prep_b = jax.lax.stop_gradient(params['prep']['B'])
	UN, WB = params['model']['UN'], params['model']['WB']

	N_man, N_rec, N_res = predict_man(prep_n, N, UN)

	N_res_pred = predict_res(prep_b, B, WB)

	return N_man, N_rec, N_res, N_res_pred


#endregion


#region TRAINING

#@jax.jit
def norm(X_proj):
	X_proj = X_proj - jnp.mean(X_proj, axis=0)
	X_proj = X_proj / (jnp.linalg.norm(X_proj, axis=0) + 1e-6)
	return X_proj


@jax.jit
def cca_loss(X_proj, Y_proj, ccadiag_lambda, ccaunit_lambda):
	corr = norm(X_proj).T @ norm(Y_proj)

	corr_diag = jnp.diag(jnp.diag(corr))
	corr_nondiag = corr - corr_diag

	return (ccadiag_lambda * jnp.mean(jnp.abs(corr_nondiag))
			+ ccaunit_lambda * jnp.mean(jnp.abs(corr_diag - jnp.eye(*corr_diag.shape))))


#@jax.jit
def loss_fn(params, N, B, S, lambdas):
	# read hyperparams
	loss_lambdas, reg_lambdas, cca_lambdas = lambdas.loss_lambdas, lambdas.reg_lambdas, lambdas.cca_lambdas

	loss_man_lambda, loss_res_lambda = loss_lambdas
	wb_reg_lambda, = reg_lambdas
	ccadiag_lambda, ccaunit_lambda = cca_lambdas

	# predict
	N_man, N_rec, N_res, N_res_pred = predict(params, N, B)

	# manifold loss =================================
	# prep_s = jax.lax.stop_gradient(params['prep']['S'])
	# S_mean, S_std = prep_s['mean'], prep_s['std']
	# US = params['model']['US']
	#
	# S_zs = (S - S_mean) / S_std
	# S_man = S_zs @ US
	# man_loss = cca_loss(N_man, S_man, ccadiag_lambda, ccaunit_lambda)
	man_loss = jnp.array(0)
	# ===============================================

	# residuals loss ================================
	WB = params['model']['WB']

	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	wb_reg_loss = jnp.linalg.norm(WB) ** 2
	res_loss = (mpe_loss
				+ wb_reg_loss * wb_reg_lambda)
	# ===============================================

	full_loss = (man_loss * loss_man_lambda +
				 res_loss * loss_res_lambda)
	return full_loss, (man_loss, res_loss, mpe_loss)


loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=(4,))


@jax.jit
def loss_fn_flattened(params, N, B, S, lambdas):
	full_loss, (man_loss, res_loss, mpe_loss) = loss_fn(params, N, B, S, lambdas)
	return full_loss, man_loss, res_loss, mpe_loss


def update(opt_update, opt_state, params, N, B, S, lambdas):
	(loss, (man_loss, res_loss, mpe_loss)), grads = loss_and_grads(params, N, B, S, lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return opt_state, params_new, (loss, man_loss, res_loss, mpe_loss)


update = jax.jit(update, static_argnums=(0, 6,))


#endregion


#region EVALUATING

#@jax.jit
def r2_score(y_eval, y_pred, baseline):
	ss_res = jnp.sum((y_eval - y_pred) ** 2)  # ^p?
	ss_tot = jnp.sum((y_eval - baseline) ** 2)  # ^p?
	return jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 0.0)


@jax.jit
def score_fn(params, N, B):
	N_man, N_rec, N_res, N_res_pred = predict(params, N, B)
	_N_res_train_means = jnp.zeros((N_res.shape[1],))  # ToDo: Use actual train residual mean!
	score_per_neuron = jax.vmap(r2_score, in_axes=(1, 1, 0))(N_res, N_res_pred, _N_res_train_means)
	return score_per_neuron, N_man, N_rec, N_res, N_res_pred

#endregion
