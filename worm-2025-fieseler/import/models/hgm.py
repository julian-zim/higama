import jax
import jax.numpy as jnp
import optax


p = 1


def get_type():
	return True

def init_params(N, B_hl, seeds, shape_U, shape_V, shape_bn, shape_dn):
	return {
		'prep': {
			'N': {
				'mean': N.mean(axis=0),
				'std': jnp.where((N_std := jnp.std(N, axis=0)) == 0.0, 1.0, N_std),
			},
			'B_hl': {
				'mean': B_hl.mean(axis=0),
				'std': jnp.where((B_hl_std := jnp.std(B_hl, axis=0)) == 0.0, 1.0, B_hl_std),
			},
		},
		'model': {
			'U': 1E-2 * jax.random.normal(seeds[0], shape_U),
			'V': 1E-2 * jax.random.normal(seeds[1], shape_V),
			'W1': 1E-2 * jax.random.normal(seeds[2], shape_bn),
			'W2': 1E-2 * jax.random.normal(seeds[3], shape_dn)
		}
	}


#region PREDICTING

@jax.jit
def predict_res(B_ll, N_man, W1, W2):
	N_man -= N_man.mean(axis=0)
	N_man /= jnp.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)

	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)

@jax.jit
def predict_man(prep_n, N, U):
	N_mean, N_std = prep_n['mean'], prep_n['std']

	N_zs = (N - N_mean) / N_std
	N_man = N_zs @ U
	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # least-squares reconstruction (NOT N_man @ jax.numpy.linalg.pinv(U))
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	return N_man, N_rec, N_res

@jax.jit
def predict(params, N, B_ll):
	prep_n = jax.lax.stop_gradient(params['prep']['N'])
	U, W1, W2 = params['model']['U'], params['model']['W1'], params['model']['W2']

	N_man, N_rec, N_res = predict_man(prep_n, N, U)

	N_res_pred = predict_res(B_ll, N_man, W1, W2)

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
def loss_fn(params, N, B_ll, B_hl, lambdas):
	# read hyperparams
	loss_lambdas, reg_lambdas, cca_lambdas = lambdas.loss_lambdas, lambdas.reg_lambdas, lambdas.cca_lambdas

	loss1_lambda, loss2_lambda = loss_lambdas
	w1_reg_lambda, w2_reg_lambda = reg_lambdas
	ccadiag_lambda, ccaunit_lambda = cca_lambdas

	# predict
	N_man, N_rec, N_res, N_res_pred = predict(params, N, B_ll)

	# manifold loss =================================
	prep_b_hl = jax.lax.stop_gradient(params['prep']['B_hl'])
	B_hl_mean, B_hl_std = prep_b_hl['mean'], prep_b_hl['std']
	V = params['model']['V']

	B_hl_zs = (B_hl - B_hl_mean) / B_hl_std
	B_hl_man = B_hl_zs @ V
	man_loss = cca_loss(N_man, B_hl_man, ccadiag_lambda, ccaunit_lambda)
	# man_loss = jnp.array(0)
	# ===============================================

	# residuals loss ================================
	W1, W2 = params['model']['W1'], params['model']['W2']

	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	w1_reg_loss, w2_reg_loss = jnp.linalg.norm(W1) ** 2, jnp.linalg.norm(W2) ** 2
	res_loss = mpe_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda
	# ===============================================

	full_loss = man_loss * loss1_lambda + res_loss * loss2_lambda
	return full_loss, (man_loss, res_loss, mpe_loss)

loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=(4,))

@jax.jit
def loss_fn_flattened(params, N, B_ll, B_hl, lambdas):
	full_loss, (man_loss, res_loss, mpe_loss) = loss_fn(params, N, B_ll, B_hl, lambdas)
	return full_loss, man_loss, res_loss, mpe_loss

def update(opt_update, opt_state, params, N, B_ll, B_hl, lambdas):
	(loss, (man_loss, res_loss, mpe_loss)), grads = loss_and_grads(params, N, B_ll, B_hl, lambdas)
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
def score_fn(params, N, B_ll):
	N_man, N_rec, N_res, N_res_pred = predict(params, N, B_ll)
	_N_res_train_means = jnp.zeros((N_res.shape[1],))  # TDL
	score_per_neuron = jax.vmap(r2_score, in_axes=(1, 1, 0))(N_res, N_res_pred, _N_res_train_means)
	return score_per_neuron, N_man, N_rec, N_res, N_res_pred

#endregion


#region OLD

from sklearn.metrics import r2_score as r2_score_sklearn

@jax.jit
def predict_OLD(B_ll, N_man, W1, W2):
	N_man -= N_man.mean(axis=0)
	N_man /= jnp.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)

	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)

@jax.jit
def obtain_OLD(params, N, B_ll, prep_data):
	N_mean, N_std = jnp.asarray(prep_data.N_mean), jnp.asarray(prep_data.N_std)

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']
	# U, V, W1, W2 = params['model']['U'], params['model']['V'], params['model']['W1'], params['model']['W2']

	N_zs = (N - N_mean) / N_std
	N_man = N_zs @ U
	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # got'em	N_man @ jax.numpy.linalg.pinv(U)
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	N_res_pred = predict_OLD(B_ll, N_man, W1, W2)

	return N_res_pred, N_res, N_rec

def score_fn_OLD(params, N, B_ll, prep_data):
	N_res_pred, N_res, _ = obtain_OLD(params, N, B_ll, prep_data)
	score_per_neuron = [(r2_score_sklearn(trace, trace_pred) if trace.var() > 0 else 0) for trace, trace_pred in zip(N_res.T, N_res_pred.T)]
	return score_per_neuron

#endregion
