import jax
import jax.numpy as jnp
import optax

from sklearn.metrics import r2_score


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


#region EVALUATING

@jax.jit
def predict_res(B_ll, N_man, W1, W2):
	# B_ll_mean, B_ll_std = prep['mean'], prep['std']

	N_man -= N_man.mean(axis=0)
	N_man /= jnp.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)

	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)

@jax.jit
def predict_man(prep, N, U):
	N_mean, N_std = prep['mean'], prep['std']

	N_zs = (N - N_mean) / N_std
	N_man = N_zs @ U
	N_zs_rec = N_man @ jnp.linalg.inv(N_man.T @ N_man) @ N_man.T @ N_zs  # least-squares reconstruction (NOT N_man @ jax.numpy.linalg.pinv(U))
	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	return N_man, N_rec, N_res

@jax.jit
def predict(params, N, B_ll):
	prep = params['prep']
	U, V, W1, W2 = params['model']['U'], params['model']['V'], params['model']['W1'], params['model']['W2']

	N_man, N_rec, N_res = predict_man(prep['N'], N, U)

	N_res_pred = predict_res(B_ll, N_man, W1, W2)

	return N_man, N_rec, N_res, N_res_pred

@jax.jit
def r2_score(y, y_pred):
	ss_res = jnp.sum((y - y_pred) ** 2)  # ^p?
	ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)  # ^p?
	return jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 0.0)

@jax.jit
def score_fn(params, N, B_ll):
	N_man, N_rec, N_res, N_res_pred = predict(params, N, B_ll)
	score_per_neuron = jax.vmap(r2_score, in_axes=(1, 1))(N_res, N_res_pred)
	return score_per_neuron, N_man, N_rec, N_res, N_res_pred

#endregion


#region TRAINING

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

def loss_fn(params, N, B_ll, B_hl, lambdas):
	# read hyperparams
	loss_lambdas, reg_lambdas, cca_lambdas = lambdas.loss_lambdas, lambdas.reg_lambdas, lambdas.cca_lambdas

	loss1_lambda, loss2_lambda = loss_lambdas
	w1_reg_lambda, w2_reg_lambda = reg_lambdas
	ccadiag_lambda, ccaunit_lambda = cca_lambdas

	# read params
	prep = jax.lax.stop_gradient(params['prep'])
	U, V, W1, W2 = params['model']['U'], params['model']['V'], params['model']['W1'], params['model']['W2']

	# manifold ================================
	N_man, N_rec, N_res = predict_man(prep['N'], N, U)

	# B_hl_mean, B_hl_std = prep['B_hl']['mean'], prep['B_hl']['std']
	# B_hl_zs = (B_hl - B_hl_mean) / B_hl_std
	# B_hl_man = B_hl_zs @ V
	#
	# man_loss = cca_loss(N_man, B_hl_man, ccadiag_lambda, ccaunit_lambda)
	man_loss = jnp.array(0)
	# =========================================

	# residuals ===============================
	N_res_pred = predict_res(B_ll, N_man, W1, W2)
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	w1_reg_loss, w2_reg_loss = jnp.linalg.norm(W1) ** 2, jnp.linalg.norm(W2) ** 2

	res_loss = mpe_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda
	# =========================================

	full_loss = man_loss * loss1_lambda + res_loss * loss2_lambda
	return full_loss, man_loss, res_loss, mpe_loss

loss_fn_grad = jax.jit(jax.grad(lambda *args: loss_fn(*args)[0]), static_argnums=(4,))

def update(opt_update, opt_state, params, N, B_ll, B_hl, lambdas):
	grads = loss_fn_grad(params, N, B_ll, B_hl, lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state

update = jax.jit(update, static_argnums=(0, 6,))

#endregion
