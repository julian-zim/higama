#region imports
import sys, os
import jax
import jax.numpy as jnp
import optax
sys.path.append(os.path.dirname(__file__))
import model_utils
#endregion

p = 1


def init_params(N, B, S, n_dimensions, seeds):
	n_bss = min(B.shape[1], 16)

	shape_UN = (N.shape[1], n_dimensions)
	shape_US = (S.shape[1], n_dimensions)
	shape_VB = (n_bss, N.shape[1])
	shape_WB = (B.shape[1], n_bss)
	shape_WM = (n_dimensions, n_bss)

	return {
		'prep': {
			'N': {
				'mean': N.mean(axis=0),
				# 'std': jnp.where((N_std := jnp.std(N, axis=0)) == 0.0, 1.0, N_std),
				'std': jnp.ones((N.shape[1],)),
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
			'VB': 1E-3 * jax.random.normal(seeds[2], shape_VB),
			'WB': 1E-3 * jax.random.normal(seeds[3], shape_WB),
			'WM': 1E-3 * jax.random.normal(seeds[4], shape_WM),
		}
	}


#region PREDICTING

#@jax.jit
def predict_res(prep_b, B, N_man, WB, VB, WM):
	B_mean, B_std = prep_b['mean'], prep_b['std']
	B_zs = (B - B_mean) / B_std

	N_man_zs = (N_man - N_man.mean(axis=0)) / jnp.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)

	B_ss = B_zs @ model_utils.orthonormalize_qr(WB) * jax.nn.sigmoid(N_man_zs @ WM)
	return B_ss @ VB, B_ss


#@jax.jit
def predict_man(prep_n, N, UN):
	N_mean, N_std = prep_n['mean'], prep_n['std']

	N_zs = (N - N_mean) / N_std

	# UN_orth = UN
	UN_orth = model_utils.orthonormalize_qr(UN)

	N_man = N_zs @ UN_orth
	# N_zs_rec = N_man @ model_utils.get_reconstruction_matrix(N_man, N_zs)
	N_zs_rec = N_man @ UN_orth.T

	N_rec = N_zs_rec * N_std + N_mean
	N_res = N - N_rec

	return N_man, N_rec, N_res


@jax.jit
def predict(params, N, B):
	prep_n = jax.lax.stop_gradient(params['prep']['N'])
	prep_b = jax.lax.stop_gradient(params['prep']['B'])
	UN, VB, WB, WM = params['model']['UN'], params['model']['VB'], params['model']['WB'], params['model']['WM']

	N_man, N_rec, N_res = predict_man(prep_n, N, UN)

	N_res_pred, B_ss = predict_res(prep_b, B, N_man, WB, VB, WM)

	return N_man, N_rec, N_res, N_res_pred, B_ss

#endregion


#region TRAINING

#@jax.jit
def loss_fn(params, N, B, S, lambdas):
	# read hyperparams
	loss_lambdas, reg_lambdas, cca_lambdas = lambdas.loss_lambdas, lambdas.reg_lambdas, lambdas.cca_lambdas

	loss_man_lambda, loss_res_lambda = loss_lambdas
	vb_reg_lambda, wm_reg_lambda = reg_lambdas
	ccadiag_lambda, ccaunit_lambda = cca_lambdas

	# predict
	N_man, N_rec, N_res, N_res_pred, B_ss = predict(params, N, B)

	# # manifold loss =================================
	# prep_s = jax.lax.stop_gradient(params['prep']['S'])
	# S_mean, S_std = prep_s['mean'], prep_s['std']
	# US = params['model']['US']
	#
	# S_zs = (S - S_mean) / S_std
	# S_man = S_zs @ US
	# man_loss = cca_loss(N_man, S_man, ccadiag_lambda, ccaunit_lambda)
	man_loss = jnp.array(0., dtype=N.dtype)
	# ===============================================

	# residuals loss ================================
	VB, WB, WM = params['model']['VB'], params['model']['WB'], params['model']['WM']

	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	vb_reg_loss, wb_reg_loss, wm_reg_loss = jnp.linalg.norm(VB) ** 2, jnp.linalg.norm(WB) ** 2, jnp.linalg.norm(WM) ** 2
	res_loss = (mpe_loss
				+ vb_reg_loss * vb_reg_lambda
				+ wm_reg_loss * wm_reg_lambda)
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

@jax.jit
def predict_and_score(params, N, B):
	N_man, N_rec, N_res, N_res_pred, B_ss = predict(params, N, B)
	N_res_ss = N_res @ model_utils.get_reconstruction_matrix(N_res_pred, B_ss)

	_N_res_train_means = jnp.zeros((N_res.shape[1],))  # ToDo: Use actual train residual mean!
	score_per_neuron = jax.vmap(model_utils.r2_score, in_axes=(1, 1, 0))(N_res, N_res_pred, _N_res_train_means)

	return score_per_neuron, N_man, N_rec, N_res, N_res_pred, N_res_ss, B_ss

#endregion
