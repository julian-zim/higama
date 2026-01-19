import jax
import jax.numpy as jnp
import optax


p = 1


def get_type():
	return True

def init_params(seeds, shape_bn, shape_dn):
	return {
		'model': {
			'W1': 1E-2 * jax.random.normal(seeds[2], shape_bn),
			'W2': 1E-2 * jax.random.normal(seeds[3], shape_dn)
		}
	}


#region TRAINING

@jax.jit
def predict(W1, W2, B_ll, N_man):
	N_man -= N_man.mean(axis=0)
	N_man /= jnp.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)

	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)

@jax.jit
def loss_fn(params, N_res, B_ll, N_man, lambdas):
	# read hyperparams
	w1_reg_lambda, w2_reg_lambda = lambdas.reg_lambdas

	# read params
	W1, W2 = params['model']['W1'], params['model']['W2']

	# predict
	N_res_pred = predict(W1, W2, B_ll, N_man)

	# loss
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	w1_reg_loss, w2_reg_loss = jnp.linalg.norm(W1) ** 2, jnp.linalg.norm(W2) ** 2
	loss = mpe_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda

	return loss, mpe_loss

loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=(4,))

def update(opt_update, opt_state, params, N_res, B_ll, N_man, lambdas):
	(loss, mpe_loss), grads = loss_and_grads(params, N_res, B_ll, N_man, lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return opt_state, params_new, (loss, mpe_loss)

update = jax.jit(update, static_argnums=(0, 6,))

#endregion


#region EVALUATING

#@jax.jit
def r2_score(y_eval, y_pred, baseline):
	ss_res = jnp.sum((y_eval - y_pred) ** 2)  # ^p?
	ss_tot = jnp.sum((y_eval - baseline) ** 2)  # ^p?
	return jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 0.0)

@jax.jit
def score_fn(params, N_res, B_ll, N_man):
	W1, W2 = params['model']['W1'], params['model']['W2']

	N_res_pred = predict(W1, W2, B_ll, N_man)

	_N_res_train_means = jnp.zeros((N_res.shape[1],))  # TDL
	score_per_neuron = jax.vmap(r2_score, in_axes=(1, 1, 0))(N_res, N_res_pred, _N_res_train_means)
	return score_per_neuron, N_res_pred

#endregion
