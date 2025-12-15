import jax
import jax.numpy as jnp
import optax


p = 1


def get_type():
	return False

def init_params(seeds, shape_bn):
	return {
		'model': {
			'W': 1E-2 * jax.random.normal(seeds[2], shape_bn)
		}
	}


#region TRAINING

@jax.jit
def predict(W, B_ll):
	return B_ll @ W

@jax.jit
def loss_fn(params, N_res, B_ll, lambdas):
	# read hyperparams
	w1_reg_lambda = lambdas.reg_lambdas[0]

	# read params
	W = params['model']['W']

	# predict
	N_res_pred = predict(W, B_ll)

	# loss
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	w1_reg_loss = jnp.linalg.norm(W) ** 2
	loss = mpe_loss + w1_reg_loss * w1_reg_lambda

	return loss, mpe_loss

loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=(3,))

def update(opt_update, opt_state, params, N_res, B_ll, lambdas):
	(loss, mpe_loss), grads = loss_and_grads(params, N_res, B_ll, lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return opt_state, params_new, (loss, mpe_loss)

update = jax.jit(update, static_argnums=(0, 5,))

#endregion


#region EVALUATING

#@jax.jit
def r2_score(y_eval, y_pred, baseline):
	ss_res = jnp.sum((y_eval - y_pred) ** 2)  # ^p?
	ss_tot = jnp.sum((y_eval - baseline) ** 2)  # ^p?
	return jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 0.0)

@jax.jit
def score_fn(params, N_res, B_ll):
	W = params['model']['W']

	N_res_pred = predict(W, B_ll)

	_N_res_train_means = jnp.zeros((N_res.shape[1],))  # TDL
	score_per_neuron = jax.vmap(r2_score, in_axes=(1, 1, 0))(N_res, N_res_pred, _N_res_train_means)
	return score_per_neuron, N_res_pred

#endregion
