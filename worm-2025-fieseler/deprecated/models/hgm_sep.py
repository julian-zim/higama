import jax
import jax.numpy as jnp
import optax

from sklearn.metrics import r2_score


p = 1


def get_type():
	return True


def init_params(seeds, shape_bn, shape_dn):
	return {
		'W1': 1E-2 * jax.random.normal(seeds[2], shape_bn),
		'W2': 1E-2 * jax.random.normal(seeds[3], shape_dn)
	}


# EVALUATING

@jax.jit
def predict(B_ll, N_man, W1, W2):
	N_man -= N_man.mean(axis=0)
	N_man /= jnp.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)

	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)

def mpe_loss_fn(params, B_ll, N_man, N_res):
	W1, W2 = params['W1'], params['W2']
	N_res_pred = predict(B_ll, N_man, W1, W2)
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	return mpe_loss

def score_fn(params, B_ll, N_man, N_res):
	W1, W2 = params['W1'], params['W2']
	N_res_pred = predict(B_ll, N_man, W1, W2)
	score_per_neuron = [(r2_score(trace, trace_pred) if trace.var() > 0 and trace_pred.var() > 0 else 0) for trace, trace_pred in zip(N_res.T, N_res_pred.T)]

	return score_per_neuron


# TRAINING

def loss_fn(params, B_ll, N_man, N_res, reg_lambdas):
	w1_reg_lambda, w2_reg_lambda = reg_lambdas[0], reg_lambdas[1]
	W1, W2 = params['W1'], params['W2']

	N_res_pred = predict(B_ll, N_man, W1, W2)

	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	w1_reg_loss = jnp.linalg.norm(W1) ** 2
	w2_reg_loss = jnp.linalg.norm(W2) ** 2

	res_loss = mpe_loss + w1_reg_loss * w1_reg_lambda + w2_reg_loss * w2_reg_lambda

	return res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4,))


def update(opt_update, opt_state, params, B_ll, N_man, N_res, reg_lambdas):
	grads = loss_fn_grad(params, B_ll, N_man, N_res, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state

update = jax.jit(update, static_argnums=(0, 6))
