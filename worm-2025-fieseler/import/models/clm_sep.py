import jax
import jax.numpy as jnp
import optax

from sklearn.metrics import r2_score


p = 1


def get_type():
	return False


def init_params(seeds, shape_bn):
	return {
		'W': 1E-2 * jax.random.normal(seeds[2], shape_bn),
	}


# EVALUATING

@jax.jit
def predict(B_bll, W):
	return B_bll @ W

def mpe_loss_fn(params, B_ll, N_res):
	W = params['W']
	N_res_pred = predict(B_ll, W)
	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	return mpe_loss

def score_fn(params, B_ll, N_res):
	W = params['W']
	N_res_pred = predict(B_ll, W)
	score_per_neuron = [r2_score(trace, trace_pred) for trace, trace_pred in zip(N_res.T, N_res_pred.T)]
	return score_per_neuron


# TRAINING

def loss_fn(params, B_ll, N_res, reg_lambdas):
	w_reg_lambda = reg_lambdas[0]
	W = params['W']

	N_res_pred = predict(B_ll, W)

	mpe_loss = jnp.mean(jnp.mean(jnp.abs(N_res_pred - N_res) ** p, axis=0))
	w_reg_loss = jnp.linalg.norm(W) ** 2

	res_loss = mpe_loss + w_reg_loss * w_reg_lambda

	return res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(3,))


def update(opt_update, opt_state, params, B_ll, N_res, reg_lambdas):
	grads = loss_fn_grad(params, B_ll, N_res, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 5))
