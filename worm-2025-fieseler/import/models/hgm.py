import jax
import jax.numpy as jnp
import optax


def get_type():
	return True


def init_params(seeds, shape_U, shape_V, shape_N, shape_N_man):
	return {
	    'W1': 1E-2 * jax.random.normal(seeds[0], shape_N),
	    'W2': 1E-2 * jax.random.normal(seeds[1], shape_N_man),
	    'U': 1E-2 * jax.random.normal(seeds[2], shape_U),
	    'V': 1E-2 * jax.random.normal(seeds[3], shape_V)
	}


@jax.jit
def predict(B_ll, N_man, W1, W2):
	return B_ll @ W1 * jax.nn.sigmoid(N_man @ W2)


# temp
def mse_loss_fn(params, N, B_ll):
	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	N_man = N @ U
	N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_res = N - N_rec
	Y_pred = predict(B_ll, N_man, W1, W2)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)

	return mse_loss


@jax.jit
def cca_loss(X_proj, Y_proj):
	eps = 1e-8

	X_centered = X_proj - jnp.mean(X_proj, axis=0, keepdims=True)
	Y_centered = Y_proj - jnp.mean(Y_proj, axis=0, keepdims=True)

	cov = jnp.mean(X_centered * Y_centered, axis=0)
	std_X = jnp.std(X_centered, axis=0) + eps
	std_Y = jnp.std(Y_centered, axis=0) + eps

	corrs = cov / (std_X * std_Y)

	return jnp.mean(1 - corrs)

def loss_fn(params, N, B_ll, B_hl, loss_lambdas, reg_lambdas):
	l1, l2 = loss_lambdas
	reg_l1, reg_l2 = reg_lambdas[0], reg_lambdas[1]

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	# manifold ================================
	N_man = N @ U
	B_hl_man = B_hl @ V

	N_rec = N_man @ jax.numpy.linalg.pinv(U)
	N_res = N - N_rec

	man_loss = cca_loss(N_man, B_hl_man)
	# =========================================

	# residuals ===============================
	Y_pred = predict(B_ll, N_man, W1, W2)
	mse_loss = jnp.mean((Y_pred - N_res) ** 2)
	w1_loss = reg_l1 * jnp.linalg.norm(W1) ** 2
	w2_loss = reg_l2 * jnp.linalg.norm(W2) ** 2

	res_loss = mse_loss + w1_loss + w2_loss
	# =========================================

	return l1 * man_loss + l2 * res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5))


def update(opt_update, opt_state, params, N, B_ll, B_hl, loss_weights, reg_lambdas):
	grads = loss_fn_grad(params, N, B_ll, B_hl, loss_weights, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7))
