import jax
import jax.numpy as jnp
import optax


def get_type():
	return True


def init_params(seed, shape_U, shape_V, shape_X, shape_X_man):
	return {
	    'U': 1E-2 * jax.random.normal(seed, shape_U),
	    'V': 1E-2 * jax.random.normal(seed, shape_V),
	    'W1': 1E-2 * jax.random.normal(seed, shape_X),
	    'W2': 1E-2 * jax.random.normal(seed, shape_X_man)
	}


def predict(B_ll, X_man, W1, W2):
	return B_ll @ W1 * jax.nn.sigmoid(X_man @ W2)


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

def loss_fn(params, X, B_ll, B_hl, loss_lambdas, reg_lambdas):
	l1, l2 = loss_lambdas
	reg_l1, reg_l2 = reg_lambdas[0], reg_lambdas[1]

	U, V, W1, W2 = params['U'], params['V'], params['W1'], params['W2']

	# manifold ================================
	X_man = X @ U
	X_man_bhl = B_hl @ V

	X_rec = X_man @ jax.numpy.linalg.pinv(U)
	X_res = X - X_rec

	man_loss = cca_loss(X_man, X_man_bhl)
	# =========================================

	# residuals ===============================
	Y_pred = predict(B_ll, X_man, W1, W2)
	mse_loss = jnp.mean((Y_pred - X_res) ** 2)
	w1_loss = reg_l1 * jnp.linalg.norm(W1) ** 2
	w2_loss = reg_l2 * jnp.linalg.norm(W2) ** 2

	res_loss = mse_loss + w1_loss + w2_loss
	# =========================================

	return l1 * man_loss + l2 * res_loss

loss_fn_grad = jax.jit(jax.grad(loss_fn), static_argnums=(4, 5))


def update(opt_update, opt_state, params, X, B_ll, B_hl, loss_weights, reg_lambdas):
	grads = loss_fn_grad(params, X, B_ll, B_hl, loss_weights, reg_lambdas)
	params_updates, opt_state = opt_update(grads, opt_state, params)
	params_new = optax.apply_updates(params, params_updates)
	return params_new, opt_state, grads  # return grads for weight interpretation

update = jax.jit(update, static_argnums=(0, 6, 7))
