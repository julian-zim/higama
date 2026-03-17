import jax
import jax.numpy as jnp


@jax.jit
def orthonormalize_qr(A):
	Q, R = jnp.linalg.qr(A, mode='reduced')
	signs = jnp.sign(jnp.diag(R))  # fix sign ambiguity for a deterministic mapping
	return Q * jnp.where(signs == 0, 1.0, signs)


@jax.jit
def get_reconstruction_matrix(X, Y):
	# reconstruct Y (target) using X (predictor)

	# return jnp.linalg.inv(X.T @ X) @ X.T @ Y
	#
	# XTX = X.T @ X
	# lam = 1E2 * jnp.finfo(XTX.dtype).eps * jnp.maximum(jnp.trace(XTX) / XTX.shape[0], 1.0)
	# return jnp.linalg.solve(XTX + lam * jnp.eye(X.shape[1], dtype=X.dtype), X.T @ Y)

	# return U.T
	# return jnp.linalg.pinv(U)
	return jnp.linalg.lstsq(X, Y)[0]

#endregion


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

#endregion


#region EVALUATING

#@jax.jit
def r2_score(y_eval, y_pred, baseline):
	ss_res = jnp.sum((y_eval - y_pred) ** 2)  # ^p?
	ss_tot = jnp.sum((y_eval - baseline) ** 2)  # ^p?
	return jnp.where(ss_tot > 0.0, 1.0 - ss_res / ss_tot, 0.0)

#endregion
