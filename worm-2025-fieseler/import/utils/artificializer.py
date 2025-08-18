import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp


def generate_params(model, seeds, shape_X_bll, shape_X_man):
	if model.get_type():
		# return {
		# 	'W1': 0.25 * jnp.tile(jax.random.normal(seeds[0], shape_X_bll[1]), (shape_X_bll[0], 1)),
		# 	'W2': 6 * jnp.tile(jnp.abs(jax.random.normal(seeds[1], shape_X_bll[1])), (shape_X_man[0], 1))
		# }
		return {
			'W1': 0.2 * jnp.tile(jax.random.normal(seeds[0], (shape_X_bll[1],)), (shape_X_bll[0], 1)),
			'W2': -5 * jnp.tile(jnp.abs(jax.random.normal(seeds[1], (shape_X_bll[1],))), (shape_X_man[0], 1))
		}
	else:
		return {
			'W': 0.4 * jnp.tile(jax.random.normal(seeds[0], (shape_X_bll[1],)), (shape_X_bll[0], 1))
		}

def generate_cca_params(n_dimensions, n_neurons, scaling=0.25):
	# (dimensions x neurons)
	# cca_weights = np.tile(np.array([math.e ** (1-d) for d in range(0, n_dimensions)]).reshape(-1, 1), (1, n_neurons))
	# cca_weights @= np.diag(np.array([(i+1) * scaling + (1. - scaling) for i in range(n_neurons)]))

	cca_weights = 0.1 * np.abs(np.random.randn(n_dimensions, n_neurons))
	cca_weights @= np.diag(np.array([(i+1) * scaling + (1. - scaling) for i in range(n_neurons)])) + 0.1 * np.abs(np.random.randn(n_neurons, n_neurons))

	return cca_weights

def generate_manifold(discrete_states, n_dimensions):
	weight = 1
	manifold = np.array((discrete_states.values[:, :n_dimensions] * 2 - 1) * weight, dtype=float)
	return manifold

def generate_reconstructed_traces(N_man, n_neurons):
	cca_weights = generate_cca_params(N_man.values.shape[1], n_neurons)
	return N_man.values @ cca_weights, cca_weights

def generate_residual_traces(model, X_bll, X_man, W1, W2):
	if model.get_type():
		traces = model.predict(X_bll.values, X_man.values, W1, W2)
	else:
		traces = model.predict(X_bll.values, W1)
	return traces

def generate_traces(N_rec, N_res):
	return N_rec.values + N_res.values

def artificialize(model, filename, neurons, eigenworms, discrete_states, n_dimensions=2):
	assert n_dimensions <= 4

	cyber_name = '_'.join(['ZIMMERMANN_artificial'] + filename.split('_')[2:])
	seeds = jax.random.PRNGKey(69), jax.random.PRNGKey(420)

	n_neurons = len(neurons) # ~100
	n_behaviors_ll = eigenworms.shape[1]  # 5

	# W1: (behaviors_ll x neurons)	W2: (dimensions x neurons)
	params = generate_params(model, seeds, (n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))

	# (samples x dimensions)
	cyber_manifold = pd.DataFrame(data=generate_manifold(discrete_states, n_dimensions),
								  columns=[f'CC{i+1}' for i in range(n_dimensions)])
	# (samples x neurons)
	N_rec, cca_params = generate_reconstructed_traces(cyber_manifold, n_neurons)
	cyber_traces_reconstructed = pd.DataFrame(data=N_rec,
											  columns=neurons)

	# (samples x neurons)
	if model.get_type():
		cyber_traces_residual = pd.DataFrame(data=generate_residual_traces(model, eigenworms, cyber_manifold, params['W1'], params['W2']),
											 columns=neurons)
	else:
		cyber_traces_residual = pd.DataFrame(data=generate_residual_traces(model, eigenworms, cyber_manifold, params['W'], None),
											 columns=neurons)

	# (samples x neurons)
	cyber_traces = pd.DataFrame(data=generate_traces(cyber_traces_reconstructed, cyber_traces_residual),
								columns=neurons)

	return cyber_name, cyber_manifold, cyber_traces_reconstructed, cyber_traces_residual, cyber_traces, params, cca_params
