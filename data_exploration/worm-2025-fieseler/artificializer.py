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
			'W1': 0.5 * jnp.tile(jax.random.normal(seeds[0], shape_X_bll[1]), (shape_X_bll[0], 1)),
			'W2': 0.5 * jnp.tile(jnp.abs(jax.random.normal(seeds[1], shape_X_bll[1])), (shape_X_man[0], 1))
		}
	else:
		return {
			'W': 0.4 * jnp.tile(jax.random.normal(seeds[0], shape_X_bll[1]), (shape_X_bll[0], 1))
		}

def generate_manifold(discrete_states, shape):
	assert shape[1] == 1
	weight = 1
	manifold = np.array((discrete_states * 2 - 1) * weight, dtype=float)
	return manifold

def generate_redisual_traces(model, X_bll, X_man, W1, W2):
	if model.get_type():
		traces = model.predict(X_bll, X_man, W1, W2)
	else:
		traces = model.predict(X_bll, W1)
	return traces

def generate_traces(X_residual, X_man):
	assert X_man.values.shape[1] == 1

	weight = 1
	X_man_reconstructed = np.tile(X_man.values * weight, (1, X_residual.values.shape[1]))  # TODO: better way (inverse CCA)
	X = pd.DataFrame(data=X_residual.values + X_man_reconstructed, columns=X_residual.columns)
	return X

def artificialize(model, filename, neurons, eigenworms, discrete_states):
	seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

	n_dimensions = 1
	n_neurons = len(neurons)
	n_behaviors_ll = eigenworms.shape[1]

	cyber_name = '_'.join(['ZIMMERMANN_artificial'] + filename.split('_')[2:])

	cyber_manifold = pd.DataFrame(data=generate_manifold(discrete_states['FWD'].values, (discrete_states['FWD'].values.shape[0], n_dimensions)),
								  columns=[f'CC{i+1}' for i in range(n_dimensions)])

	params = generate_params(model, seeds, (n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))

	if model.get_type():
		cyber_traces_residual = pd.DataFrame(data=generate_redisual_traces(model, eigenworms.values, cyber_manifold.values, params['W1'], params['W2']),
											 columns=neurons)
	else:
		cyber_traces_residual = pd.DataFrame(data=generate_redisual_traces(model, eigenworms.values, cyber_manifold.values, params['W'], None),
											 columns=neurons)

	cyber_traces = pd.DataFrame(data=generate_traces(cyber_traces_residual, cyber_manifold),
								columns=neurons)

	return cyber_name, cyber_manifold, cyber_traces_residual, cyber_traces, params
