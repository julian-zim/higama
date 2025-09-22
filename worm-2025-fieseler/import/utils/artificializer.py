import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp


def generate_params(weights, model, seeds, shape_B_ll, shape_N_man):
	bom_lin = weights[0]
	hgm_sig = weights[1]
	hgm_lin = bom_lin / jax.nn.sigmoid(hgm_sig)

	if model.get_type():
		return {
			'W1': hgm_lin * jnp.tile(jax.random.normal(seeds[0], (shape_B_ll[1],)), (shape_B_ll[0], 1)),
			'W2': -hgm_sig * jnp.ones((shape_N_man[0], shape_B_ll[1])),
			# 'W2': -hgm_sig * jnp.tile(jnp.abs(jax.random.normal(seeds[1], (shape_B_ll[1],))), (shape_N_man[0], 1))
		}
	else:
		return {
			'W': bom_lin * jnp.tile(jax.random.normal(seeds[0], (shape_B_ll[1],)), (shape_B_ll[0], 1))
			# 'W': bom_lin * jnp.ones(shape_B_ll),
		}

def generate_cca_params(seeds, n_dimensions, n_neurons, scaling=0.25):
	# (dimensions x neurons)
	# cca_weights = np.tile(np.array([math.e ** (1-d) for d in range(0, n_dimensions)]).reshape(-1, 1), (1, n_neurons))
	# cca_weights @= np.diag(np.array([(i+1) * scaling + (1. - scaling) for i in range(n_neurons)]))

	cca_weights = 0.1 * np.abs(jax.random.normal(seeds[0], (n_dimensions, n_neurons)))
	cca_weights @= (np.diag(np.array([(i+1) * scaling + (1. - scaling) for i in range(n_neurons)]))
					+ 0.1 * np.abs(jax.random.normal(seeds[1], (n_neurons, n_neurons))))

	return cca_weights

def generate_manifold(discrete_states, n_dimensions):
	# define the neuronal manifold as the high level behavior
	manifold = np.array((discrete_states.values[:, :n_dimensions] * 2 - 1), dtype=float)
	return manifold

def generate_reconstructed_traces(N_man, cca_weights):
	return N_man.values @ cca_weights

def generate_residual_traces(model, B_ll, N_man, W1, W2):
	if model.get_type():
		traces = model.predict(B_ll.values, N_man.values, W1, W2)
	else:
		traces = model.predict(B_ll.values, W1)
	return traces

def generate_traces(N_rec, N_res):
	return N_rec.values + N_res.values

def artificialize(weights, model, filename, neurons, eigenworms, discrete_states, n_dimensions=2):
	assert n_dimensions <= 4

	cyber_name = '_'.join(['ZIMMERMANN_artificial'] + filename.split('_')[2:])
	seeds = jax.random.PRNGKey(69), jax.random.PRNGKey(420)

	n_neurons = len(neurons) # ~100
	n_behaviors_ll = eigenworms.shape[1]  # 5

	# W1: (behaviors_ll x neurons)	W2: (dimensions x neurons)
	params = generate_params(weights, model, seeds, (n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))

	# (samples x dimensions)
	cyber_manifold = pd.DataFrame(data=generate_manifold(discrete_states, n_dimensions),
								  columns=[f'CC{i+1}' for i in range(n_dimensions)])

	# (dimensions x neurons)
	cca_params = generate_cca_params(seeds, cyber_manifold.values.shape[1], n_neurons)

	# (samples x neurons)
	cyber_traces_reconstructed = pd.DataFrame(data=generate_reconstructed_traces(cyber_manifold, cca_params),
											  columns=neurons)

	# (samples x neurons)
	cyber_traces_residual = pd.DataFrame(data=generate_residual_traces(model, eigenworms, cyber_manifold, params['W' + ('1' if model.get_type() else '')], (params['W2'] if model.get_type() else None)),
										 columns=neurons)

	# (samples x neurons)
	cyber_traces = pd.DataFrame(data=generate_traces(cyber_traces_reconstructed, cyber_traces_residual),
								columns=neurons)

	return cyber_name, cyber_manifold, cyber_traces_reconstructed, cyber_traces_residual, cyber_traces, params, cca_params
