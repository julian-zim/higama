import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import math
from scipy.ndimage import gaussian_filter1d


def generate_params(weights, model, seeds, shape_B_ll, shape_N_man, scaling=0.1):
	bom_lin = weights[0]

	if model.get_type():
		gating = abs(weights[1])
		hgm_lin = (1.0 + gating) * bom_lin
		hgm_sig = -math.log(gating) * np.sign(weights[1])

		return {
			'W1': hgm_lin * jnp.tile(jnp.ones((shape_N_man[1],)),(shape_B_ll[0], 1))
				  + scaling * jax.random.normal(seeds[0], (shape_B_ll[0], shape_N_man[1])),
			'W2': hgm_sig * jnp.tile(jnp.ones((shape_N_man[1],)),(shape_N_man[0], 1))
				  + scaling * jax.random.normal(seeds[0], (shape_N_man[0], shape_N_man[1])),
		}
	else:
		return {
			'W': bom_lin * jnp.tile(jnp.ones((shape_N_man[1],)), (shape_B_ll[0], 1))
				 + scaling * jax.random.normal(seeds[0], (shape_B_ll[0], shape_N_man[1])),
		}

def generate_manifold(discrete_states, n_dimensions):
	# define the neuronal manifold as a smoothed high level behavior
	manifold = discrete_states.values[:, :n_dimensions] - np.mean(discrete_states.values[:, :n_dimensions], axis=0)
	return gaussian_filter1d(manifold, sigma=2.0, axis=0)

def generate_cca_params(seeds, n_dimensions, n_neurons, scaling=1):
	# (dimensions x neurons)
	cca_weights = np.tile(
		np.array([d / n_dimensions for d in range(n_dimensions, 0, -1)], dtype=np.float32).reshape(-1, 1),
		(1, n_neurons)
	) + scaling * jax.random.normal(seeds[0], (n_dimensions, n_neurons))
	# cca_weights @= scaling * np.abs(jax.random.normal(seeds[1], (n_neurons, n_neurons)))

	return cca_weights

def generate_reconstructed_traces(N_man, cca_weights):
	return N_man.values @ cca_weights  # normalize?

def generate_residual_traces(model, B_ll, N_man, W1, W2):
	if model.get_type():
		traces = model.predict_res(B_ll.values, N_man.values, W1, W2)
	else:
		traces = model.predict_res(B_ll.values, W1)
	return traces

def generate_traces(N_rec, N_res):
	return N_rec.values + N_res.values

def artificialize(weights, model, filename, neurons, eigenworms, discrete_states, n_dimensions=2):
	assert n_dimensions <= 4

	cyber_name = filename.replace('worm', 'wyrm')
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
