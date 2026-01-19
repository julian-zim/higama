import math
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

import jax
import jax.numpy as jnp


def generate_params(weights, model, seeds, shape_WB, shape_WM, scaling=0.1):
	bom_lin = weights[0]

	if model.get_type():
		gating = abs(weights[1])
		hgm_lin = (1.0 + gating) * bom_lin
		hgm_sig = -math.log(gating) * np.sign(weights[1])

		return {
			# (behavior x neurons)
			'WB': hgm_lin * jnp.tile(jnp.ones((shape_WB[1],)),(shape_WB[0], 1))
				  + scaling * jax.random.normal(seeds[0], (shape_WB[0], shape_WB[1])),
			# (dimensions x neurons)
			'WM': hgm_sig * jnp.tile(jnp.ones((shape_WB[1],)),(shape_WM[0], 1))
				  + scaling * jax.random.normal(seeds[1], (shape_WM[0], shape_WB[1])),
		}
	else:
		return {
			# (behavior x neurons)
			'WB': bom_lin * jnp.tile(jnp.ones((shape_WB[1],)), (shape_WB[0], 1))
				 + scaling * jax.random.normal(seeds[0], (shape_WB[0], shape_WB[1])),
		}

def generate_manifold(states, n_dimensions):
	# define the neuronal manifold as a smoothed high level behavior
	states_n = states[:, :n_dimensions]
	states_zs = (states_n - states_n.mean(axis=0)) / jnp.where((states_std := states_n.std(axis=0)) == 0.0, 1.0, states_std)

	manifold = states_zs
	return gaussian_filter1d(manifold, sigma=2.0, axis=0)

def generate_cca_params(seeds, n_dimensions, n_neurons, scaling=1):
	# (dimensions x neurons)
	weights = np.tile(
		np.array([d / n_dimensions for d in range(n_dimensions, 0, -1)], dtype=np.float32).reshape(-1, 1),
		(1, n_neurons)
	) + scaling * jax.random.normal(seeds[2], (n_dimensions, n_neurons))
	return weights

def generate_reconstructed_traces(N_man, cca_weights):
	return N_man.values @ cca_weights

def generate_residual_traces(model, B, N_man, WB, WM):
	prep_p = {
		'mean': B.mean(axis=0),
		'std': jnp.where((B_std := jnp.std(B, axis=0)) == 0.0, 1.0, B_std),
	}

	if model.get_type():
		traces = model.predict_res(prep_p, B, N_man, WB, WM)
	else:
		traces = model.predict_res(prep_p, B, WB)
	return traces

def generate_traces(N_rec, N_res):
	return N_rec.values + N_res.values

def artificialize(weights, model, filename, neuron_labels, behavior, states, n_dimensions=2):
	assert n_dimensions <= states.values.shape[1]

	cyber_name = filename.replace('worm', 'wyrm')
	seeds = jax.random.PRNGKey(69), jax.random.PRNGKey(420), jax.random.PRNGKey(0)

	n_neurons = len(neuron_labels) # ~100
	n_behaviors = behavior.values.shape[1]  # 5

	# WB: (behaviors_ll x neurons)	WM: (dimensions x neurons)
	params = generate_params(weights, model, seeds, (n_behaviors, n_neurons), (n_dimensions, n_neurons))

	# (timepoints x dimensions)
	cyber_manifold = pd.DataFrame(data=generate_manifold(states.values, n_dimensions),
								  columns=[f'CV{i+1}' for i in range(n_dimensions)])

	# (dimensions x neurons)
	cca_params = generate_cca_params(seeds, n_dimensions, n_neurons)

	# (timepoints x neurons)
	cyber_traces_reconstructed = pd.DataFrame(data=generate_reconstructed_traces(cyber_manifold, cca_params),
											  columns=neuron_labels)

	# (timepoints x neurons)
	cyber_traces_residual = pd.DataFrame(data=generate_residual_traces(model, behavior.values, cyber_manifold.values, params['WB'], (params['WM'] if model.get_type() else None)),
										 columns=neuron_labels)

	# (timepoints x neurons)
	cyber_traces = pd.DataFrame(data=generate_traces(cyber_traces_reconstructed, cyber_traces_residual),
								columns=neuron_labels)

	return cyber_name, cyber_manifold, cyber_traces_reconstructed, cyber_traces_residual, cyber_traces, params, cca_params
