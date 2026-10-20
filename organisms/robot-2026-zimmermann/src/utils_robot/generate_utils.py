#region imports
import sys, os
import math
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

import jax
import jax.numpy as jnp

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
from models import bom
from models import hgm
#endregion


def generate_params(behavior, gating, model_name, n_dimensions, n_neurons, n_behaviors, seeds):
	# shape_WB = (n_behaviors, n_neurons)
	# shape_WZ = (n_dimensions, n_neurons)

	WB = jnp.ones((n_behaviors, n_neurons)) * behavior + .2 * jax.random.normal(seeds[0], (n_behaviors, n_neurons))
	# print(f'{WB=}')

	if model_name == 'bom':
		return {
			# (behavior x neurons)
			'WB': WB,
		}

	gating = jnp.atleast_1d(jnp.array(gating))
	# print(f'{gating.shape=}', f'{gating=}')
	if len(gating.shape) == 1:
		gating = jnp.tile(gating[:, None], (1, n_neurons))
		# print(f'{gating.shape=}', f'{gating=}')

	signs = jnp.sign(gating)
	gating = jnp.abs(gating)
	# print(f'{signs.shape=}', f'{signs=}')

	if gating.shape[0] == 1:
		signs = np.tile(signs, (n_dimensions, 1))
		gating = np.tile(gating ** (1. / n_dimensions), (n_dimensions, 1))
		print(f'{gating.shape=}', f'{gating=}')
		# print(f'{signs.shape=}', f'{signs=}')

	gating_noise = .2 * gating * jnp.abs(jax.random.normal(seeds[1], gating.shape))
	gating = np.clip(gating + gating_noise, 1e-12, 1.)

	WZ = signs * -jnp.log(jnp.abs(gating))
	# print(f'{WZ=}')
	gating_min = jnp.exp(-jnp.abs(WZ).sum(axis=0))
	print(f'{gating_min=}')
	WB_hgm = (1. + gating_min) * WB
	# print(f'{WB_hgm=}')

	return {
		# (behavior x neurons)
		'WB': WB_hgm,
		# (dimensions x neurons)
		'WZ': WZ,
	}


def generate_manifold(states, n_dimensions):
	# define the neuronal manifold as a smoothed high level behavior
	states_n = states[:, :n_dimensions]
	states_zs = (states_n - states_n.mean(axis=0)) / jnp.where((states_std := states_n.std(axis=0)) == 0.0, 1.0,
															   states_std)

	manifold = states_zs
	return gaussian_filter1d(manifold, sigma=2.0, axis=0)


def generate_manifold_params(n_dimensions, n_neurons, seeds):
	# (dimensions x neurons)
	weights = 1. * jax.random.normal(seeds[2], (n_dimensions, n_neurons)) * np.tile(
		np.array([d / n_dimensions for d in range(n_dimensions, 0, -1)], dtype=np.float32).reshape(-1, 1),
		(1, n_neurons)
	)
	return weights


def generate_reconstructed_traces(N_man, cca_weights):
	return N_man.values @ cca_weights


def generate_concrete_traces(model_name, B, N_man, WB, WZ):
	prep_p = {
		'mean': B.mean(axis=0),
		'std': jnp.where((B_std := jnp.std(B, axis=0)) == 0.0, 1.0, B_std),
	}

	if model_name == 'hgm':
		traces = hgm.predict_res(prep_p, B, N_man, WB, WZ)

		# N_man_zs = (N_man - N_man.mean(axis=0)) / np.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)
		# N_man_zs = (N_man - N_man.mean(axis=0)) / np.where((N_std := N_man.std(axis=0)) == 0.0, 1.0, N_std)
		N_man_zs = (N_man - N_man.min(axis=0)) / jnp.where((N_range := N_man.max(axis=0) - N_man.min(axis=0)) == 0.0, 1.0, N_range) * 2 - 1  # BUGFIX!
		modulation_terms_copy = 1. / (1. + jnp.exp(-(N_man_zs @ WZ)))
	else:
		traces = bom.predict_res(prep_p, B, WB)
		modulation_terms_copy = np.ones(traces.shape, dtype=traces.dtype)

	return traces, modulation_terms_copy


def generate_other_traces(concrete_traces, rate, dt, decay=None):
	if decay is None:
		decay = 2.0 * dt
	shape = concrete_traces.values.shape

	# poisson series
	p = rate * dt
	events = (np.random.rand(*shape) < p).astype(np.float32) * np.sign(np.random.rand(*shape) - 0.5) * concrete_traces.values.max(axis=0)

	# two side exponential filtering
	gamma = np.exp(-dt / decay)
	trace_fwd = np.zeros(events.shape, dtype=np.float32)
	for t in range(1, events.shape[0]):
		trace_fwd[t, :] = gamma * trace_fwd[t - 1, :] + events[t, :]

	trace_bwd = np.zeros(events.shape, dtype=np.float32)
	for t in range(events.shape[0] - 2, -1, -1):
		trace_bwd[t, :] = gamma * trace_bwd[t + 1, :] + events[t, :]

	trace = concrete_traces.values.max(axis=0) / 5. * (trace_fwd + trace_bwd)
	return trace


def generate_residual_traces(concrete_traces, other_traces):
	return concrete_traces.values + other_traces.values


def generate_traces(N_rec, N_res):
	return N_rec.values + N_res.values


def artificialize(weights, model_name, filename, neuron_labels, behavior, states, sampling_rate, n_dimensions=2):
	assert n_dimensions <= states.values.shape[1]

	behavior_scale, gating = weights

	cyber_name = filename.replace('worm', 'wyrm')
	seeds = jax.random.PRNGKey(69), jax.random.PRNGKey(420), jax.random.PRNGKey(0)
	np.random.seed(12)

	n_neurons = len(neuron_labels)  # ~100
	n_behaviors = behavior.values.shape[1]  # 5

	# WB: (behaviors_ll x neurons)	WZ: (dimensions x neurons)
	params = generate_params(behavior_scale, gating, model_name, n_dimensions, n_neurons, n_behaviors, seeds)

	# (timepoints x dimensions)
	cyber_manifold = pd.DataFrame(data=generate_manifold(states.values, n_dimensions),
								  columns=[f'M{i + 1}' for i in range(n_dimensions)])

	# (dimensions x neurons)
	params_manifold = generate_manifold_params(n_dimensions, n_neurons, seeds)

	# (timepoints x neurons)
	cyber_traces_reconstructed = pd.DataFrame(data=generate_reconstructed_traces(cyber_manifold, params_manifold),
											  columns=neuron_labels)

	# (timepoints x neurons)
	concrete_traces, modulation_terms = generate_concrete_traces(model_name, behavior.values, cyber_manifold.values, params['WB'],
							 (params['WZ'] if model_name == 'hgm' else None))
	cyber_traces_concrete = pd.DataFrame(
		data=concrete_traces,
		columns=neuron_labels)
	cyber_modulation_terms = pd.DataFrame(
		data=modulation_terms,
		columns=neuron_labels)

	# (timepoints x neurons)
	cyber_traces_other = pd.DataFrame(
		data=generate_other_traces(cyber_traces_concrete, rate=0.02, dt=1.0 / sampling_rate, decay=None),
		columns=neuron_labels)

	# (timepoints x neurons)
	cyber_traces_residual = pd.DataFrame(
		data=generate_residual_traces(cyber_traces_concrete, cyber_traces_other),
		columns=neuron_labels)

	# (timepoints x neurons)
	cyber_traces = pd.DataFrame(data=generate_traces(cyber_traces_reconstructed, cyber_traces_residual),
								columns=neuron_labels)

	return (cyber_name,
			cyber_manifold, cyber_traces_reconstructed,
			cyber_traces_residual, cyber_modulation_terms, cyber_traces_concrete, cyber_traces_other, cyber_traces,
			params, params_manifold)
