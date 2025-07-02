import numpy as np
import pandas as pd
import jax


def generate_params(shape_X, shape_X_man):
	w1_offsets = np.tile(jax.random.normal(jax.random.PRNGKey(0), shape_X[1]), (5, 1))
	w2_offsets = np.tile(jax.random.normal(jax.random.PRNGKey(1), shape_X[1]), (1, 1))
	params = {
		'W1': np.ones(shape_X) + w1_offsets,
		'W2': np.ones(shape_X_man) + w2_offsets
	}
	return params

def generate_manifold(cyber_discrete_states, shape):
	# N = shape[0]
	# change_points = np.linspace(0, N, num=11, dtype=int)
	# values = (np.arange(11) % 2 - 0.5) * 2
	# result = np.zeros(N, dtype=int)
	# for i in range(len(change_points) - 1):
	# 	start = change_points[i]
	# 	end = change_points[i + 1]
	# 	result[start:end] = values[i]
	# dim1 = result.reshape(N,)

	weight = -1
	dim1 = (cyber_discrete_states * 2 - 1) * weight
	if shape[1] <= 1: return dim1

	dims_n = list()
	for i in range(shape[1] - 1):
		dims_n.append(np.zeros((shape[0],)))
	dims_n = np.vstack(dims_n)

	return np.vstack((dim1, dims_n)).T

def generate_redisual_traces(X, X_man, W1, W2):
	traces = X @ W1 * jax.nn.sigmoid(X_man @ W2)
	return traces #- np.quantile(traces, 0.1, axis=0)

def generate_traces(X_man, X_residual, variance=1):
	if X_man.values.shape[1] != 1:
		print('warning! no idea what to do with a multi-dim manifold')

	X_values = X_residual.values
	X_man_expanded = np.tile(X_man.values, (1, X_values.shape[1]))
	X_man_expanded *= variance
	X_man_expanded = X_man_expanded #+ variance * 2 * 0.75

	X = pd.DataFrame(data=X_values + X_man_expanded, columns=X_residual.columns)

	return X

def artificialize(filename, neurons, eigenworms, discrete_states):
	dim = 1

	cybername = '_'.join(['ZIMMERMANN_artificial'] + filename.split('_')[2:])
	cyber_eigenworms = eigenworms * 0.2
	cyber_discrete_states = discrete_states['REV'].values

	X = cyber_eigenworms
	X_man = pd.DataFrame(data=generate_manifold(cyber_discrete_states, (len(neurons), dim)), columns=[f'PC{i+1}' for i in range(dim)])

	n_neurons = len(neurons)
	n_behaviors_ll = X.shape[1]
	n_dimensions = X_man.shape[1]

	params = generate_params((n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))
	cyber_traces_residual = pd.DataFrame(data=generate_redisual_traces(cyber_eigenworms.values, X_man.values, params['W1'], params['W2']),
										 columns=neurons)

	cyber_traces = pd.DataFrame(data=generate_traces(X_man, cyber_traces_residual), columns=neurons)

	return cybername, cyber_traces, cyber_traces_residual, X_man, params
