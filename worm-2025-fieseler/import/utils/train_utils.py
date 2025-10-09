import os
import numpy as np
import pandas as pd
from flax.serialization import to_bytes, from_bytes

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def get_manifolds(traces, behavior, n_cc=2, n_pc=5):
	modes_neural, modes_behavioral = list(), list()
	recon_neural, recon_behavioral = list(), list()
	loadings_neural, loadings_behavioral = list(), list()
	vars_expl = list()
	vars_expl_rs = list()  # variance explained with respect to the reconstructed PCA data
	for trace, behavior in zip(traces, behavior):
		if n_pc < 1:
			trace_reconstructed_pca = trace
		else:
			pca = PCA(n_components=n_pc)
			trace_reduced_pca = pca.fit_transform(trace.values)
			trace_reconstructed_pca = pca.inverse_transform(trace_reduced_pca)

		cca = CCA(n_components=n_cc, scale=True)
		trace_reduced, behavior_reduced = cca.fit_transform(trace_reconstructed_pca, behavior.values)
		trace_reconstructed, behavior_reconstructed = cca.inverse_transform(trace_reduced, behavior_reduced)

		modes_neural.append(pd.DataFrame(data=trace_reduced, columns=[f'CC{n + 1}' for n in range(n_cc)]))
		modes_behavioral.append(pd.DataFrame(data=behavior_reduced, columns=[f'CC{n + 1}' for n in range(n_cc)]))

		recon_neural.append(pd.DataFrame(data=trace_reconstructed, columns=trace.columns))
		recon_behavioral.append(pd.DataFrame(data=behavior_reconstructed, columns=behavior.columns))

		loadings_neural.append(cca.x_loadings_)
		loadings_behavioral.append(cca.y_loadings_)

		var_ratio = get_variance_explained(trace_reconstructed, trace)
		var_ratio_rs = get_variance_explained(trace_reconstructed, trace_reconstructed_pca)
		vars_expl.append(var_ratio)
		vars_expl_rs.append(var_ratio_rs)

	return modes_neural, modes_behavioral, recon_neural, recon_behavioral, loadings_neural, loadings_behavioral, vars_expl, vars_expl_rs


def get_manifolds_pca(traces, n_pc=2):
	modes_manifold = list()
	traces_manifold = list()
	loadings = list()
	vars_expl = list()
	for trace in traces:
		pca = PCA(n_components=n_pc)
		trace_reduced = pca.fit_transform(trace)
		trace_reconstructed = pca.inverse_transform(trace_reduced)

		modes_manifold.append(pd.DataFrame(data=trace_reduced, columns=[f'PC{n + 1}' for n in range(2)]))
		traces_manifold.append(pd.DataFrame(data=trace_reconstructed, columns=trace.columns))

		loadings.append(pca.components_)

		var_ratio = sum(pca.explained_variance_ratio_)
		vars_expl.append(var_ratio)

	return modes_manifold, traces_manifold, loadings, vars_expl


def get_variance_explained(X_rec, X):
	X_res = X - X_rec

	X_var = X.var().sum()
	X_res_var = X_res.var().sum()

	X_var_ratio = 1 - X_res_var / X_var

	return X_var_ratio


def get_variance_explained_alt(X_rec, X):
	X_var = np.linalg.trace(np.cov(X))
	X_rec_var = np.linalg.trace(np.cov(X_rec))

	X_var_ratio = X_rec_var / X_var

	return X_var_ratio


def save_params(filenames, params, save_name):
	outdir_path = f'./out/models/{save_name}/params'
	os.makedirs(outdir_path, exist_ok=True)
	for filename, param in zip(filenames, params):
		with open(f'{outdir_path}/{filename}.msgpack', 'wb') as f:
			f.write(to_bytes(param))


def load_params(load_name, filenames=None):
	params = list()
	indir_path = f'./out/models/{load_name}/params'
	if filenames is None:
		filenames = sorted(os.listdir(indir_path))
	else:
		filenames = [filename + '.msgpack' for filename in filenames]
	for filename in filenames:
		with open(f'{indir_path}/{filename}', 'rb') as f:
			param_dict = from_bytes(None, f.read())
			param = list()
			for key in param_dict.keys():
				param.append(param_dict[key])
			params.append(param)
	return params


def save_losses(filenames, losses, save_name):
	# format: dataset x array(type x split x epoch)
	outdir_path = f'./out/models/{save_name}/hist'
	os.makedirs(outdir_path, exist_ok=True)
	for filename, loss in zip(filenames, losses):
		np.save(f'{outdir_path}/{filename}.npy', loss)


def load_losses(load_name, filenames=None):
	losses = list()
	indir_path = f'./out/models/{load_name}/hist'
	if filenames is None:
		filenames = sorted(os.listdir(indir_path))
	else:
		filenames = [filename + '.npy' for filename in filenames]
	for filename in filenames:
		losses.append(np.load(f'{indir_path}/{filename}'))
	return losses


def get_good_neuron_labels():
	return [
		"VB02",
		"DB01",
		"DD01",
		"RMDDR",
		"RMDDL",
		"SMDDR",
		"SMDDL",
		"RMED",
		"RMEV",
		"RMDVR",
		"DB02",
		"SIADR",
		"URXL",
		"URYVR",
		"RID",
	]
