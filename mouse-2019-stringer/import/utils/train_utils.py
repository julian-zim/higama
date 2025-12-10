import os
import numpy as np
import pandas as pd
from flax.serialization import to_bytes, from_bytes

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical


#region manifolds

def get_manifolds_plsc(traces, behavior, n_ls=3):
	modes_neural, modes_behavioral = list(), list()
	recon_neural, recon_behavioral = list(), list()
	loadings_neural, loadings_behavioral = list(), list()
	vars_expl = list()
	for trace, behavior in zip(traces, behavior):
		# reduction
		plsc = PLSCanonical(n_components=n_ls, scale=True)
		trace_reduced, behavior_reduced = plsc.fit_transform(trace, behavior.values)
		trace_reconstructed, behavior_reconstructed = plsc.inverse_transform(trace_reduced, behavior_reduced)

		modes_neural.append(pd.DataFrame(data=trace_reduced, columns=[f'LS{n + 1}' for n in range(n_ls)]))
		modes_behavioral.append(pd.DataFrame(data=behavior_reduced, columns=[f'LS{n + 1}' for n in range(n_ls)]))

		recon_neural.append(pd.DataFrame(data=trace_reconstructed, columns=trace.columns))
		recon_behavioral.append(pd.DataFrame(data=behavior_reconstructed, columns=behavior.columns))

		loadings_neural.append(plsc.x_loadings_)
		loadings_behavioral.append(plsc.y_loadings_)

		# variance explained
		var_ratio = list()

		var_ratio_cumulative = list()
		for k in range(1, n_ls+1):  # VE k = VE :k - VE :k-1
			trace_reduced_to_k = np.zeros_like(trace_reduced)
			trace_reduced_to_k[:, :k] = trace_reduced[:, :k]
			behavior_reduced_to_k = np.zeros_like(behavior_reduced)
			behavior_reduced_to_k[:, :k] = behavior_reduced[:, :k]

			trace_reconstructed_to_k, _ = plsc.inverse_transform(trace_reduced_to_k, behavior_reduced_to_k)

			var_ratio_cumulative.append(get_variance_explained(trace_reconstructed_to_k, trace))
			var_ratio_k = var_ratio_cumulative[k-1] - (var_ratio_cumulative[k-1-1] if k > 1 else 0)

			var_ratio.append(var_ratio_k)

		vars_expl.append(var_ratio)

	return modes_neural, modes_behavioral, recon_neural, recon_behavioral, loadings_neural, loadings_behavioral, vars_expl


def get_manifolds_cca(traces, behavior, n_cv=3, n_pc=0):
	modes_neural, modes_behavioral = list(), list()
	recon_neural, recon_behavioral = list(), list()
	loadings_neural, loadings_behavioral = list(), list()
	vars_expl, vars_expl_rs = list(), list()
	for trace, behavior in zip(traces, behavior):
		# pca preprocessing
		if n_pc > 0:
			pca = PCA(n_components=n_pc)
			trace_reduced_pca = pca.fit_transform(trace.values)
			trace_reconstructed_pca = pca.inverse_transform(trace_reduced_pca)
		else:
			trace_reconstructed_pca = trace

		# reduction
		cca = CCA(n_components=n_cv, scale=True)
		trace_reduced, behavior_reduced = cca.fit_transform(trace_reconstructed_pca, behavior.values)
		trace_reconstructed, behavior_reconstructed = cca.inverse_transform(trace_reduced, behavior_reduced)

		modes_neural.append(pd.DataFrame(data=trace_reduced, columns=[f'CV{n + 1}' for n in range(n_cv)]))
		modes_behavioral.append(pd.DataFrame(data=behavior_reduced, columns=[f'CV{n + 1}' for n in range(n_cv)]))

		recon_neural.append(pd.DataFrame(data=trace_reconstructed, columns=trace.columns))
		recon_behavioral.append(pd.DataFrame(data=behavior_reconstructed, columns=behavior.columns))

		loadings_neural.append(cca.x_loadings_)
		loadings_behavioral.append(cca.y_loadings_)

		# variance explained
		var_ratio = list()
		var_ratio_rs = list()

		var_ratio_cumulative = list()
		var_ratio_rs_cumulative = list()
		for k in range(1, n_cv+1):  # VE k = VE :k - VE :k-1
			trace_reduced_to_k = np.zeros_like(trace_reduced)
			trace_reduced_to_k[:, :k] = trace_reduced[:, :k]
			behavior_reduced_to_k = np.zeros_like(behavior_reduced)
			behavior_reduced_to_k[:, :k] = behavior_reduced[:, :k]

			trace_reconstructed_to_k, _ = cca.inverse_transform(trace_reduced_to_k, behavior_reduced_to_k)

			var_ratio_cumulative.append(get_variance_explained(trace_reconstructed_to_k, trace))
			var_ratio_rs_cumulative.append(get_variance_explained(trace_reconstructed_to_k, trace_reconstructed_pca))

			var_ratio_k = var_ratio_cumulative[k-1] - (var_ratio_cumulative[k-1-1] if k > 1 else 0)
			var_ratio_rs_k = var_ratio_rs_cumulative[k-1] - (var_ratio_rs_cumulative[k-1-1] if k > 1 else 0)

			var_ratio.append(var_ratio_k)
			var_ratio_rs.append(var_ratio_rs_k)

		vars_expl.append(var_ratio)
		vars_expl_rs.append(var_ratio_rs)

	return modes_neural, modes_behavioral, recon_neural, recon_behavioral, loadings_neural, loadings_behavioral, vars_expl, vars_expl_rs


def get_manifolds_pca(traces, n_pc=3):
	modes = list()
	recon = list()
	loadings = list()
	vars_expl = list()
	for trace in traces:
		# reduction
		pca = PCA(n_components=n_pc)
		trace_reduced = pca.fit_transform(trace)
		trace_reconstructed = pca.inverse_transform(trace_reduced)

		modes.append(pd.DataFrame(data=trace_reduced, columns=[f'PC{n + 1}' for n in range(n_pc)]))
		recon.append(pd.DataFrame(data=trace_reconstructed, columns=trace.columns))

		loadings.append(pca.components_)

		# variance explained
		vars_expl.append(pca.explained_variance_ratio_)

	return modes, recon, loadings, vars_expl

#endregion


#region variance explained

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

#endregion


#region models

def save_params(filenames, params, save_name):
	outdir_path = f'./out/keep/models/{save_name}/params'
	os.makedirs(outdir_path, exist_ok=True)
	for filename, param in zip(filenames, params):
		with open(f'{outdir_path}/{filename}.msgpack', 'wb') as f:
			f.write(to_bytes(param))


def load_params(load_name, filenames=None):
	params = list()
	indir_path = f'./out/keep/models/{load_name}/params'
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
	outdir_path = f'./out/keep/models/{save_name}/hist'
	os.makedirs(outdir_path, exist_ok=True)
	for filename, loss in zip(filenames, losses):
		np.save(f'{outdir_path}/{filename}.npy', loss)


def load_losses(load_name, filenames=None):
	losses = list()
	indir_path = f'./out/keep/models/{load_name}/hist'
	if filenames is None:
		filenames = sorted(os.listdir(indir_path))
	else:
		filenames = [filename + '.npy' for filename in filenames]
	for filename in filenames:
		losses.append(np.load(f'{indir_path}/{filename}'))
	return losses

#endregion


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
