import os
import numpy as np
import pandas as pd

from flax.serialization import to_bytes, from_bytes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical


#region manifolds

def get_manifolds_plsc(traces, behavior, zscore=(False, False), n_ls=3):
	modes_neural, modes_behavioral = list(), list()
	recon_neural, recon_behavioral = list(), list()
	loadings_neural, loadings_behavioral = list(), list()
	vars_expl = list()
	for trace, behavior in zip(traces, behavior):
		# zscoring
		if zscore[0]:
			trace_scaler = StandardScaler()
			trace = pd.DataFrame(data=trace_scaler.fit_transform(trace.values), columns=trace.columns)
		if zscore[1]:
			behavior_scaler = StandardScaler()
			behavior = pd.DataFrame(data=behavior_scaler.fit_transform(behavior.values), columns=behavior.columns)

		# reduction
		plsc = PLSCanonical(n_components=n_ls, scale=False)
		trace_reduced, behavior_reduced = plsc.fit_transform(trace.values, behavior.values)
		trace_reconstructed, behavior_reconstructed = plsc.inverse_transform(trace_reduced, behavior_reduced)

		modes_neural.append(pd.DataFrame(data=trace_reduced, columns=[f'LS{n + 1}' for n in range(n_ls)]))
		modes_behavioral.append(pd.DataFrame(data=behavior_reduced, columns=[f'LS{n + 1}' for n in range(n_ls)]))

		if zscore[0]: trace_reconstructed = trace_scaler.inverse_transform(trace_reconstructed)
		if zscore[1]: behavior_reconstructed = behavior_scaler.inverse_transform(behavior_reconstructed)

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

			var_ratio_cumulative.append(get_variance_explained(trace_reconstructed_to_k, trace.values))
			var_ratio_k = var_ratio_cumulative[k-1] - (var_ratio_cumulative[k-1-1] if k > 1 else 0)

			var_ratio.append(var_ratio_k)

		vars_expl.append(var_ratio)

	return modes_neural, modes_behavioral, recon_neural, recon_behavioral, loadings_neural, loadings_behavioral, vars_expl


def get_manifolds_cca(traces, behavior, zscore=(False, False), n_cv=3, n_pc=0):
	modes_neural, modes_behavioral = list(), list()
	recon_neural, recon_behavioral = list(), list()
	loadings_neural, loadings_behavioral = list(), list()
	vars_expl, vars_expl_rs = list(), list()
	for trace, behavior in zip(traces, behavior):
		# zscoring
		if zscore[0]:
			trace_scaler = StandardScaler()
			trace = pd.DataFrame(data=trace_scaler.fit_transform(trace.values), columns=trace.columns)
		if zscore[1]:
			behavior_scaler = StandardScaler()
			behavior = pd.DataFrame(data=behavior_scaler.fit_transform(behavior.values), columns=behavior.columns)

		# pca preprocessing
		if n_pc > 0:
			pca = PCA(n_components=n_pc)
			trace_reduced_pca = pca.fit_transform(trace.values)
			trace_reconstructed_pca = pca.inverse_transform(trace_reduced_pca)

		# reduction
		cca = CCA(n_components=n_cv, scale=False)
		trace_reduced, behavior_reduced = cca.fit_transform(trace_reconstructed_pca if n_pc > 0 else trace.values, behavior.values)
		trace_reconstructed, behavior_reconstructed = cca.inverse_transform(trace_reduced, behavior_reduced)

		if zscore[0]: trace_reconstructed = trace_scaler.inverse_transform(trace_reconstructed)
		if zscore[1]: behavior_reconstructed = behavior_scaler.inverse_transform(behavior_reconstructed)

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

			var_expl = get_variance_explained(trace_reconstructed_to_k, trace.values)
			var_ratio_cumulative.append(var_expl)
			if n_pc > 0:
				var_ratio_rs_cumulative.append(get_variance_explained(trace_reconstructed_to_k, trace_reconstructed_pca))
			else:
				var_ratio_rs_cumulative.append(var_expl)

			var_ratio_k = var_ratio_cumulative[k-1] - (var_ratio_cumulative[k-1-1] if k > 1 else 0)
			var_ratio_rs_k = var_ratio_rs_cumulative[k-1] - (var_ratio_rs_cumulative[k-1-1] if k > 1 else 0)

			var_ratio.append(var_ratio_k)
			var_ratio_rs.append(var_ratio_rs_k)

		vars_expl.append(var_ratio)
		vars_expl_rs.append(var_ratio_rs)

	return modes_neural, modes_behavioral, recon_neural, recon_behavioral, loadings_neural, loadings_behavioral, vars_expl, vars_expl_rs


def get_manifolds_pca(traces, zscore=False, n_pc=3):
	modes = list()
	recon = list()
	loadings = list()
	vars_expl = list()
	for trace in traces:
		# zscoring
		if zscore:
			scaler = StandardScaler()
			trace = pd.DataFrame(data=scaler.fit_transform(trace.values), columns=trace.columns)

		# reduction
		pca = PCA(n_components=n_pc)
		trace_reduced = pca.fit_transform(trace.values)
		trace_reconstructed = pca.inverse_transform(trace_reduced)

		if zscore: trace_reconstructed = scaler.inverse_transform(trace_reconstructed)

		modes.append(pd.DataFrame(data=trace_reduced, columns=[f'PC{n + 1}' for n in range(n_pc)]))
		recon.append(pd.DataFrame(data=trace_reconstructed, columns=trace.columns))

		loadings.append(pca.components_)

		# variance explained
		vars_expl.append(pca.explained_variance_ratio_)

	return modes, recon, loadings, vars_expl

#endregion


#region variance explained

def get_variance_explained(X_rec, X):
	X_rec_var = np.linalg.trace(np.cov(X_rec, rowvar=False))
	X_var = np.linalg.trace(np.cov(X, rowvar=False))

	X_var_ratio = X_rec_var / X_var

	return X_var_ratio


def get_variance_explained_orth(X_rec, X):
	X_res = X - X_rec

	X_res_var = X_res.var().sum()
	X_var = X.var().sum()

	X_var_ratio = 1 - X_res_var / X_var

	return X_var_ratio

#endregion


#region models

def save_params(filenames, params, save_path):
	organism_name = save_path.split('/')[0]
	model_name = '/'.join(save_path.split('/')[1:])
	outdir_path = f'{os.path.dirname(os.path.abspath(__file__))}/../organisms/{organism_name}/out/models/{model_name}/params'
	os.makedirs(outdir_path, exist_ok=True)
	for filename, param in zip(filenames, params):
		with open(f'{outdir_path}/{filename}.msgpack', 'wb') as f:
			f.write(to_bytes(param))


def load_params(load_path, filenames=None):
	organism_name = load_path.split('/')[0]
	model_name = '/'.join(load_path.split('/')[1:])
	params = list()
	indir_path = f'{os.path.dirname(os.path.abspath(__file__))}/../organisms/{organism_name}/out/models/{model_name}/params'
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


def save_losses(filenames, losses, save_path):
	organism_name = save_path.split('/')[0]
	model_name = '/'.join(save_path.split('/')[1:])
	# format: dataset x array(type x split x epoch)
	outdir_path = f'{os.path.dirname(os.path.abspath(__file__))}/../organisms/{organism_name}/out/models/{model_name}/hist'
	os.makedirs(outdir_path, exist_ok=True)
	for filename, loss in zip(filenames, losses):
		np.save(f'{outdir_path}/{filename}.npy', loss)


def load_losses(load_path, filenames=None):
	organism_name = load_path.split('/')[0]
	model_name = '/'.join(load_path.split('/')[1:])
	losses = list()
	indir_path = f'{os.path.dirname(os.path.abspath(__file__))}/../organisms/{organism_name}/out/models/{model_name}/hist'
	if filenames is None:
		filenames = sorted(os.listdir(indir_path))
	else:
		filenames = [filename + '.npy' for filename in filenames]
	for filename in filenames:
		losses.append(np.load(f'{indir_path}/{filename}'))
	return losses


def save_predictions(spn_man_rec_res_pres_resss_bss, save_path):
	(models_spn,
	 models_man,
	 models_rec,
	 models_res,
	 models_pres,
	 models_resss,
	 models_bss) = spn_man_rec_res_pres_resss_bss

	os.makedirs(save_path, exist_ok=True)

	for i in range(5):
		os.makedirs(os.path.join(save_path, f's{i}'), exist_ok=True)
		if models_spn is not None:
			models_spn[i].to_parquet(f'{save_path}/s{i}/score_per_neuron.parquet')
		if models_man is not None:
			models_man[i].to_parquet(f'{save_path}/s{i}/timeseries_manifold.parquet')
		if models_rec is not None:
			models_rec[i].to_parquet(f'{save_path}/s{i}/timeseries_reconstruction.parquet')
		if models_res is not None:
			models_res[i].to_parquet(f'{save_path}/s{i}/timeseries_residual.parquet')
		if models_pres is not None:
			models_pres[i].to_parquet(f'{save_path}/s{i}/timeseries_residual_predicted.parquet')
		if models_resss is not None:
			models_resss[i].to_parquet(f'{save_path}/s{i}/timeseries_behavioral_subspace.parquet')
		if models_bss is not None:
			models_bss[i].to_parquet(f'{save_path}/s{i}/timeseries_residual_subspace.parquet')


def load_predictions(load_path):
	[models_spn,
	 models_man,
	 models_rec,
	 models_res,
	 models_pres,
	 models_resss,
	 models_bss] = [list() for _ in range(7)]

	if not os.path.exists(f'{load_path}'):
		raise FileNotFoundError(f'{load_path} not found')

	for i in range(5):
		if os.path.exists(f'{load_path}/s{i}/score_per_neuron.parquet'):
			models_spn.append(pd.read_parquet(f'{load_path}/s{i}/score_per_neuron.parquet'))
		if os.path.exists(f'{load_path}/s{i}/timeseries_manifold.parquet'):
			models_man.append(pd.read_parquet(f'{load_path}/s{i}/timeseries_manifold.parquet'))
		if os.path.exists(f'{load_path}/s{i}/timeseries_reconstruction.parquet'):
			models_rec.append(pd.read_parquet(f'{load_path}/s{i}/timeseries_reconstruction.parquet'))
		if os.path.exists(f'{load_path}/s{i}/timeseries_residual.parquet'):
			models_res.append(pd.read_parquet(f'{load_path}/s{i}/timeseries_residual.parquet'))
		if os.path.exists(f'{load_path}/s{i}/timeseries_residual_predicted.parquet'):
			models_pres.append(pd.read_parquet(f'{load_path}/s{i}/timeseries_residual_predicted.parquet'))
		if os.path.exists(f'{load_path}/s{i}/timeseries_behavioral_subspace.parquet'):
			models_resss.append(pd.read_parquet(f'{load_path}/s{i}/timeseries_behavioral_subspace.parquet'))
		if os.path.exists(f'{load_path}/s{i}/timeseries_residual_subspace.parquet'):
			models_bss.append(pd.read_parquet(f'{load_path}/s{i}/timeseries_residual_subspace.parquet'))

	for _list in [models_spn, models_man, models_rec, models_res, models_pres, models_resss, models_bss]:
		if len(_list) == 0:
			_list = None

	return models_spn, models_man, models_rec, models_res, models_pres, models_resss, models_bss

#endregion

