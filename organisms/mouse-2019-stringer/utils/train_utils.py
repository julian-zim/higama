import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from flax.serialization import to_bytes, from_bytes


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
