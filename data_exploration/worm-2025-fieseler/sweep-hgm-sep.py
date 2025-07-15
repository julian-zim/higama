import os
import math
from functools import partial

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

import jax
import jax.numpy as jnp
import optax
import wandb

import read_utils
import plot_utils
import train_utils
import artificializer

from models import clm_sep as clm
from models import hgm_sep as hgm



def train_numerical_model(project_name, model, splits, X_bll, Y, X_man=None):
	with wandb.init(project=project_name):
		config = wandb.config

		is_hgm = model.get_type()

		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

		n_neurons = Y.shape[1]
		n_behaviors_ll = X_bll.shape[1]
		if is_hgm: n_dimensions = X_man.shape[1]

		# Hyperparamters ============
		hyperparams = {
			'max_epochs': config.max_epochs,
			'min_epochs': 100,
			'max_patience': 20,
			'learning_rate': config.learning_rate,
			'reg_l1': config.reg_l1,
			'reg_l2': config.reg_l2,
		}

		max_epochs = hyperparams['max_epochs']
		min_epochs = hyperparams['min_epochs']
		max_patience = hyperparams['max_patience']

		optimizer = optax.adam(hyperparams['learning_rate'])

		reg_lambdas = hyperparams['reg_l1'], hyperparams['reg_l2']
		# ===========================

		# training loop =======================================
		params_per_split = list()
		neuron_scores_per_split = list()
		losses_per_split = list()
		losses_tr_per_split = list()
		mse_losses_per_split = list()
		mse_losses_tr_per_split = list()
		__grads_per_split = list()
		for train, test in splits:
			# splitting and initializing ===============================================================
			X_train, X_test = X_bll[train], X_bll[test]
			if is_hgm: X_man_train, X_man_test = X_man[train], X_man[test]
			Y_train, Y_test = Y[train], Y[test]

			if is_hgm:
				params = model.init_params(seeds, (n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))
			else:
				params = model.init_params(seeds, (n_behaviors_ll, n_neurons))
			opt_state = optimizer.init(params)
			# ==========================================================================================

			if is_hgm:
				losses = [model.loss_fn(params, X_test, X_man_test, Y_test, reg_lambdas).item()]
				losses_tr = [model.loss_fn(params, X_train, X_man_train, Y_train, reg_lambdas).item()]
				mse_losses = [model.mse_loss_fn(params, X_test, X_man_test, Y_test).item()]
				mse_losses_tr = [model.mse_loss_fn(params, X_train, X_man_train, Y_train).item()]
			else:
				losses = [model.loss_fn(params, X_test, Y_test, reg_lambdas).item()]
				losses_tr = [model.loss_fn(params, X_train, Y_train, reg_lambdas).item()]
				mse_losses = [model.mse_loss_fn(params, X_test, Y_test).item()]
				mse_losses_tr = [model.mse_loss_fn(params, X_train, Y_train).item()]
			__grads = list()

			best_params = params
			best_loss = 1E3
			best_loss_tr = 1E3
			best_mse_loss = 1E3
			best_mse_loss_tr = 1E3

			patience = 0
			for epoch in range(max_epochs):
				if is_hgm:
					params, opt_state, __grad = model.update(optimizer.update, opt_state, params, X_train, X_man_train, Y_train, reg_lambdas)
					loss = model.loss_fn(params, X_test, X_man_test, Y_test, reg_lambdas).item()
					loss_tr = model.loss_fn(params, X_train, X_man_train, Y_train, reg_lambdas).item()
					mse_loss = model.mse_loss_fn(params, X_test, X_man_test, Y_test).item()
					mse_loss_tr = model.mse_loss_fn(params, X_train, X_man_train, Y_train).item()
				else:
					params, opt_state, __grad = model.update(optimizer.update, opt_state, params, X_train, Y_train, reg_lambdas)
					loss = model.loss_fn(params, X_test, Y_test, reg_lambdas).item()
					loss_tr = model.loss_fn(params, X_train, Y_train, reg_lambdas).item()
					mse_loss = model.mse_loss_fn(params, X_test, Y_test).item()
					mse_loss_tr = model.mse_loss_fn(params, X_train, Y_train).item()

				__grads.append(__grad)

				# early stopping ================ (inactive)
				if epoch > min_epochs:
					if loss < best_loss:
						losses.append(loss)
						losses_tr.append(loss_tr)
						best_loss = loss
						best_loss_tr = loss_tr
						best_mse_loss = mse_loss
						best_mse_loss_tr = mse_loss_tr
						best_params = params
						patience = 0
					else:
						patience += 1
						if patience > max_patience:
							losses = losses + [best_loss for _ in range(max_epochs - len(losses) + 1)]
							losses_tr = losses_tr + [best_loss_tr for _ in range(max_epochs - len(losses_tr) + 1)]
							mse_losses = mse_losses + [best_mse_loss for _ in range(max_epochs - len(mse_losses) + 1)]
							mse_losses_tr = mse_losses_tr + [best_mse_loss_tr for _ in range(max_epochs - len(mse_losses_tr) + 1)]
							break
						losses.append(loss)
						losses_tr.append(loss_tr)
				# ===============================
				else:
					best_params = params
					losses.append(loss)
					losses_tr.append(loss_tr)
					mse_losses.append(mse_loss)
					mse_losses_tr.append(mse_loss_tr)

			params_per_split.append(best_params)
			losses_per_split.append(losses)
			losses_tr_per_split.append(losses_tr)
			mse_losses_per_split.append(mse_losses)
			mse_losses_tr_per_split.append(mse_losses_tr)

			__grads_per_split.append(__grads)

			# predicting and scoring ========================================================================
			if is_hgm:
				Y_pred = model.predict(X_test, X_man_test, best_params['W1'], best_params['W2'])
			else:
				Y_pred = model.predict(X_test, best_params['W'])

			score_per_neuron = [r2_score(trace, trace_pred) for trace, trace_pred in zip(Y_test.T, Y_pred.T)]
			neuron_scores_per_split.append(score_per_neuron)
			# ===============================================================================================

		# =====================================================

		wandb.log({'score': np.mean(np.array(neuron_scores_per_split)), 'loss': np.mean(np.array(mse_losses_per_split)), 'reg_loss': np.mean(np.array(losses_per_split))})

	return params_per_split, np.array(neuron_scores_per_split), np.array(losses_per_split), np.array(losses_tr_per_split), np.array(mse_losses_per_split), np.array(mse_losses_tr_per_split), __grads_per_split



def read():
	# read
	print('reading...')

	path = os.path.join(read_utils.get_data_path(), 'exported_data_')
	filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read_utils.get_data(path, 9)

	for i in range(len(filenames)):
		mask = eigenworms[i].notna().all(axis=1)
		#continuous_behaviors[i] = continuous_behaviors[i][mask]
		discrete_states[i] = discrete_states[i][mask]
		eigenworms[i] = eigenworms[i][mask]
		traces[i] = traces[i][mask]

		#kymographs[i] = kymographs[i][mask]
		#filenames[i] = filenames[i][mask]

	return filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces



def manifold(traces, discrete_states):
	# manifold
	print('manifolding...')
	selection = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']

	(modes_manifold,
	 traces_manifold,
	 cca_loadings_man,
	 vars_expl_man, vars_expl_man_rs) = train_utils.get_manifolds(traces,
																  [ds[selection] for ds in discrete_states],
																  n_cc=2, n_pc=5)
	traces_residual = [pd.DataFrame(data=trace.values - trace_manifold.values, columns=trace.columns) for trace, trace_manifold in zip(traces, traces_manifold)]

	return modes_manifold, traces_residual, cca_loadings_man, vars_expl_man, vars_expl_man_rs



def sweep(config):
	filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read()
	modes_manifold, traces_residual, cca_loadings_man, vars_expl_man, vars_expl_man_rs = manifold(traces, discrete_states)


	# sweep
	print('sweeping...')

	wandb.login()

	for i in range(len(filenames)):
		project_name = f'hgm_sep_{i}_{filenames[i]}'
		print(f'{i + 1}/{len(filenames)}: {filenames[i]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5).split(np.zeros(eigenworms[i].values.shape[0]), discrete_states[i]['REV'].values))

		def train_model_wrapper():
			return train_numerical_model(project_name, hgm, splits, jnp.array(eigenworms[i].values), jnp.array(traces_residual[i].values), jnp.array(modes_manifold[i].values))

		sweep_id = wandb.sweep(config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)



sweep_config = {
	'method': 'grid',
	'parameters': {
		'max_epochs': {'values': [1000, 5000]},
		'learning_rate': {'values': [1, 1E-1, 1E-2, 1E-3, 1E-4]},
		'reg_l1': {'values': [1, 1E-1, 1E-2, 1E-3, 1E-4]},
		'reg_l2': {'values': [1, 1E-1, 1E-2, 1E-3, 1E-4]}
	}
}
sweep(sweep_config)
