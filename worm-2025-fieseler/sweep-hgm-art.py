import sys
import os
import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import optax
import wandb


sys.path.append(os.path.join(os.getcwd(), 'import'))

from utils import read_utils
from utils import artificializer

from models import hgm



def train_joint_model(project_name, model, splits, N, B_ll, B_hl):
	with wandb.init(project=project_name):
		config = wandb.config

		is_hgm = model.get_type()

		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3)

		n_neurons = N.shape[1]
		n_behaviors_hl = B_hl.shape[1]
		n_behaviors_ll = B_ll.shape[1]

		# Hyperparamters ============
		hyperparams = {
			'max_epochs': config.max_epochs,
			'min_epochs': 100,
			'max_patience': 20,
			'learning_rate': config.learning_rate,
			'reg_l1': config.reg_l1,
			'reg_l2': config.reg_l2,
			'loss_l1': config.loss_l1,
			'loss_l2': 1 - config.loss_l1,
			'dimensions': config.dimensions
		}

		max_epochs = hyperparams['max_epochs']
		min_epochs = hyperparams['min_epochs']
		max_patience = hyperparams['max_patience']

		optimizer = optax.adam(hyperparams['learning_rate'])

		reg_lambdas = hyperparams['reg_l1'], hyperparams['reg_l2']
		loss_lambdas = hyperparams['loss_l1'], hyperparams['loss_l2']

		n_dimensions = hyperparams['dimensions']
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
			N_train, N_test = N[train], N[test]
			B_hl_train, B_hl_test = B_hl[train], B_hl[test]
			B_ll_train, B_ll_test = B_ll[train], B_ll[test]

			if is_hgm:
				params = model.init_params(seeds, (n_neurons, n_dimensions), (n_behaviors_hl, n_dimensions), (n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))
			else:
				params = model.init_params(seeds, (n_neurons, n_dimensions), (n_behaviors_hl, n_dimensions), (n_behaviors_ll, n_neurons))
			opt_state = optimizer.init(params)
			# ==========================================================================================

			if is_hgm:
				losses = [model.loss_fn(params, N_test, B_ll_test, B_hl_test, loss_lambdas, reg_lambdas).item()]
				losses_tr = [model.loss_fn(params, N_train, B_ll_train, B_hl_train, loss_lambdas, reg_lambdas).item()]
				mse_losses = [model.mse_loss_fn(params, N_test, B_ll_test).item()]
				mse_losses_tr = [model.mse_loss_fn(params, N_train, B_ll_train).item()]
			else:
				losses = [model.loss_fn(params, N_test, B_ll_test, B_hl_test, loss_lambdas, reg_lambdas[0]).item()]
				losses_tr = [model.loss_fn(params, N_train, B_ll_train, B_hl_train, loss_lambdas, reg_lambdas[0]).item()]
				mse_losses = [model.mse_loss_fn(params, N_test, B_ll_test).item()]
				mse_losses_tr = [model.mse_loss_fn(params, N_train, B_ll_train).item()]
			__grads = list()

			best_params = params
			best_loss = 1E3
			best_loss_tr = 1E3
			best_mse_loss = 1E3
			best_mse_loss_tr = 1E3

			patience = 0
			for epoch in range(max_epochs):
				if is_hgm:
					params, opt_state, __grad = model.update(optimizer.update, opt_state, params, N, B_ll, B_hl, loss_lambdas, reg_lambdas)
					loss = model.loss_fn(params, N_test, B_ll_test, B_hl_test, loss_lambdas, reg_lambdas).item()
					loss_tr = model.loss_fn(params, N_train, B_ll_train, B_hl_train, loss_lambdas, reg_lambdas).item()
					mse_loss = model.mse_loss_fn(params, N_test, B_ll_test).item()
					mse_loss_tr = model.mse_loss_fn(params, N_train, B_ll_train).item()
				else:
					params, opt_state, __grad = model.update(optimizer.update, opt_state, params, N, B_ll, B_hl, loss_lambdas, reg_lambdas[0])
					loss = model.loss_fn(params, N_test, B_ll_test, B_hl_test, loss_lambdas, reg_lambdas[0]).item()
					loss_tr = model.loss_fn(params, N_train, B_ll_train, B_hl_train, loss_lambdas, reg_lambdas[0]).item()
					mse_loss = model.mse_loss_fn(params, N_test, B_ll_test).item()
					mse_loss_tr = model.mse_loss_fn(params, N_train, B_ll_train).item()

				__grads.append(__grad)

				# early stopping =============
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
				# ============================
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
			N_test_man = N_test @ best_params['U']
			N_test_rec = N_test_man @ jax.numpy.linalg.pinv(best_params['U'])
			N_test_res = N_test - N_test_rec
			if is_hgm:
				Y_pred = model.predict(B_ll_test, N_test_man, best_params['W1'], best_params['W2'])
			else:
				Y_pred = model.predict(B_ll_test, best_params['W'])

			score_per_neuron = [r2_score(trace, trace_pred) for trace, trace_pred in zip(N_test_res.T, Y_pred.T)]
			neuron_scores_per_split.append(score_per_neuron)
			# ===============================================================================================

		# =====================================================

		wandb.log({'score': np.mean(np.array(neuron_scores_per_split)), 'loss': np.mean(np.array(mse_losses_per_split)), 'reg_loss': np.mean(np.array(losses_per_split))})

	return params_per_split, np.array(neuron_scores_per_split), np.array(losses_per_split), np.array(losses_tr_per_split), np.array(mse_losses_per_split), np.array(mse_losses_tr_per_split), __grads_per_split



def read():
	# read
	print('reading...')

	path = os.path.join(read_utils.get_data_path(), 'worm-2025-fieseler', 'exported_data_')
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



def artificialize(filenames, traces, eigenworms, discrete_states):
	# artificialize
	print('artificializing...')

	filenames_art = list()
	discrete_states_art = discrete_states.copy()
	eigenworms_art = eigenworms.copy()
	modes_manifold_art = list()
	traces_residual_art = list()
	traces_art = list()
	for i in range(len(filenames)):
		art_filename, art_mode_manifold, art_trace_residual, art_trace, art_params = artificializer.artificialize(hgm, filenames[i], traces[i].columns, eigenworms_art[i], discrete_states_art[i])
		filenames_art.append(art_filename)
		modes_manifold_art.append(art_mode_manifold)
		traces_residual_art.append(art_trace_residual)
		traces_art.append(art_trace)

	return filenames_art, discrete_states_art, eigenworms_art, modes_manifold_art, traces_residual_art, traces_art



def sweep(config):
	filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read()
	filenames_art, discrete_states_art, eigenworms_art, modes_manifold_art, traces_residual_art, traces_art = artificialize(filenames, traces, eigenworms, discrete_states)


	# sweep
	print('sweeping...')

	wandb.login()

	for i in range(len(filenames_art)):
		project_name = f'hgm_art_{i}_{filenames_art[i]}'
		print(f'{i + 1}/{len(filenames_art)}: {filenames_art[i]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5).split(np.zeros(traces_art[i].values.shape[0]), discrete_states_art[i]['REV'].values))

		def train_model_wrapper():
			return train_joint_model(project_name, hgm, splits, jnp.array(traces_art[i].values), jnp.array(eigenworms_art[i].values), jnp.array(discrete_states_art[i].values))

		sweep_id = wandb.sweep(config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)



sweep_config = {
	'method': 'grid',
	'parameters': {
		'max_epochs': {'values': [1000, 5000, 10000]},
		'learning_rate': {'values': [1, 1E-1, 1E-2, 1E-3, 1E-4]},
		'reg_l1': {'values': [1, 1E-1, 1E-2, 1E-3, 1E-4]},
		'reg_l2': {'values': [1, 1E-1, 1E-2, 1E-3, 1E-4]},
		'loss_l1': {'values': [0.8, 0.75, 0.67, 0.5, 0.33, 0.25, 0.2]},
		'dimensions': {'values': [1, 2, 3]},
	}
}
sweep(sweep_config)
