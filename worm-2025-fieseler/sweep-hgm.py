import sys
import os
import math
import time
from typing import NamedTuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
import optax
import wandb

sys.path.append(os.path.join(os.getcwd(), 'import'))
from utils import read_utils
from utils import plot_utils
from utils import artificializer
from models import hgm


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float]
	cca_lambdas: tuple[float, float]

class PrepData(NamedTuple):
	N_mean: tuple[float, ...]
	N_std: tuple[float, ...]
	B_hl_mean: tuple[float, ...] | None = None
	B_hl_std: tuple[float, ...] | None = None


def train_joint_model(project_name, model, splits, N, B_ll, B_hl, hyperparameters=None):
	with wandb.init(project=project_name):
		config = wandb.config

		# Hyperparamters ============
		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3)

		if hyperparameters is None:
			hyperparameters = {
				'max_epochs': 5000,
				'max_patience': 100,
				'learning_rate': config.learning_rate,  # 0.001,
				'w1_reg_lambda': config.w1_reg_lambda,  # 1E-3,
				'w2_reg_lambda': config.w2_reg_lambda,  # 1E-5,
				'ccadiag_lambda': config.ccadiag_lambda,  # 0.5,
				'ccaunit_lambda': config.ccaunit_lambda,  # 0.5,
				'loss1_lambda': config.loss1_lambda,  # 0.5,
				'loss2_lambda': 1. - config.loss1_lambda,  # 0.5,
				'dimensions': 1  # 3
			}

		max_epochs = hyperparameters['max_epochs']
		max_patience = hyperparameters['max_patience']
		optimizer = optax.adam(hyperparameters['learning_rate'])
		n_dimensions = hyperparameters['dimensions']

		lambdas = Lambdas(
			loss_lambdas=(hyperparameters['loss1_lambda'], hyperparameters['loss2_lambda']),
			reg_lambdas=(hyperparameters['w1_reg_lambda'], hyperparameters['w2_reg_lambda']),
			cca_lambdas=(hyperparameters['ccaunit_lambda'], hyperparameters['ccadiag_lambda'])
		)
		# ===========================

		n_neurons = N.shape[1]
		n_behaviors_hl = B_hl.shape[1]
		n_behaviors_ll = B_ll.shape[1]

		# training loop =======================================
		params_per_split = list()
		neuron_scores_per_split = list()
		losses_per_split = list()
		losses_tr_per_split = list()
		mse_losses_per_split = list()
		mse_losses_tr_per_split = list()
		__grads_per_split = list()
		for sid, (train, test) in enumerate(splits):
			# splitting and initializing ========================== (this also moves the data to the GPU)
			N_train, N_test = jnp.array(N[train]), jnp.array(N[test])
			B_hl_train, B_hl_test = jnp.array(B_hl[train]), jnp.array(B_hl[test])
			B_ll_train, B_ll_test = jnp.array(B_ll[train]), jnp.array(B_ll[test])

			prep_data_train = PrepData(
				N_mean=tuple((map(float, N_train.mean(axis=0)))),
				N_std=tuple((map(float, jnp.where((N_std := jnp.std(N_train, axis=0)) == 0.0, 1.0, N_std)))),
				B_hl_mean=tuple((map(float, B_hl_train.mean(axis=0)))),
				B_hl_std=tuple((map(float, jnp.where((B_hl_std := jnp.std(B_hl_train, axis=0)) == 0.0, 1.0, B_hl_std))))
			)
			prep_data_test = PrepData(
				N_mean=tuple((map(float, N_test.mean(axis=0)))),
				N_std=tuple((map(float, jnp.where((N_std := jnp.std(N_test, axis=0)) == 0.0, 1.0, N_std)))),
				B_hl_mean=tuple((map(float, B_hl_test.mean(axis=0)))),
				B_hl_std=tuple((map(float, jnp.where((B_hl_std := jnp.std(B_hl_test, axis=0)) == 0.0, 1.0, B_hl_std))))
			)

			params = model.init_params(seeds,
									   (n_neurons, n_dimensions),
									   (n_behaviors_hl, n_dimensions),
									   (n_behaviors_ll, n_neurons),
									   *((n_dimensions, n_neurons),) if model.get_type() else ())
			opt_state = optimizer.init(params)
			# ===========================================================================================

			losses = [model.loss_fn(params, N_test, B_ll_test, B_hl_test, lambdas, prep_data_test).item()]
			losses_tr = [model.loss_fn(params, N_train, B_ll_train, B_hl_train, lambdas, prep_data_train).item()]
			mse_losses = [model.mse_loss_fn(params, N_test, B_ll_test, prep_data_test).item()]
			mse_losses_tr = [model.mse_loss_fn(params, N_train, B_ll_train, prep_data_train).item()]
			__grads = list()

			best_params = params
			best_loss = sys.float_info.max
			best_loss_tr = sys.float_info.max
			best_mse_loss = sys.float_info.max
			best_mse_loss_tr = sys.float_info.max

			patience = 0
			for epoch in range(max_epochs):
				params, opt_state, __grad = model.update(optimizer.update, opt_state, params, N_train, B_ll_train, B_hl_train, lambdas, prep_data_train)

				loss = model.loss_fn(params, N_test, B_ll_test, B_hl_test, lambdas, prep_data_test).item()
				loss_tr = model.loss_fn(params, N_train, B_ll_train, B_hl_train, lambdas, prep_data_train).item()
				mse_loss = model.mse_loss_fn(params, N_test, B_ll_test, prep_data_test).item()
				mse_loss_tr = model.mse_loss_fn(params, N_train, B_ll_train, prep_data_train).item()
				__grads.append(__grad)

				# early stopping ================
				if loss < best_loss:
					patience = 0

					best_params = params
					best_loss = loss
					best_loss_tr = loss_tr
					best_mse_loss = mse_loss
					best_mse_loss_tr = mse_loss_tr
				else:
					patience += 1
					if patience > max_patience:
						params = best_params
						losses = losses + [best_loss for _ in range(max_epochs - epoch)]
						losses_tr = losses_tr + [best_loss_tr for _ in range(max_epochs - epoch)]
						mse_losses = mse_losses + [best_mse_loss for _ in range(max_epochs - epoch)]
						mse_losses_tr = mse_losses_tr + [best_mse_loss_tr for _ in range(max_epochs - epoch)]
						for rest_epoch in range(epoch + 1, max_epochs):
							wandb.log({f'epoch_s{sid}': rest_epoch, f'loss_s{sid}': best_loss, f'loss_tr_s{sid}': best_loss_tr, f'mse_loss_s{sid}': best_mse_loss, f'mse_loss_tr_s{sid}': best_mse_loss_tr})
						break

				losses.append(loss)
				losses_tr.append(loss_tr)
				mse_losses.append(mse_loss)
				mse_losses_tr.append(mse_loss_tr)
				wandb.log({f'epoch_s{sid}': epoch, f'loss_s{sid}': loss, f'loss_tr_s{sid}': loss_tr, f'mse_loss_s{sid}': mse_loss, f'mse_loss_tr_s{sid}': mse_loss_tr})
			# ===============================

			params_per_split.append(params)
			losses_per_split.append(losses)
			losses_tr_per_split.append(losses_tr)
			mse_losses_per_split.append(mse_losses)
			mse_losses_tr_per_split.append(mse_losses_tr)
			__grads_per_split.append(__grads)

			# predicting and scoring ========================================================================
			score_per_neuron = model.score_fn(params, N_test, B_ll_test, prep_data_test)
			neuron_scores_per_split.append(score_per_neuron)
		# ===============================================================================================

		# =====================================================

		wandb.log({'score': np.mean(np.mean(np.array(neuron_scores_per_split), axis=1)),
				   'loss': np.mean(np.array(losses_per_split)[:, -1]),
				   'loss_tr': np.mean(np.array(losses_tr_per_split)[:, -1]),
				   'mse_loss': np.mean(np.array(mse_losses_per_split)[:, -1]),
				   'mse_loss_tr': np.mean(np.array(mse_losses_tr_per_split)[:, -1])})

	return params_per_split, np.array(neuron_scores_per_split), np.array(losses_per_split), np.array(losses_tr_per_split), np.array(mse_losses_per_split), np.array(mse_losses_tr_per_split), __grads_per_split


def read(artificial=False):
	# read
	print('reading...')

	path = os.path.join(read_utils.get_data_path(), 'worm-2025-fieseler', 'exported_data_')
	filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read_utils.get_data(path, slice=(24, 24 + 1))

	for i in range(len(filenames)):
		mask = eigenworms[i].notna().all(axis=1)
		#continuous_behaviors[i] = continuous_behaviors[i][mask]
		discrete_states[i] = discrete_states[i][mask]
		eigenworms[i] = eigenworms[i][mask]
		traces[i] = traces[i][mask]

		#kymographs[i] = kymographs[i][mask]
		#filenames[i] = filenames[i][mask]


	# artificialize
	if artificial:
		selection_bhl = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']
		dims_generation = 1
		weights = (0.5, 2.)

		filenames_art = list()
		discrete_states_art = [ds[selection_bhl] for ds in discrete_states.copy()]
		eigenworms_art = eigenworms.copy()
		neural_modes_art = list()
		traces_reconstructed_art = list()
		traces_residual_art = list()
		traces_art = list()
		for i in range(len(filenames)):
			(filename_art,
			 neural_modes_art_,
			 traces_reconstructed_art_,
			 traces_residual_art_,
			 traces_art_,
			 params_art_,
			 params_cca_art_) = artificializer.artificialize(weights, hgm,
															 filenames[i],
															 traces[i].columns,
															 eigenworms_art[i],
															 discrete_states_art[i],
															 n_dimensions=dims_generation)
			filenames_art.append(filename_art)
			neural_modes_art.append(neural_modes_art_)
			traces_reconstructed_art.append(traces_reconstructed_art_)
			traces_residual_art.append(traces_residual_art_)
			traces_art.append(traces_art_)

		# plot_utils.plot_traces(f'[{filenames_art[0]}] traces artificial',
		# 					   traces_art[0][['AVAR', 'ALA', 'VB02', 'IL2L', 'SMDDR', 'RID']],
		# 					   discrete_states_art[0]['REV'],
		# 					   n_plots=None, n_columns=2, ylimits=(-8, 8))
		return filenames_art, continuous_behaviors, discrete_states_art, eigenworms_art, kymographs, traces_art


	# plot_utils.plot_traces(f'[{filenames_art[0]}] traces artificial',
	# 					   traces[0][['AVAR', 'ALA', 'VB02', 'IL2L', 'SMDDR', 'RID']],
	# 					   discrete_states[0]['REV'],
	# 					   n_plots=None, n_columns=2, ylimits=(-8, 8))
	return filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces



def sweep(config):
	filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read(artificial=True)

	print('sweeping...')

	wandb.login()

	for i in range(len(filenames)):
		project_name = f'hgm {time.strftime("%Y-%m-%d %Hh%M", time.localtime())} DATASET {i}-{filenames[i]}'
		print(f'{i + 1}/{len(filenames)}: {filenames[i]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5).split(np.zeros(traces[i].values.shape[0]), discrete_states[i]['REV'].values))

		def train_model_wrapper():
			return train_joint_model(project_name, hgm, splits, traces[i].values, eigenworms[i].values, discrete_states[i].values)

		sweep_id = wandb.sweep(config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)



print('Visible devices: ', jax.devices())  
sweep_config = {
	'method': 'bayes',
    'metric': {
        'name': 'mse_loss',
        'goal': 'minimize'
    },
	'parameters': {
		'learning_rate': {'min': 1E-4, 'max': 1.},
		'w1_reg_lambda': {'min': 0., 'max': 1E-1},
		'w2_reg_lambda': {'min': 0., 'max': 1E-1},
		'ccadiag_lambda': {'min': 1E-1, 'max': 1.},
		'ccaunit_lambda': {'min': 1E-1, 'max': 1.},
		'loss1_lambda': {'values': [0.8, 0.75, 0.67, 0.5]},
	}
}
sweep(sweep_config)

