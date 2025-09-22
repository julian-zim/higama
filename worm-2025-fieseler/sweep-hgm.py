#region imports
import sys
import os
import math
import time
from typing import NamedTuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
import jax
import jax.numpy as jnp
import optax
import wandb

sys.path.append(os.path.join(os.getcwd(), 'import'))
from utils import read_utils
# from utils import plot_utils
from utils import artificializer
from models import hgm
#endregion


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float]
	cca_lambdas: tuple[float, float]

class PrepData(NamedTuple):
	N_mean: tuple[float, ...]
	N_std: tuple[float, ...]
	B_hl_mean: tuple[float, ...] | None = None
	B_hl_std: tuple[float, ...] | None = None


def train_model(project_name, model, splits, N, B_ll, B_hl, hyperparameters=None):
	with wandb.init(project=project_name):
		config = wandb.config

		#region hyperparamters ===============================================================
		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3)

		if hyperparameters is None:
			hyperparameters = {
				'max_epochs': config.max_epochs,
				'max_patience': config.max_epochs / 4,
				'learning_rate': config.learning_rate,
				'w1_reg_lambda': config.w1_reg_lambda,
				'w2_reg_lambda': config.w2_reg_lambda,
				'ccadiag_lambda': 0.6,
				'ccaunit_lambda': 1. - 0.6,
				'loss1_lambda': 0.5,
				'loss2_lambda': 1. - 0.5,
				'dimensions': 2
			}

		max_epochs = hyperparameters['max_epochs']
		max_patience = hyperparameters['max_patience']

		learning_rate = hyperparameters['learning_rate']

		n_dimensions = hyperparameters['dimensions']

		lambdas = Lambdas(
			loss_lambdas=(hyperparameters['loss1_lambda'], hyperparameters['loss2_lambda']),
			reg_lambdas=(hyperparameters['w1_reg_lambda'], hyperparameters['w2_reg_lambda']),
			cca_lambdas=(hyperparameters['ccadiag_lambda'], hyperparameters['ccaunit_lambda'])
		)
		#endregion ===========================================================================

		n_neurons = N.shape[1]
		n_behaviors_hl = B_hl.shape[1]
		n_behaviors_ll = B_ll.shape[1]

		# training loop =======================================
		params_per_split = list()
		neuron_scores_per_split = list()
		losses_per_split = list()
		losses_tr_per_split = list()
		mpe_losses_per_split = list()
		mpe_losses_tr_per_split = list()
		for s_id, (train, eval) in enumerate(splits):
			#region splitting and initializing (this also moves the data to the GPU) ==========================
			N_train, N_eval = jnp.array(N[train]), jnp.array(N[eval])
			B_hl_train, B_hl_eval = jnp.array(B_hl[train]), jnp.array(B_hl[eval])
			B_ll_train, B_ll_eval = jnp.array(B_ll[train]), jnp.array(B_ll[eval])

			prep_data_train = PrepData(
				N_mean=tuple((map(float, N_train.mean(axis=0)))),
				N_std=tuple((map(float, jnp.where((N_std := jnp.std(N_train, axis=0)) == 0.0, 1.0, N_std)))),
				B_hl_mean=tuple((map(float, B_hl_train.mean(axis=0)))),
				B_hl_std=tuple((map(float, jnp.where((B_hl_std := jnp.std(B_hl_train, axis=0)) == 0.0, 1.0, B_hl_std))))
			)
			prep_data_eval = PrepData(
				N_mean=tuple((map(float, N_eval.mean(axis=0)))),
				N_std=tuple((map(float, jnp.where((N_std := jnp.std(N_eval, axis=0)) == 0.0, 1.0, N_std)))),
				B_hl_mean=tuple((map(float, B_hl_eval.mean(axis=0)))),
				B_hl_std=tuple((map(float, jnp.where((B_hl_std := jnp.std(B_hl_eval, axis=0)) == 0.0, 1.0, B_hl_std))))
			)

			params = model.init_params(seeds,
									   (n_neurons, n_dimensions),
									   (n_behaviors_hl, n_dimensions),
									   (n_behaviors_ll, n_neurons),
									   * ((n_dimensions, n_neurons),) if model.get_type() else ())

			# optimizer = optax.adam(learning_rate=optax.cosine_decay_schedule(
			# 	init_value=learning_rate,
			# 	decay_steps=max_epochs,
			# 	alpha=1E-2
			# ))
			optimizer = optax.adam(learning_rate)
			opt_state = optimizer.init(params)
			#endregion ========================================================================================

			losses = [model.loss_fn(params, N_eval, B_ll_eval, B_hl_eval, lambdas, prep_data_eval).item()]
			losses_tr = [model.loss_fn(params, N_train, B_ll_train, B_hl_train, lambdas, prep_data_train).item()]
			mpe_losses = [model.mpe_loss_fn(params, N_eval, B_ll_eval, prep_data_eval).item()]
			mpe_losses_tr = [model.mpe_loss_fn(params, N_train, B_ll_train, prep_data_train).item()]

			best_params = params
			best_loss = sys.float_info.max
			best_loss_tr = sys.float_info.max
			best_mpe_loss = sys.float_info.max
			best_mpe_loss_tr = sys.float_info.max

			# lr_step = 0
			patience = 0
			for epoch in range(max_epochs):
				params, opt_state = model.update(optimizer.update, opt_state, params, N_train, B_ll_train, B_hl_train, lambdas, prep_data_train)

				loss = model.loss_fn(params, N_eval, B_ll_eval, B_hl_eval, lambdas, prep_data_eval).item()
				loss_tr = model.loss_fn(params, N_train, B_ll_train, B_hl_train, lambdas, prep_data_train).item()
				mpe_loss = model.mpe_loss_fn(params, N_eval, B_ll_eval, prep_data_eval).item()
				mpe_loss_tr = model.mpe_loss_fn(params, N_train, B_ll_train, prep_data_train).item()

				#region early stopping ===========================================================================
				if epoch >= max_patience:  # give it an initial time
					if loss < best_loss:
						patience = 0

						best_params = params
						best_loss = loss
						best_loss_tr = loss_tr
						best_mpe_loss = mpe_loss
						best_mpe_loss_tr = mpe_loss_tr
					else:
						patience += 1
						if patience > max_patience:
							params = best_params
							losses = losses + [best_loss for _ in range(max_epochs - epoch)]
							losses_tr = losses_tr + [best_loss_tr for _ in range(max_epochs - epoch)]
							mpe_losses = mpe_losses + [best_mpe_loss for _ in range(max_epochs - epoch)]
							mpe_losses_tr = mpe_losses_tr + [best_mpe_loss_tr for _ in range(max_epochs - epoch)]

							for rest_epoch in range(epoch + 1, max_epochs):
								wandb.log({f'epoch_s{s_id}': rest_epoch, f'loss_s{s_id}': best_loss, f'loss_tr_s{s_id}': best_loss_tr, f'mpe_loss_s{s_id}': best_mpe_loss, f'mpe_loss_tr_s{s_id}': best_mpe_loss_tr})
							break

							# lr_step += 1
							# optimizer = optax.adam(learning_rate / (10 ** lr_step))
							# patience = 0
							# print(f'\nReduced learning rate from {learning_rate / (10 ** (lr_step - 1))} to {learning_rate / (10 ** lr_step)} after epoch {epoch} in split {s_id}', end='')
				#endregion =======================================================================================

				losses.append(loss)
				losses_tr.append(loss_tr)
				mpe_losses.append(mpe_loss)
				mpe_losses_tr.append(mpe_loss_tr)
				wandb.log({f'epoch_s{s_id}': epoch, f'loss_s{s_id}': loss, f'loss_tr_s{s_id}': loss_tr, f'mpe_loss_s{s_id}': mpe_loss, f'mpe_loss_tr_s{s_id}': mpe_loss_tr})

			params_per_split.append(params)
			losses_per_split.append(losses)
			losses_tr_per_split.append(losses_tr)
			mpe_losses_per_split.append(mpe_losses)
			mpe_losses_tr_per_split.append(mpe_losses_tr)

			# predicting and scoring ========================================================================
			score_per_neuron = model.score_fn(params, N_eval, B_ll_eval, prep_data_eval)
			neuron_scores_per_split.append(score_per_neuron)
			# ===============================================================================================

		# =====================================================

		wandb.log({'score': np.mean(np.mean(np.array(neuron_scores_per_split), axis=1)),
				   'loss': np.mean(np.array(losses_per_split)[:, -1]),
				   'loss_tr': np.mean(np.array(losses_tr_per_split)[:, -1]),
				   'mpe_loss': np.mean(np.array(mpe_losses_per_split)[:, -1]),
				   'mpe_loss_tr': np.mean(np.array(mpe_losses_tr_per_split)[:, -1])})

	return params_per_split, np.array(neuron_scores_per_split), np.array(losses_per_split), np.array(losses_tr_per_split), np.array(mpe_losses_per_split), np.array(mpe_losses_tr_per_split)


def read(artificial=False):
	print('reading...')

	# read_ids = [0, 12, 14, 23, 29, 32]
	read_ids = [14]
	# selection_n = ['DD01', 'VB02', 'SMDDR', 'AVAR']
	selection_bhl = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']

	path = os.path.join(read_utils.get_data_path(), 'worm-2025-fieseler', 'exported_data_gcamp')
	file_ids, filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read_utils.get_data(path, read_ids)

	for i in range(len(filenames)):
		mask = eigenworms[i].notna().all(axis=1)
		#continuous_behaviors[i] = continuous_behaviors[i][mask]
		discrete_states[i] = discrete_states[i][mask]
		eigenworms[i] = eigenworms[i][mask]
		traces[i] = traces[i][mask]

		#kymographs[i] = kymographs[i][mask]
		#filenames[i] = filenames[i][mask]

	discrete_states = [ds[selection_bhl] for ds in discrete_states]


	# artificialize
	if artificial:
		dims_generation = 2
		weights = (0.5, 2.)

		file_ids_art = file_ids.copy()
		filenames_art = list()
		discrete_states_art = discrete_states.copy()
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
		# 					   discrete_states_art[0]['REV'],
		# 					   traces_art[0][selection_n],
		# 					   plots_per_row=2, ylimits=(-8, 8))
		return file_ids_art, filenames_art, continuous_behaviors, discrete_states_art, eigenworms_art, kymographs, traces_art


	# plot_utils.plot_traces(f'[{filenames[0]}] traces',
	# 					   discrete_states[0]['REV'],
	# 					   traces[0][selection_n],
	# 					   plots_per_row=2, ylimits=(-2, 2))
	return file_ids, filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces


def sweep(config):
	file_ids, filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read(artificial=False)

	print('sweeping...')

	wandb.login()

	for _id in file_ids:
		project_name = f'hgm {time.strftime("%Y-%m-%d %Hh%M", time.localtime())} DATASET {_id}-{filenames[file_ids.index(_id)]}'
		print(f'{_id}: {filenames[file_ids.index(_id)]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5).split(np.zeros(traces[file_ids.index(_id)].values.shape[0]), discrete_states[file_ids.index(_id)]['REV'].values))

		def train_model_wrapper():
			return train_model(project_name, hgm, splits,
							   traces[file_ids.index(_id)].values,
							   eigenworms[file_ids.index(_id)].values,
							   discrete_states[file_ids.index(_id)].values)

		sweep_id = wandb.sweep(config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)



print('Visible devices: ', jax.devices())  
sweep_config = {
	'method': 'bayes',
    'metric': {
        'name': 'mpe_loss',
        'goal': 'minimize'
    },
	'parameters': {
		'learning_rate': {'min': 1E-3, 'max': 1},
		'w1_reg_lambda': {'min': 1E-4, 'max': 1E-2},
		'w2_reg_lambda': {'min': 1E-6, 'max': 1E-3},
	}
}
sweep(sweep_config)
