#region imports
import sys
import os
import time
from typing import NamedTuple
import math
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import jax
import jax.numpy as jnp
import optax
import wandb


sys.path.append(os.path.join(os.getcwd(), '../'))
from utils_worm import artificializer
from utils_worm import read_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from models import hgm_rs as hgm_gen
#endregion


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float, float]
	cca_lambdas: tuple[float, float]


def train_model(model, splits, N, B, S, default_hyperparameters, log_path):
	with wandb.init(dir=log_path):
		config = wandb.config

		#region hyperparamters ===============================================================
		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3), jax.random.PRNGKey(4)

		hyperparameters = default_hyperparameters.copy()
		for key in config:
			hyperparameters[key] = config[key]

		max_epochs = hyperparameters['max_epochs']
		max_patience = hyperparameters['max_patience']

		learning_rate = hyperparameters['learning_rate']

		n_dimensions = hyperparameters['dimensions']

		lambdas = Lambdas(
			loss_lambdas=(hyperparameters['loss_man_lambda'], hyperparameters['loss_res_lambda']),
			reg_lambdas=(hyperparameters['vb_reg_lambda'], hyperparameters['wb_reg_lambda'], hyperparameters['wm_reg_lambda']),
			cca_lambdas=(hyperparameters['ccadiag_lambda'], hyperparameters['ccaunit_lambda'])
		)
		#endregion ===========================================================================

		n_neurons = N.shape[1]
		n_states = S.shape[1]
		n_behaviors = B.shape[1]

		# training loop ==============================
		params_per_split = list()
		(losses_train_per_split,
		 man_losses_train_per_split,
		 res_losses_train_per_split,
		 mpe_losses_train_per_split) = list(), list(), list(), list()
		(losses_per_split,
		 man_losses_per_split,
		 res_losses_per_split,
		 mpe_losses_per_split) = list(), list(), list(), list()
		for s_id, (train, eval) in enumerate(splits):
			#region splitting and initializing (this also moves the data to the GPU) ==========================
			N_train, N_eval = jnp.array(N[train]), jnp.array(N[eval])
			S_train, S_eval = jnp.array(S[train]), jnp.array(S[eval])
			B_train, B_eval = jnp.array(B[train]), jnp.array(B[eval])

			params = model.init_params(N_train, B_train, S_train,
									   seeds,
									   (n_neurons, n_dimensions),
									   (n_states, n_dimensions),
									   (n_behaviors, n_neurons),
									   (n_behaviors, n_behaviors),
									   *((n_dimensions, n_behaviors),) if model.get_type() else (),
									   )

			optimizer = optax.adam(learning_rate=optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max_epochs, alpha=1E-4))
			# optimizer = optax.adam(learning_rate)
			optimizer = optax.masked(optimizer, {'prep': False, 'model': True})
			opt_state = optimizer.init(params)
			#endregion ========================================================================================

			(losses_train,
			 man_losses_train,
			 res_losses_train,
			 mpe_losses_train) = [], [], [], []
			(losses,
			 man_losses,
			 res_losses,
			 mpe_losses) = [], [], [], []

			best_params = params
			(best_loss_train,
			 best_man_loss_train,
			 best_res_loss_train,
			 best_mpe_loss_train) = sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max
			(best_loss,
			 best_man_loss,
			 best_res_loss,
			 best_mpe_loss) = sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max

			# lr_step = 0
			patience = 0
			for epoch in range(max_epochs):

				(loss,
				 man_loss,
				 res_loss,
				 mpe_loss) = tuple(l.item() for l in model.loss_fn_flattened(params, N_eval, B_eval, S_eval, lambdas))

				(opt_state, params, loss_types_train) = model.update(optimizer.update, opt_state, params, N_train, B_train, S_train, lambdas)

				(loss_train,
				 man_loss_train,
				 res_loss_train,
				 mpe_loss_train) = tuple(l.item() for l in loss_types_train)

				#region early stopping ===========================================================================
				if epoch >= max_patience:  # give it an initial time
					if loss < best_loss:
						patience = 0

						best_params = params
						(best_loss_train,
						 best_man_loss_train,
						 best_res_loss_train,
						 best_mpe_loss_train) = loss_train, man_loss_train, res_loss_train, mpe_loss_train
						(best_loss,
						 best_man_loss,
						 best_res_loss,
						 best_mpe_loss) = loss, man_loss, res_loss, mpe_loss
					else:
						patience += 1
						if patience > max_patience:
							params = best_params

							losses_train = losses_train + [best_loss_train for _ in range(max_epochs - epoch)]
							man_losses_train = man_losses_train + [best_man_loss_train for _ in range(max_epochs - epoch)]
							res_losses_train = res_losses_train + [best_res_loss_train for _ in range(max_epochs - epoch)]
							mpe_losses_train = mpe_losses_train + [best_mpe_loss_train for _ in range(max_epochs - epoch)]

							losses = losses + [best_loss for _ in range(max_epochs - epoch)]
							man_losses = man_losses + [best_man_loss for _ in range(max_epochs - epoch)]
							res_losses = res_losses + [best_res_loss for _ in range(max_epochs - epoch)]
							mpe_losses = mpe_losses + [best_mpe_loss for _ in range(max_epochs - epoch)]

							for rest_epoch in range(epoch + 1, max_epochs):
								wandb.log({f'epoch_s{s_id}': rest_epoch,
										   f'loss_train_s{s_id}': best_loss_train,
										   f'man_loss_train_s{s_id}': best_man_loss_train,
										   f'res_loss_train_s{s_id}': best_res_loss_train,
										   f'mpe_loss_train_s{s_id}': best_mpe_loss_train,
										   f'loss_s{s_id}': best_loss,
										   f'man_loss_s{s_id}': best_man_loss,
										   f'res_loss_s{s_id}': best_res_loss,
										   f'mpe_loss_s{s_id}': best_mpe_loss})
							break

							# lr_step += 1
							# optimizer = optax.adam(learning_rate / (10 ** lr_step))
							# optimizer = optax.masked(optimizer, {'prep': False, 'model': True})
							# patience = 0
							# print(f'\nReduced learning rate from {learning_rate / (10 ** (lr_step - 1))} to {learning_rate / (10 ** lr_step)} after epoch {epoch} in split {s_id}', end='')
				#endregion =======================================================================================

				losses_train.append(loss_train)
				man_losses_train.append(man_loss_train)
				res_losses_train.append(res_loss_train)
				mpe_losses_train.append(mpe_loss_train)

				losses.append(loss)
				man_losses.append(man_loss)
				res_losses.append(res_loss)
				mpe_losses.append(mpe_loss)

				wandb.log({f'epoch_s{s_id}': epoch,
						   f'loss_train_s{s_id}': loss_train,
						   f'man_loss_train_s{s_id}': man_loss_train,
						   f'res_loss_train_s{s_id}': res_loss_train,
						   f'mpe_loss_train_s{s_id}': mpe_loss_train,
						   f'loss_s{s_id}': loss,
						   f'man_loss_s{s_id}': man_loss,
						   f'res_loss_s{s_id}': res_loss,
						   f'mpe_loss_s{s_id}': mpe_loss})


			params_per_split.append(params)

			losses_train_per_split.append(losses_train)
			man_losses_train_per_split.append(man_losses_train)
			res_losses_train_per_split.append(res_losses_train)
			mpe_losses_train_per_split.append(mpe_losses_train)

			losses_per_split.append(losses)
			man_losses_per_split.append(man_losses)
			res_losses_per_split.append(res_losses)
			mpe_losses_per_split.append(mpe_losses)

			print(f'Split {s_id} completed. ', end='')

		wandb.log({'loss_train': np.mean(np.array(losses_train_per_split)[:, -1]),
				   'man_loss_train': np.mean(np.array(man_losses_train_per_split)[:, -1]),
				   'res_loss_train': np.mean(np.array(res_losses_train_per_split)[:, -1]),
				   'mpe_loss_train': np.mean(np.array(mpe_losses_train_per_split)[:, -1]),
				   'loss': np.mean(np.array(losses_per_split)[:, -1]),
				   'man_loss': np.mean(np.array(man_losses_per_split)[:, -1]),
				   'res_loss': np.mean(np.array(res_losses_per_split)[:, -1]),
				   'mpe_loss': np.mean(np.array(mpe_losses_per_split)[:, -1])})

		# ============================================

	# return (this also moves the data to the CPU)
	return (params_per_split,
			np.array(losses_train_per_split), np.array(man_losses_train_per_split), np.array(res_losses_train_per_split), np.array(mpe_losses_train_per_split),
			np.array(losses_per_split), np.array(man_losses_per_split), np.array(res_losses_per_split), np.array(mpe_losses_per_split))


def read(artificial=False):
	print('reading...')

	# read
	read_ids_gcamp = list(range(0, 33+1))
	# read_ids_gcamp = [0,   # minrevs
	# 				  1,   # minswitch
	# 				  7,   # maxswitch
	# 				  11,  # median
	# 				  12,  # maxrevs
	# 				  17,  # mean
	# 				  19,  # paper
	# 				  21]  # clean
	# read_ids_gcamp = [19]

	path = os.path.join(read_utils.get_data_path(), 'exported_data_gcamp')
	(file_ids_gcamp, filenames_gcamp,
	 traces_gcamp,
	 continuous_behaviors_gcamp,
	 discrete_states_gcamp, eigenworms_gcamp,
	 kymographs_gcamp) = read_utils.get_datas(path, read_ids_gcamp)

	# clear nan values
	for i in range(len(file_ids_gcamp)):
		mask = eigenworms_gcamp[i].notna().all(axis=1)

		traces_gcamp[i] = traces_gcamp[i][mask]
		# continuous_behaviors_gcamp[i] = continuous_behaviors_gcamp[i][mask]
		eigenworms_gcamp[i] = eigenworms_gcamp[i][mask]
		discrete_states_gcamp[i] = discrete_states_gcamp[i][mask]

	# kymographs_gcamp[i] = kymographs_gcamp[i][mask]
	# filenames_gcamp[i] = filenames_gcamp[i][mask]

	# naive ID unknown neurons
	for i in range(len(file_ids_gcamp)):
		traces_gcamp[i].columns = pd.Index([
			f'{file_ids_gcamp[i]:02d}_{n_id:03d}' if empty else label for n_id, (label, empty) in
			enumerate(zip(traces_gcamp[i].columns, traces_gcamp[i].columns == ''))
		])


	# artificialize
	if artificial:
		dims_generation = 2
		weights = (0.1, -0.25)

		selection_bhl = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']

		file_ids_art = list()
		filenames_art = list()

		discrete_states_art = list()
		eigenworms_art = list()

		modes_art = list()
		traces_reconstructed_art = list()
		traces_residual_art = list()
		traces_art = list()

		for i in range(len(file_ids_gcamp)):
			(filename_art,
			 neural_modes_art_,
			 traces_reconstructed_art_,
			 traces_residual_art_,
			 traces_art_,
			 params_art_,
			 params_cca_art_) = artificializer.artificialize(weights, hgm_gen,
															 filenames_gcamp[i],
															 traces_gcamp[i].columns,
															 eigenworms_gcamp[i],
															 [ds[selection_bhl] for ds in discrete_states_gcamp][i],
															 n_dimensions=dims_generation)
			filenames_art.append(filename_art)

			modes_art.append(neural_modes_art_)
			traces_reconstructed_art.append(traces_reconstructed_art_)
			traces_residual_art.append(traces_residual_art_)
			traces_art.append(traces_art_)

			file_ids_art.append(file_ids_gcamp[i])

			discrete_states_art.append(discrete_states_gcamp[i].copy())
			eigenworms_art.append(eigenworms_gcamp[i].copy())

		return file_ids_art, filenames_art, discrete_states_art, eigenworms_art, traces_art


	return file_ids_gcamp, filenames_gcamp, traces_gcamp, discrete_states_gcamp, eigenworms_gcamp


def sweep(prefix, model, config, default_hyperparameters, log_path):
	file_ids, filenames, traces, discrete_states, eigenworms = read(artificial=False)

	print('sweeping...')

	wandb.login()

	for file_id in file_ids:
		project_name = f'{prefix} {time.strftime("%d.%m %Hh%M", time.localtime())} {file_id} {filenames[file_ids.index(file_id)]}'
		print(f'{file_id}: {filenames[file_ids.index(file_id)]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces[file_ids.index(file_id)].values.shape[0]), discrete_states[file_ids.index(file_id)][['REV']].values))

		def train_model_wrapper():
			return train_model(model, splits,
							   traces[file_ids.index(file_id)].values,
							   eigenworms[file_ids.index(file_id)].values,
							   discrete_states[file_ids.index(file_id)].values,
							   default_hyperparameters,
							   log_path)

		sweep_id = wandb.sweep(sweep=config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)
