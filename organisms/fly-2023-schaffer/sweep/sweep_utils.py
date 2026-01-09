#region imports
import sys
import os
import time
from typing import NamedTuple
import math
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import jax
import jax.numpy as jnp
import optax
import wandb

sys.path.append(os.path.join(os.getcwd()))
from utils import read_utils
from utils import plot_utils
from utils import train_utils

# sys.path.append(os.path.join(os.getcwd(), '../../'))
# from models import bom
# from models import hgm
#endregion


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float]
	cca_lambdas: tuple[float, float]


def train_model(model, splits, N, B_ll, B_hl, default_hyperparameters, log_path):
	with wandb.init(dir=log_path):
		config = wandb.config

		#region hyperparamters ===============================================================
		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3)

		hyperparameters = default_hyperparameters.copy()
		for key in config:
			hyperparameters[key] = config[key]

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
			B_hl_train, B_hl_eval = jnp.array(B_hl[train]), jnp.array(B_hl[eval])
			B_ll_train, B_ll_eval = jnp.array(B_ll[train]), jnp.array(B_ll[eval])

			params = model.init_params(N_train, B_hl_train,
									   seeds,
									   (n_neurons, n_dimensions),
									   (n_behaviors_hl, n_dimensions),
									   (n_behaviors_ll, n_neurons),
									   * ((n_dimensions, n_neurons),) if model.get_type() else (),
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
				 mpe_loss) = tuple(l.item() for l in model.loss_fn_flattened(params, N_eval, B_ll_eval, B_hl_eval, lambdas))

				(opt_state, params, loss_types_train) = model.update(optimizer.update, opt_state, params, N_train, B_ll_train, B_hl_train, lambdas)

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


def read():
	print('reading...')

	# read_ids = list(range(0, 13+1))
	# read_ids = [0, 2, 3, 8, 9]
	read_ids = [0]
	plot_id = 0

	selection_bhl = ['still', 'walk', 'front-groom', 'back-groom', 'abdomen-move']

	# read

	path = os.path.join(read_utils.get_data_path(), 'fly-2023-schaffer/brain')

	(file_ids, filenames,
	 traces,
	 states, markers) = read_utils.get_data(path, read_ids)

	# # clear nan values
	# for i in range(len(file_ids)):
	# 	mask = markers[i].notna().all(axis=1)
	#
	# 	traces[i] = traces[i][mask]
	# 	markers[i] = markers[i][mask]
	# 	states[i] = states[i][mask]
	#
	# 	#filenames[i] = filenames[i][mask]

	# # remove unused discrete state labels
	# states = [ds[selection_bhl] for ds in states]

	# naive ID unknown neurons
	for i in range(len(file_ids)):
		traces[i].columns = pd.Index([
			f'{file_ids[i]:02d}_{n_id:03d}' if empty else label for n_id, (label, empty) in
			enumerate(zip(traces[i].columns, traces[i].columns == ''))
		])

	return file_ids, filenames, states, markers, traces


def sweep(prefix, model, config, default_hyperparameters, log_path):
	file_ids, filenames, states, markers, traces = read()

	def highpass_butter(x, fs, cutoff_hz, order=4, axis=0):
		return sp.signal.sosfiltfilt(sp.signal.butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos'), x,
									 axis=axis)

	path = os.path.join(read_utils.get_data_path(), 'fly-2023-schaffer/brain')
	sampling_rates = [read_utils.get_sampling_rate(os.path.join(path, filenames[file_ids.index(file_id)])) for file_id
					  in file_ids]

	traces = [pd.DataFrame(data=trace.values[1:, :], columns=trace.columns) for trace in traces]
	traces = [pd.DataFrame(highpass_butter(trace.values, sr, 1 / 100), columns=trace.columns) for trace, sr in
			  zip(traces, sampling_rates)]

	states = [pd.DataFrame(data=state.values[1:, :], columns=state.columns) for state in states]

	markers = [pd.DataFrame(data=marker.values[1:, :] - marker.values[:-1, :], columns=marker.columns) for marker in
			   markers]
	markers = [
		pd.DataFrame(
			data=(marker.values - marker.values.mean(axis=0)) / np.where(
				(marker_std := np.std(marker.values, axis=0)) == 0.0, 1.0, marker_std),
			columns=marker.columns)
		for marker in markers
	]

	print('sweeping...')

	wandb.login()

	for file_id in file_ids:
		project_name = f'{prefix} {file_id} {filenames[file_ids.index(file_id)]} {time.strftime("%d.%m %Hh%M", time.localtime())}'
		print(f'{file_id}: {filenames[file_ids.index(file_id)]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces[file_ids.index(file_id)].values.shape[0]), states[file_ids.index(file_id)]['still'].values))

		def train_model_wrapper():
			return train_model(model, splits,
							   traces[file_ids.index(file_id)].values,
							   markers[file_ids.index(file_id)].values,
							   states[file_ids.index(file_id)].values,
							   default_hyperparameters,
							   log_path)

		sweep_id = wandb.sweep(sweep=config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)
