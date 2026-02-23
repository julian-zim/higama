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


sys.path.append(os.path.join(os.getcwd(), '../'))
from utils_mouse import read_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from models import trainer
#endregion


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float]
	cca_lambdas: tuple[float, float]


def train_model(model, splits, N, B, S, manual_hyperparameters, log_path):
	with wandb.init(dir=log_path):
		config = wandb.config

		# region hyperparamters ===============================================================
		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3), jax.random.PRNGKey(4)

		hyperparameters = manual_hyperparameters.copy()
		for key in config:
			hyperparameters[key] = config[key]

		max_epochs = hyperparameters['max_epochs']
		max_patience = hyperparameters['max_patience']

		learning_rate = hyperparameters['learning_rate']

		n_dimensions = hyperparameters['dimensions']

		lambdas = Lambdas(
			loss_lambdas=(hyperparameters['loss_man_lambda'], hyperparameters['loss_res_lambda']),
			reg_lambdas=(hyperparameters['vb_reg_lambda'], hyperparameters['wm_reg_lambda']),
			cca_lambdas=(hyperparameters['ccadiag_lambda'], hyperparameters['ccaunit_lambda'])
		)
		# endregion ===========================================================================

		n_neurons = N.shape[1]
		n_states = S.shape[1]
		n_behaviors = B.shape[1]

		N, B, S = jnp.asarray(N), jnp.asarray(B), jnp.asarray(S)

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
			# region splitting and initializing (this also moves the data to the GPU) ==========================
			N_train, N_eval = N[train], N[eval]
			B_train, B_eval = B[train], B[eval]
			S_train, S_eval = S[train], S[eval]

			params = model.init_params(N_train, B_train, S_train,
									   seeds,
									   (n_neurons, n_dimensions),
									   (n_states, n_dimensions),
									   (n_behaviors, n_neurons),
									   (n_behaviors, n_behaviors),
									   *((n_dimensions, n_behaviors),) if model.get_type() else (),
									  )

			optimizer = optax.adam(learning_rate=optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max_epochs, alpha=1E-4))
			# optimizer = optax.adam(learning_rate=learning_rate)
			optimizer = optax.masked(optimizer, {'prep': False, 'model': True})
			opt_state = optimizer.init(params)
			# endregion ========================================================================================

			# train the split
			(params,
			 losses_train, man_losses_train, res_losses_train, mpe_losses_train,
			 losses, man_losses, res_losses, mpe_losses, epoch_end
			 ) = trainer.train(lambdas,
							   max_epochs, max_patience,
							   model, optimizer.update,
							   opt_state, params,
							   N_train, B_train, S_train,
							   N_eval,  B_eval,  S_eval)

			# save the split
			params_per_split.append(jax.device_get(params))

			losses_train_per_split.append(jax.device_get(losses_train))
			man_losses_train_per_split.append(jax.device_get(man_losses_train))
			res_losses_train_per_split.append(jax.device_get(res_losses_train))
			mpe_losses_train_per_split.append(jax.device_get(mpe_losses_train))

			losses_per_split.append(jax.device_get(losses))
			man_losses_per_split.append(jax.device_get(man_losses))
			res_losses_per_split.append(jax.device_get(res_losses))
			mpe_losses_per_split.append(jax.device_get(mpe_losses))

		# log
		losses_train_per_split = np.stack(losses_train_per_split)
		man_losses_train_per_split = np.stack(man_losses_train_per_split)
		res_losses_train_per_split = np.stack(res_losses_train_per_split)
		mpe_losses_train_per_split = np.stack(mpe_losses_train_per_split)
		losses_per_split = np.stack(losses_per_split)
		man_losses_per_split = np.stack(man_losses_per_split)
		res_losses_per_split = np.stack(res_losses_per_split)
		mpe_losses_per_split = np.stack(mpe_losses_per_split)

		wandb.log({'loss_train': np.mean(losses_train_per_split[:, -1]),
				   'man_loss_train': np.mean(man_losses_train_per_split[:, -1]),
				   'res_loss_train': np.mean(res_losses_train_per_split[:, -1]),
				   'mpe_loss_train': np.mean(mpe_losses_train_per_split[:, -1]),
				   'loss': np.mean(losses_per_split[:, -1]),
				   'man_loss': np.mean(man_losses_per_split[:, -1]),
				   'res_loss': np.mean(res_losses_per_split[:, -1]),
				   'mpe_loss': np.mean(mpe_losses_per_split[:, -1])})
	# ============================================

	# return
	return (params_per_split,
			losses_train_per_split,
			man_losses_train_per_split,
			res_losses_train_per_split,
			mpe_losses_train_per_split,
			losses_per_split,
			man_losses_per_split,
			res_losses_per_split,
			mpe_losses_per_split)


def sweep(file_id, prefix, model, config, manual_hyperparameters, log_path):
	wandb.login()

	# read
	path = os.path.join(read_utils.get_data_path(), '6163622/sessions')
	(filename,
	 traces,
	 continuous_behaviors,
	 motion_pcs,
	 motion_masks) = read_utils.get_data(path, file_id, with_whisking=True)

	# clear nan values
	# mask = motion_pcs[0].notna().all(axis=1)
	mask = continuous_behaviors.notna().all(axis=1)

	traces = traces[mask]
	motion_pcs[0] = motion_pcs[0][mask]
	motion_pcs[1] = motion_pcs[1][mask]
	continuous_behaviors = continuous_behaviors[mask]

	# naive ID unknown neurons
	traces.columns = pd.Index([
		f'{file_id:02d}_{n_id:03d}' if empty or type(label) is not str else label for n_id, (label, empty) in
		enumerate(zip(traces.columns, traces.columns == ''))
	])

	# preprocess
	def reconstruct_calcium(spikes, fs, decay=2.0):
		dt = 1.0 / fs
		gamma = np.exp(-dt / decay)
		trace = np.zeros(spikes.shape, dtype=np.float32)
		for t in range(1, spikes.shape[0]):
			trace[t, :] = gamma * trace[t - 1, :] + spikes[t, :]
		return trace

	sampling_rate = read_utils.get_sampling_rate(os.path.join(path, filename))
	traces = pd.DataFrame(reconstruct_calcium(traces.values, fs=sampling_rate), columns=traces.columns)
	behaviors = motion_pcs[0]
	classes = pd.DataFrame(continuous_behaviors[['running_speed']].values > np.quantile(continuous_behaviors[['running_speed']].values, 0.05), columns=['run'])

	# sweep
	print(f'Sweeping {file_id}: {filename}...')
	project_name = f'MOUSE {prefix} {time.strftime("%d.%m %Hh%M", time.localtime())} {file_id} {filename}'
	print(f'{file_id}: {filename}  ', end='')

	splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.values.shape[0]), classes))

	def train_model_wrapper():
		return train_model(model, splits,
						   traces.values,
						   behaviors.values,
						   classes.values,
						   manual_hyperparameters,
						   log_path)

	sweep_id = wandb.sweep(sweep=config, project=project_name)
	wandb.agent(sweep_id, function=train_model_wrapper)
