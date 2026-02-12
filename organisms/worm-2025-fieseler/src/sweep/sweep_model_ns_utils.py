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
from models import trainer
from models import hgm_rs as hgm_gen
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
			reg_lambdas=(hyperparameters['wb_reg_lambda'], hyperparameters['wm_reg_lambda']),
			cca_lambdas=(hyperparameters['ccadiag_lambda'], hyperparameters['ccaunit_lambda'])
		)
		# endregion ===========================================================================

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
			# region splitting and initializing (this also moves the data to the GPU) ==========================
			N, B, S = jnp.array(N), jnp.array(B), jnp.array(S)
			N_train, N_eval = N[train], N[eval]
			B_train, B_eval = B[train], B[eval]
			S_train, S_eval = S[train], S[eval]

			params = model.init_params(N_train, B_train, S_train,
									   seeds,
									   (n_neurons, n_dimensions),
									   (n_states, n_dimensions),
									   (n_behaviors, n_neurons),
									   *((n_dimensions, n_neurons),) if model.get_type() else (),
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
							   model, optimizer,
							   opt_state, params,
							   N_train, B_train, S_train,
							   N_eval, B_eval, S_eval, )

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
			wandb.log({'loss_train': np.mean(np.array(losses_train_per_split)[:, -1]),
					   'man_loss_train': np.mean(np.array(man_losses_train_per_split)[:, -1]),
					   'res_loss_train': np.mean(np.array(res_losses_train_per_split)[:, -1]),
					   'mpe_loss_train': np.mean(np.array(mpe_losses_train_per_split)[:, -1]),
					   'loss': np.mean(np.array(losses_per_split)[:, -1]),
					   'man_loss': np.mean(np.array(man_losses_per_split)[:, -1]),
					   'res_loss': np.mean(np.array(res_losses_per_split)[:, -1]),
					   'mpe_loss': np.mean(np.array(mpe_losses_per_split)[:, -1])})

			print(f'{f'Stopped after {epoch_end} epochs: ' if epoch_end < max_epochs else ''}Split {s_id} completed. ', end='')
		# ============================================

	# return
	return (params_per_split,
			np.array(losses_train_per_split),
			np.array(man_losses_train_per_split),
			np.array(res_losses_train_per_split),
			np.array(mpe_losses_train_per_split),
			np.array(losses_per_split),
			np.array(man_losses_per_split),
			np.array(res_losses_per_split),
			np.array(mpe_losses_per_split))


def read():
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

	return file_ids_gcamp, filenames_gcamp, traces_gcamp, discrete_states_gcamp, eigenworms_gcamp


def sweep(prefix, model, config, manual_hyperparameters, log_path):
	file_ids, filenames, traces, discrete_states, eigenworms = read()

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
							   manual_hyperparameters,
							   log_path)

		sweep_id = wandb.sweep(sweep=config, project=project_name)
		wandb.agent(sweep_id, function=train_model_wrapper)
