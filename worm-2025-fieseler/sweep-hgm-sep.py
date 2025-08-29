import sys
import os
import math
import time

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
# from utils import plot_utils
from utils import train_utils
from utils import artificializer
from models import hgm_sep as hgm


def cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min_ratio):
	if step < warmup_steps:
		return lr_max * step / warmup_steps
	progress = (step - warmup_steps) / (total_steps - warmup_steps)
	lr_min = lr_max * lr_min_ratio
	return lr_min + 0.5*(lr_max - lr_min)*(1 + math.cos(math.pi*progress))

def train_numerical_model(project_name, model, splits, B_ll, N_res, N_man=None, hyperparameters=None):
	with wandb.init(project=project_name):
		config = wandb.config

		is_hgm = model.get_type()

		# Hyperparamters ============
		seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(3)

		if hyperparameters is None:
			hyperparameters = {
				'max_epochs': 10000,
				'max_patience': 10000,
				'learning_rate': config.learning_rate,
				'w1_reg_lambda': config.w1_reg_lambda,
				'w2_reg_lambda': config.w2_reg_lambda,
			}

		max_epochs = hyperparameters['max_epochs']
		max_patience = hyperparameters['max_patience']

		learning_rate = hyperparameters['learning_rate']

		reg_lambdas = hyperparameters['w1_reg_lambda'], hyperparameters['w2_reg_lambda']
		# ===========================

		n_neurons = N_res.shape[1]
		if is_hgm: n_dimensions = N_man.shape[1]
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
			# splitting and initializing ========================= (this also moves the data to the GPU)
			B_ll_train, B_ll_test = jnp.array(B_ll[train]), jnp.array(B_ll[test])
			if is_hgm: N_man_train, N_man_test = jnp.array(N_man[train]), jnp.array(N_man[test])
			N_res_train, N_res_test = jnp.array(N_res[train]), jnp.array(N_res[test])

			if is_hgm:
				params = model.init_params(seeds, (n_behaviors_ll, n_neurons), (n_dimensions, n_neurons))
			else:
				params = model.init_params(seeds, (n_behaviors_ll, n_neurons))
			optimizer = optax.adam(learning_rate)
			opt_state = optimizer.init(params)
			# ==========================================================================================

			if is_hgm:
				losses = [model.loss_fn(params, B_ll_test, N_man_test, N_res_test, reg_lambdas).item()]
				losses_tr = [model.loss_fn(params, B_ll_train, N_man_train, N_res_train, reg_lambdas).item()]
				mse_losses = [model.mse_loss_fn(params, B_ll_test, N_man_test, N_res_test).item()]
				mse_losses_tr = [model.mse_loss_fn(params, B_ll_train, N_man_train, N_res_train).item()]
			else:
				losses = [model.loss_fn(params, B_ll_test, N_res_test, reg_lambdas).item()]
				losses_tr = [model.loss_fn(params, B_ll_train, N_res_train, reg_lambdas).item()]
				mse_losses = [model.mse_loss_fn(params, B_ll_test, N_res_test).item()]
				mse_losses_tr = [model.mse_loss_fn(params, B_ll_train, N_res_train).item()]
			__grads = list()

			best_params = params
			best_loss = sys.float_info.max
			best_loss_tr = sys.float_info.max
			best_mse_loss = sys.float_info.max
			best_mse_loss_tr = sys.float_info.max

			# lr_step = 0
			patience = 0
			for epoch in range(max_epochs):
				# lr = cosine_lr(epoch, max_epochs / 10, max_epochs, learning_rate, 1E-6)
				# optimizer = optax.adam(lr)

				if is_hgm:
					params, opt_state, __grad = model.update(optimizer.update, opt_state, params, B_ll_train,
															 N_man_train, N_res_train, reg_lambdas)
					loss = model.loss_fn(params, B_ll_test, N_man_test, N_res_test, reg_lambdas).item()
					loss_tr = model.loss_fn(params, B_ll_train, N_man_train, N_res_train, reg_lambdas).item()
					mse_loss = model.mse_loss_fn(params, B_ll_test, N_man_test, N_res_test).item()
					mse_loss_tr = model.mse_loss_fn(params, B_ll_train, N_man_train, N_res_train).item()
				else:
					params, opt_state, __grad = model.update(optimizer.update, opt_state, params, B_ll_train,
															 N_res_train, reg_lambdas)
					loss = model.loss_fn(params, B_ll_test, N_res_test, reg_lambdas).item()
					loss_tr = model.loss_fn(params, B_ll_train, N_res_train, reg_lambdas).item()
					mse_loss = model.mse_loss_fn(params, B_ll_test, N_res_test).item()
					mse_loss_tr = model.mse_loss_fn(params, B_ll_train, N_res_train).item()
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

						# lr_step += 1
						# optimizer = optax.adam(learning_rate / (10 ** lr_step))
						# patience = 0
						# print(f'\nReduced learning rate from {learning_rate / (10 ** (lr_step - 1))} to {learning_rate / (10 ** lr_step)} after epoch {epoch} in split {s_id}', end='')

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
			if is_hgm:
				score_per_neuron = model.score_fn(params, B_ll, N_man, N_res)
			else:
				score_per_neuron = model.score_fn(params, B_ll, N_res)
			neuron_scores_per_split.append(score_per_neuron)
		# ===============================================================================================

		# =====================================================

		wandb.log({'score': np.mean(np.mean(np.array(neuron_scores_per_split), axis=1)),
				   'loss': np.mean(np.array(losses_per_split)[:, -1]),
				   'loss_tr': np.mean(np.array(losses_tr_per_split)[:, -1]),
				   'mse_loss': np.mean(np.array(mse_losses_per_split)[:, -1]),
				   'mse_loss_tr': np.mean(np.array(mse_losses_tr_per_split)[:, -1])})

		return params_per_split, np.array(neuron_scores_per_split), np.array(losses_per_split), np.array(
			losses_tr_per_split), np.array(mse_losses_per_split), np.array(mse_losses_tr_per_split), __grads_per_split



def read(artificial=False):
	print('reading...')

	selection_bhl = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']

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

	discrete_states = [ds[selection_bhl] for ds in discrete_states]


	# artificialize
	if artificial:
		dims_generation = 1
		weights = (0.5, 2.)

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
		# 					   traces_art[0][['AVAR', 'ALA', 'VB02', 'IL2L', 'SMDDR', 'RID']],
		# 					   discrete_states_art[0]['REV'],
		# 					   n_plots=None, plots_per_row=2, ylimits=(-8, 8))
		return filenames_art, continuous_behaviors, discrete_states_art, eigenworms_art, kymographs, traces_art


	# plot_utils.plot_traces(f'[{filenames[0]}] traces artificial',
	# 					   traces[0][['AVAR', 'ALA', 'VB02', 'IL2L', 'SMDDR', 'RID']],
	# 					   discrete_states[0]['REV'],
	# 					   n_plots=None, plots_per_row=2, ylimits=(-2, 2))
	return filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces



def manifold(traces, discrete_states):
	print('manifolding...')

	(neural_modes,
	 traces_manifold,
	 cca_loadings_man,
	 vars_expl_man, vars_expl_man_rs) = train_utils.get_manifolds(traces,
																  discrete_states,
																  n_cc=2, n_pc=5)
	traces_residual = [pd.DataFrame(data=trace.values - trace_manifold.values, columns=trace.columns) for trace, trace_manifold in zip(traces, traces_manifold)]

	return neural_modes, traces_residual, cca_loadings_man, vars_expl_man, vars_expl_man_rs



def sweep(config):
	filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces = read(artificial=False)
	neural_modes, traces_residual, cca_loadings_man, vars_expl_man, vars_expl_man_rs = manifold(traces, discrete_states)

	print('sweeping...')

	wandb.login()

	for i in range(len(filenames)):
		project_name = f'hgm {time.strftime("%Y-%m-%d %Hh%M", time.localtime())} DATASET {i}-{filenames[i]}'
		print(f'{i + 1}/{len(filenames)}: {filenames[i]}  ', end='')

		splits = list(StratifiedKFold(n_splits=5).split(np.zeros(eigenworms[i].values.shape[0]), discrete_states[i]['REV'].values))

		def train_model_wrapper():
			return train_numerical_model(project_name, hgm, splits, eigenworms[i].values, traces_residual[i].values, neural_modes[i].values)

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
		'learning_rate': {'min': 1E-6, 'max': 1.},
		'w1_reg_lambda': {'min': 0., 'max': 1.},
		'w2_reg_lambda': {'min': 0., 'max': 1.},
	}
}
sweep(sweep_config)

