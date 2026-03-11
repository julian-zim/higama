#region imports
import sys, os
import gc
import time
import json

import wandb

from abc import ABC, abstractmethod
from typing import NamedTuple

import math
import numpy as np
import scipy as sp
import pandas as pd

import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils import plot_utils
from utils import train_utils
from models import trainer
#endregion


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float]
	cca_lambdas: tuple[float, float]


def run_dataset_on_all(model, model_params, traces, behavior, model_variant):
	# predict from all for traces
	model_predictions = [
		model.predict_and_score(params, traces.values, behavior.values) for params in model_params
	]

	models_man = [
		pd.DataFrame(data=np.array(split[1]), columns=[f'M{i+1}' for i in range(np.array(split[1]).shape[1])])
		for split in model_predictions
	]
	(models_rec,
	 models_res,
	 models_pres) = tuple([
							  pd.DataFrame(data=np.array(split[m]), columns=traces.columns)
							  for split in model_predictions
						  ] for m in [2, 3, 4])

	models_resss, models_bss = None, None
	if model_variant == 'ss':
		(models_resss,
		 models_bss) = tuple(
			[
				pd.DataFrame(data=np.array(split[m]), columns=[f'SS{j+1}' for j in range(np.array(split[m]).shape[1])])
				for split in model_predictions
			] for m in [5, 6]
		)

	print('Done.', end='')
	return None, models_man, models_rec, models_res, models_pres, models_resss, models_bss


def run_dataset(model, model_params, splits, traces, behavior, model_variant):
	# predict from eval set for neuron scores
	model_predictions = [
		model.predict_and_score(params, traces.values[eval], behavior.values[eval]) for params, (train, eval) in
		zip(model_params, splits)
	]

	models_spn = [
		pd.DataFrame(data=np.array(split[0])[None], columns=traces.columns)
		for split in model_predictions
	]

	# predict from all for traces
	model_predictions = [
		model.predict_and_score(params, traces.values, behavior.values) for params in model_params
	]

	models_man = [
		pd.DataFrame(data=np.array(split[1]), columns=[f'M{i+1}' for i in range(np.array(split[1]).shape[1])])
		for split in model_predictions
	]
	(models_rec,
	 models_res,
	 models_pres) = tuple([
							  pd.DataFrame(data=np.array(split[m]), columns=traces.columns)
							  for split in model_predictions
						  ] for m in [2, 3, 4])

	models_resss, models_bss = None, None
	if model_variant == 'ss':
		(models_resss,
		 models_bss) = tuple(
			[
				pd.DataFrame(data=np.array(split[m]), columns=[f'SS{j+1}' for j in range(np.array(split[m]).shape[1])])
				for split in model_predictions
			] for m in [5, 6]
		)

	print('Done.', end='')
	return models_spn, models_man, models_rec, models_res, models_pres, models_resss, models_bss


def train_dataset_on_all(model, N, B, S, hyperparams):
	# region hyperparamters ===============================================================
	seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(
		3), jax.random.PRNGKey(4)

	max_epochs = hyperparams['max_epochs']
	max_patience = hyperparams['max_patience']
	learning_rate = hyperparams['learning_rate']
	n_dimensions = hyperparams['dimensions']

	lambdas = Lambdas(
		loss_lambdas=(hyperparams['loss_man_lambda'], hyperparams['loss_res_lambda']),
		reg_lambdas=tuple(hyperparams['reg_lambdas'][param] for param in hyperparams['reg_lambdas'].keys()),
		cca_lambdas=(hyperparams['ccadiag_lambda'], hyperparams['ccaunit_lambda'])
	)
	# endregion ===========================================================================

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

	# region splitting and initializing (this also moves the data to the GPU) ==========================
	params = model.init_params(N, B, S,
							   n_dimensions,
							   seeds)

	optimizer = optax.adam(
		learning_rate=optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max_epochs,
												  alpha=1E-4))
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
					   N, B, S,
					   jnp.empty(N.shape), jnp.empty(B.shape), jnp.empty(S.shape))

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
	print(f'{f'Stopped after {epoch_end} epochs. ' if epoch_end < max_epochs else ''}',
		  end='')

	# ============================================

	# return
	return (params_per_split,
			np.stack(losses_train_per_split),
			np.stack(man_losses_train_per_split),
			np.stack(res_losses_train_per_split),
			np.stack(mpe_losses_train_per_split),
			np.stack(losses_per_split),
			np.stack(man_losses_per_split),
			np.stack(res_losses_per_split),
			np.stack(mpe_losses_per_split))


def train_dataset(model, splits, N, B, S, hyperparams):
	# region hyperparamters ===============================================================
	seeds = jax.random.PRNGKey(0), jax.random.PRNGKey(1), jax.random.PRNGKey(2), jax.random.PRNGKey(
		3), jax.random.PRNGKey(4)

	max_epochs = hyperparams['max_epochs']
	max_patience = hyperparams['max_patience']
	learning_rate = hyperparams['learning_rate']
	n_dimensions = hyperparams['dimensions']

	lambdas = Lambdas(
		loss_lambdas=(hyperparams['loss_man_lambda'], hyperparams['loss_res_lambda']),
		reg_lambdas=tuple(hyperparams['reg_lambdas'][param] for param in hyperparams['reg_lambdas'].keys()),
		cca_lambdas=(hyperparams['ccadiag_lambda'], hyperparams['ccaunit_lambda'])
	)
	# endregion ===========================================================================

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
								   n_dimensions,
								   seeds)

		optimizer = optax.adam(
			learning_rate=optax.cosine_decay_schedule(init_value=learning_rate, decay_steps=max_epochs,
													  alpha=1E-4))
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
						   N_eval, B_eval, S_eval)

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
		print(f'{f'Stopped after {epoch_end} epochs: ' if epoch_end < max_epochs else ''}Split {s_id} completed. ',
			  end='')

	# ============================================

	# return
	return (params_per_split,
			np.stack(losses_train_per_split),
			np.stack(man_losses_train_per_split),
			np.stack(res_losses_train_per_split),
			np.stack(mpe_losses_train_per_split),
			np.stack(losses_per_split),
			np.stack(man_losses_per_split),
			np.stack(res_losses_per_split),
			np.stack(mpe_losses_per_split))


def sweep_dataset(model, splits, N, B, S, hyperparameters, log_path):
	with wandb.init(dir=log_path):
		config = wandb.config
		hyperparams = hyperparameters.copy()
		for key in config:
			if key != '_wandb': hyperparams['reg_lambdas'][key] = config[key]

		(params_per_split,
		losses_train_per_split,
		man_losses_train_per_split,
		res_losses_train_per_split,
		mpe_losses_train_per_split,
		losses_per_split,
		man_losses_per_split,
		res_losses_per_split,
		mpe_losses_per_split) = train_dataset(model, splits, N, B, S, hyperparams)

		wandb.log({'loss_train': np.mean(losses_train_per_split[:, -1]),
				   'man_loss_train': np.mean(man_losses_train_per_split[:, -1]),
				   'res_loss_train': np.mean(res_losses_train_per_split[:, -1]),
				   'mpe_loss_train': np.mean(mpe_losses_train_per_split[:, -1]),
				   'loss': np.mean(losses_per_split[:, -1]),
				   'man_loss': np.mean(man_losses_per_split[:, -1]),
				   'res_loss': np.mean(res_losses_per_split[:, -1]),
				   'mpe_loss': np.mean(mpe_losses_per_split[:, -1])})


class Organism(ABC):
	model_names: list
	model_variants: list
	data_variant: str
	sweep_config: dict
	hyperparameters: dict
	file_ids: list
	filenames: list
	subfolder: str

	def __init__(self, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.file_ids = file_ids

		self.model_names = model_names
		self.model_variants = model_variants
		self.data_variant = data_variant

		self.hyperparameters = {
			'ccadiag_lambda': 0.6,
			'ccaunit_lambda': 1.0 - 0.6,
			'loss_man_lambda': 0.0,
			'loss_res_lambda': 1.0 - 0.0,
		}
		for param in hyperparameters.keys():
			self.hyperparameters[param] = hyperparameters[param]
		reg_lambdas = self.read_reg_lambdas(self.hyperparameters['reg_lambdas'] == 'manual')
		self.hyperparameters['reg_lambdas'] = reg_lambdas
		self.sweep_config = sweep_config
		self.sweep_config = {
			'method': 'bayes',
			'metric': {
				'name': 'mpe_loss',
				'goal': 'minimize',
			},
			'parameters': dict()
		}
		for param in sweep_config.keys():
			self.sweep_config['parameters'][param] = sweep_config[param]

	@abstractmethod
	def get_split_class(self):
		pass

	@abstractmethod
	def get_model_path(self):
		pass

	@abstractmethod
	def read(self, file_id, data_variant='gcamp'):
		pass

	def read_reg_lambdas(self, manual):
		ll = f'll{int(self.hyperparameters['loss_man_lambda'] * 100):02d}'

		if manual:
			with open('../hyperparams_manual.json', 'r') as f:
				reg_lambdas = json.load(f)
		else:
			with open('../hyperparams.json', 'r') as f:
				reg_lambdas = json.load(f)['geometric'][ll]

		return reg_lambdas

	def run_on_all(self):
		for model_variant in self.model_variants:
			if model_variant == 'ns':
				from models import bom as bom, hgm as hgm
			elif model_variant == 'ss':
				from models import bom_ss as bom, hgm_ss as hgm
			else:
				raise ValueError(f'Unknown model variant: {model_variant}. Use either \'ns\' or \'ss\'.')

			for model_name in self.model_names:
				try:
					model = {'bom': bom, 'hgm': hgm}[model_name]
				except KeyError:
					raise KeyError(f'Unknown model name: {model_name}. Use either \'bom\' or \'hgm\'.')

				hyperparams = self.hyperparameters.copy()
				hyperparams['reg_lambdas'] = hyperparams['reg_lambdas'][model_variant][model_name]

				model_folder = (f'{self.data_variant}_{model_name}_{model_variant}_'
								+ '_'.join([f'{param.split('_')[0]}{hyperparams['reg_lambdas'][param]:.2E}' for param in hyperparams['reg_lambdas'].keys()]))

				print(f'\nRunning {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)

					model_params = train_utils.load_params(os.path.join(f'{self.get_model_path()}/ALL', 'params', model_folder), filename)

					print(f'\n\t{file_id}: {filename}... ', end='')
					(model_spn,
					 model_man,
					 model_rec,
					 model_res,
					 model_pres,
					 model_resss,
					 model_bss) = run_dataset_on_all(model, model_params, traces, behavior, model_variant)

					train_utils.save_predictions(
						filename,
						(model_spn,
						 model_man,
						 model_rec,
						 model_res,
						 model_pres,
						 model_resss,
						 model_bss),
						save_path=os.path.join(os.path.join(f'{self.get_model_path()}/ALL', 'predictions', model_folder)))

	def run(self):
		for model_variant in self.model_variants:
			if model_variant == 'ns':
				from models import bom as bom, hgm as hgm
			elif model_variant == 'ss':
				from models import bom_ss as bom, hgm_ss as hgm
			else:
				raise ValueError(f'Unknown model variant: {model_variant}. Use either \'ns\' or \'ss\'.')

			for model_name in self.model_names:
				try:
					model = {'bom': bom, 'hgm': hgm}[model_name]
				except KeyError:
					raise KeyError(f'Unknown model name: {model_name}. Use either \'bom\' or \'hgm\'.')

				hyperparams = self.hyperparameters.copy()
				hyperparams['reg_lambdas'] = hyperparams['reg_lambdas'][model_variant][model_name]

				model_folder = (f'{self.data_variant}_{model_name}_{model_variant}_'
								+ '_'.join([f'{param.split('_')[0]}{hyperparams['reg_lambdas'][param]:.2E}' for param in hyperparams['reg_lambdas'].keys()]))

				print(f'\nRunning {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)
					splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.shape[0]),
																				   classes[[self.get_split_class()]].values))

					model_params = train_utils.load_params(os.path.join(f'{self.get_model_path()}', 'params', model_folder), filename)

					print(f'\n\t{file_id}: {filename}... ', end='')
					(model_spn,
					 model_man,
					 model_rec,
					 model_res,
					 model_pres,
					 model_resss,
					 model_bss) = run_dataset(model, model_params, splits, traces, behavior, model_variant)

					train_utils.save_predictions(
						filename,
						(model_spn,
						 model_man,
						 model_rec,
						 model_res,
						 model_pres,
						 model_resss,
						 model_bss),
						save_path=os.path.join(os.path.join(f'{self.get_model_path()}', 'predictions', model_folder)))

	def train_on_all(self):
		for model_variant in self.model_variants:
			if model_variant == 'ns':
				from models import bom as bom, hgm as hgm
			elif model_variant == 'ss':
				from models import bom_ss as bom, hgm_ss as hgm
			else:
				raise ValueError(f'Unknown model variant: {model_variant}. Use either \'ns\' or \'ss\'.')

			for model_name in self.model_names:
				try:
					model = {'bom': bom, 'hgm': hgm}[model_name]
				except KeyError:
					raise KeyError(f'Unknown model name: {model_name}. Use either \'bom\' or \'hgm\'.')

				hyperparams = self.hyperparameters.copy()
				hyperparams['reg_lambdas'] = hyperparams['reg_lambdas'][model_variant][model_name]

				model_folder = (f'{self.data_variant}_{model_name}_{model_variant}_'
								+ '_'.join([f'{param.split('_')[0]}{hyperparams['reg_lambdas'][param]:.2E}' for param in hyperparams['reg_lambdas'].keys()]))

				print(f'\nTraining {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)

					print(f'\n\t{file_id}: {filename}... ', end='')
					(params,
					 losses_train, man_losses_train, res_losses_train, mpe_losses_train,
					 losses, man_losses, res_losses, mpe_losses
					 ) = train_dataset_on_all(model,
									   traces.values,
									   behavior.values,
									   classes.values,
									   hyperparams)

					train_utils.save_params(filename, params,
											os.path.join(f'{self.get_model_path()}/ALL', 'params', model_folder))
					train_utils.save_losses(filename, np.stack([
						losses_train, man_losses_train, res_losses_train, mpe_losses_train,
						losses, man_losses, res_losses, mpe_losses
					]), os.path.join(f'{self.get_model_path()}/ALL', 'history', model_folder))

	def train(self):
		for model_variant in self.model_variants:
			if model_variant == 'ns':
				from models import bom as bom, hgm as hgm
			elif model_variant == 'ss':
				from models import bom_ss as bom, hgm_ss as hgm
			else:
				raise ValueError(f'Unknown model variant: {model_variant}. Use either \'ns\' or \'ss\'.')

			for model_name in self.model_names:
				try:
					model = {'bom': bom, 'hgm': hgm}[model_name]
				except KeyError:
					raise KeyError(f'Unknown model name: {model_name}. Use either \'bom\' or \'hgm\'.')

				hyperparams = self.hyperparameters.copy()
				hyperparams['reg_lambdas'] = hyperparams['reg_lambdas'][model_variant][model_name]

				model_folder = (f'{self.data_variant}_{model_name}_{model_variant}_'
								+ '_'.join([f'{param.split('_')[0]}{hyperparams['reg_lambdas'][param]:.2E}' for param in hyperparams['reg_lambdas'].keys()]))

				print(f'\nTraining {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)
					splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.shape[0]),
																				   classes[[self.get_split_class()]].values))

					print(f'\n\t{file_id}: {filename}... ', end='')
					(params,
					 losses_train, man_losses_train, res_losses_train, mpe_losses_train,
					 losses, man_losses, res_losses, mpe_losses
					 ) = train_dataset(model, splits,
									   traces.values,
									   behavior.values,
									   classes.values,
									   hyperparams)

					train_utils.save_params(filename, params,
											os.path.join(f'{self.get_model_path()}', 'params', model_folder))
					train_utils.save_losses(filename, np.stack([
						losses_train, man_losses_train, res_losses_train, mpe_losses_train,
						losses, man_losses, res_losses, mpe_losses
					]), os.path.join(f'{self.get_model_path()}', 'history', model_folder))

	def sweep(self):
		log_path = '/lisc/data/scratch/neurobiology/zimmer/zimmermann/Sweeps/'
		if not (os.path.exists(log_path) and os.access(log_path, os.W_OK)):
			log_path = f'./out/'
		wandb.login()

		for model_variant in self.model_variants:
			if model_variant == 'ns':
				from models import bom as bom, hgm as hgm
			elif model_variant == 'ss':
				from models import bom_ss as bom, hgm_ss as hgm
			else:
				raise ValueError(f'Unknown model variant: {model_variant}. use either \'ns\' or \'ss\'.')

			for model_name in self.model_names:
				try:
					model = {'bom': bom, 'hgm': hgm}[model_name]
				except KeyError:
					raise KeyError(f'Unknown model name: {model_name}. Use either \'bom\' or \'hgm\'.')

				hyperparams = self.hyperparameters.copy()
				hyperparams['reg_lambdas'] = hyperparams['reg_lambdas'][model_variant][model_name]

				sweep_cfg = self.sweep_config.copy()
				sweep_cfg['run_cap'] = sweep_cfg['parameters'][model_variant][model_name]['run_cap']
				sweep_cfg['parameters'] = sweep_cfg['parameters'][model_variant][model_name]['parameters']

				print(f'\nSweeping {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)
					splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.shape[0]),
																					   classes[[self.get_split_class()]].values))

					def sweep_dataset_wrapper():
						return sweep_dataset(model, splits,
											 traces.values,
											 behavior.values,
											 classes.values,
											 hyperparams,
											 log_path)

					project_name = f'{self.subfolder.split('-')[0]} {model_name}_{model_variant}-{f'll{int(hyperparams['loss_man_lambda'] * 100):02d}'} {time.strftime("%d.%m %Hh%M", time.localtime())} {file_id} {filename}'
					sweep_id = wandb.sweep(sweep=sweep_cfg, project=project_name)

					print(f'\n\t{file_id}: {filename}... ', end='')
					wandb.agent(sweep_id, function=sweep_dataset_wrapper)

					jax.clear_caches()
					gc.collect()
