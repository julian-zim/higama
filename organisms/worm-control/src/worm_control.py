import sys, os
import numpy as np
import scipy as sp
import pandas as pd
from typing import NamedTuple

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
import jax
import jax.numpy as jnp
import optax

sys.path.append(os.path.join(os.path.dirname(__file__), '../../worm-2025-fieseler/src'))
from utils_worm import read_utils
from utils_worm_control import read_utils as read_utils_control

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism
import organism
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from utils import train_utils
from models import trainer


class Lambdas(NamedTuple):
	loss_lambdas: tuple[float, float]
	reg_lambdas: tuple[float, float]
	cca_lambdas: tuple[float, float]


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

	print('Done.', end='')
	return models_spn, None, None, None, None, None, None


class Worm(Organism):
	shuffle_id: int

	def __init__(self, shuffle_id, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.shuffle_id = shuffle_id
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'REV'

	def get_model_path(self):
		return read_utils_control.get_model_path(self.shuffle_id)

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
								+ '_'.join(
							[f'{param.split('_')[0]}{hyperparams['reg_lambdas'][param]:.2E}' for param in
							 hyperparams['reg_lambdas'].keys()]))

				print(f'\nRunning {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)
					splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.shape[0]),
																				   classes[[self.get_split_class()]].values))

					model_params = train_utils.load_params(os.path.join(f'{self.get_model_path()}', 'Splits', 'params', model_folder), filename)

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
						 None,
						 None,
						 None,
						 None,
						 None),
						save_path=os.path.join(os.path.join(f'{self.get_model_path()}', 'Splits', 'predictions', model_folder)))

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
								+ '_'.join(
							[f'{param.split('_')[0]}{hyperparams['reg_lambdas'][param]:.2E}' for param in
							 hyperparams['reg_lambdas'].keys()]))

				print(f'\nTraining {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)
					splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.shape[0]),
																				   classes[[self.get_split_class()]].values))

					print(f'\n\t{file_id}: {filename}... ', end='')
					(params,
					 losses_train, man_losses_train, res_losses_train, mpe_losses_train,
					 losses, man_losses, res_losses, mpe_losses
					 ) = organism.train_dataset(model, splits,
									   traces.values,
									   behavior.values,
									   classes.values,
									   hyperparams)

					train_utils.save_params(filename, params,
											os.path.join(f'{self.get_model_path()}', 'Splits', 'params', model_folder))
					# train_utils.save_losses(filename, np.stack([
					# 	losses_train, man_losses_train, res_losses_train, mpe_losses_train,
					# 	losses, man_losses, res_losses, mpe_losses
					# ]), os.path.join(f'{self.get_model_path()}', 'Splits', 'history', model_folder))

	def read(self, file_id, data_variant='gcamp'):
		if data_variant not in ['gcamp', 'gfp']:
			print(f'Warning: Unavailable data variant \'{data_variant}\'.')

		path = os.path.join(read_utils.get_data_path(), f'exported_data_{data_variant}')

		# read
		(filename,
		 traces,
		 continuous_behaviors,
		 discrete_states, eigenworms,
		 kymograph) = read_utils.get_data(path, file_id)

		# clear nan values
		mask = eigenworms.notna().all(axis=1)

		traces = traces[mask]
		continuous_behaviors = continuous_behaviors[mask]
		eigenworms = eigenworms[mask]
		discrete_states = discrete_states[mask]
		kymograph = kymograph[mask]

		# naive ID unknown neurons
		traces.columns = pd.Index([
			f'{file_id:02d}_{n_id:03d}' if empty else label for n_id, (label, empty) in
			enumerate(zip(traces.columns, traces.columns == ''))
		])

		# preprocess
		pass

		# kill all information in behavior
		# eigenworms = pd.DataFrame(np.random.randn(*eigenworms.values.shape), columns=eigenworms.columns)
		# eigenworms = pd.DataFrame(data=np.zeros(eigenworms.values.shape), columns=eigenworms.columns)

		# shuffle behavior
		def surrogate_block_permute(X, block_size=50, random_state=None):
			"""
			Permute contiguous time blocks.

			Good general-purpose surrogate for autocorrelated time series.
			Does not assume circularity.
			"""
			rng = np.random.default_rng(random_state)
			n = len(X)

			blocks = [X.iloc[i:i + block_size] for i in range(0, n, block_size)]
			order = rng.permutation(len(blocks))

			Xs = pd.concat([blocks[i] for i in order], axis=0).copy()
			Xs.index = X.index
			return Xs

		behavior_shuffled = surrogate_block_permute(eigenworms, random_state=self.shuffle_id)

		return filename, traces, behavior_shuffled, discrete_states, (continuous_behaviors, kymograph)
