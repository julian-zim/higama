#region imports
import sys, os
import gc
import time
import numpy as np
import scipy as sp
import pandas as pd

import wandb
import jax
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

from utils_mouse_control import read_utils as read_utils_control
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism
import organism

sys.path.append(os.path.join(os.path.dirname(__file__), '../../mouse-2019-stringer/src'))
from utils_mouse import read_utils
from utils import train_utils
#endregion


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


class Mouse(Organism):
	shuffle_id: int

	def __init__(self, shuffle_id, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.shuffle_id = shuffle_id
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'run'

	def get_model_path(self):
		return read_utils_control.get_model_path(self.shuffle_id)

	def do(self, action, full=False):
		if action == 'sweep':
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

				if action == 'sweep':
					sweep_cfg = self.sweep_config.copy()
					sweep_cfg['run_cap'] = sweep_cfg['parameters'][model_variant][model_name]['run_cap']
					sweep_cfg['parameters'] = sweep_cfg['parameters'][model_variant][model_name]['parameters']

				print(f'\n{action.capitalize()} {model_name}_{model_variant}:', end='')
				for file_id in self.file_ids:
					filename, traces, behavior, classes, aux = self.read(file_id, self.data_variant)
					if full:
						splits = [(np.arange(traces.shape[0]), np.arange(traces.shape[0]))]
					else:
						splits = list(StratifiedKFold(n_splits=5, shuffle=False).split(np.zeros(traces.shape[0]),
																					   classes[[self.get_split_class()]].values))

					print(f'\n\t{file_id}: {filename}... ', end='')
					match action:
						case 'run':
							model_params = train_utils.load_params(
								os.path.join(f'{self.get_model_path()}', 'Full' if full else 'Splits', 'params', model_folder),
								filename)

							os.remove(os.path.join(f'{self.get_model_path()}', 'Full' if full else 'Splits', 'params', model_folder, f'{filename}.msgpack'))

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
								save_path=os.path.join(
									os.path.join(f'{self.get_model_path()}', 'Full' if full else 'Splits', 'predictions',
												 model_folder)))

						case 'train':
							(params,
							 losses_train, man_losses_train, res_losses_train, mpe_losses_train,
							 losses, man_losses, res_losses, mpe_losses
							 ) = organism.train_dataset(model, splits,
											   traces.values,
											   behavior.values,
											   classes.values,
											   hyperparams)

							train_utils.save_params(filename, params,
													os.path.join(f'{self.get_model_path()}', 'Full' if full else 'Splits',
																 'params', model_folder))

							# train_utils.save_losses(filename, np.stack([
							# 	losses_train, man_losses_train, res_losses_train, mpe_losses_train,
							# 	losses, man_losses, res_losses, mpe_losses
							# ]), os.path.join(f'{self.get_model_path()}', 'Full' if full else 'Splits', 'history', model_folder))

						case 'sweep':
							def sweep_dataset_wrapper():
								return organism.sweep_dataset(model, splits,
													 traces.values,
													 behavior.values,
													 classes.values,
													 hyperparams,
													 log_path)

							project_name = f'{os.path.basename(self.get_model_path()).split('-')[0]} {model_name}_{model_variant}-{f'll{int(hyperparams['loss_man_lambda'] * 100):02d}'} {time.strftime("%d.%m %Hh%M", time.localtime())} {file_id} {filename}'
							sweep_id = wandb.sweep(sweep=sweep_cfg, project=project_name)
							wandb.agent(sweep_id, function=sweep_dataset_wrapper)

							jax.clear_caches()
							gc.collect()

						case _:
							raise ValueError()

	def read(self, file_id, data_variant='gcamp'):
		if data_variant != 'gcamp':
			print(f'Warning: Unavailable data variant \'{data_variant}\'.')

		path = os.path.join(read_utils.get_data_path(), '6163622/sessions')

		# read
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
		classes = pd.DataFrame(
			continuous_behaviors[['running_speed']].values > np.quantile(continuous_behaviors[['running_speed']].values,
																		 0.05),
			columns=['run']
		)

		# # kill all information in behavior
		# motion_pcs[0] = pd.DataFrame(np.random.randn(*motion_pcs[0].values.shape), columns=motion_pcs[0].columns)
		# motion_pcs[0] = pd.DataFrame(data=np.zeros(motion_pcs[0].values.shape), columns=motion_pcs[0].columns)

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

		behavior_shuffled = surrogate_block_permute(motion_pcs[0], random_state=self.shuffle_id)

		return filename, traces, behavior_shuffled, classes, (continuous_behaviors, motion_masks, motion_pcs[1])
