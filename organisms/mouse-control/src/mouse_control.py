#region imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_mouse_control import read_utils as read_utils_control
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism

sys.path.append(os.path.join(os.path.dirname(__file__), '../../mouse-2019-stringer/src'))
from utils_mouse import read_utils
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
