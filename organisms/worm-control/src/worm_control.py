#region imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_worm_control import read_utils as read_utils_control
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism

sys.path.append(os.path.join(os.path.dirname(__file__), '../../worm-2025-fieseler/src'))
from utils_worm import read_utils
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


class Worm(Organism):
	shuffle_id: int

	def __init__(self, shuffle_id, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.shuffle_id = shuffle_id
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'REV'

	def get_model_path(self):
		return read_utils_control.get_model_path(self.shuffle_id)

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
