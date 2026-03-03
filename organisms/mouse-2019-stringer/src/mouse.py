import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_mouse import read_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism


class Mouse(Organism):

	def __init__(self, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.subfolder = 'mouse-2019-stringer'
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'run'

	def get_model_path(self):
		return read_utils.get_model_path()

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

		return filename, traces, motion_pcs[0], classes, (continuous_behaviors, motion_masks, motion_pcs[1])
