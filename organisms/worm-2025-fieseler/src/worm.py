import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_worm import read_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism


class Worm(Organism):

	def __init__(self, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.subfolder = 'worm-2025-fieseler'
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'REV'

	def get_model_path(self):
		return read_utils.get_model_path()

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

		return filename, traces, eigenworms, discrete_states, (continuous_behaviors, kymograph)
