#region imports
import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_robot import read_utils
from utils_robot import generate_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism
#endregion


class Robot(Organism):
	weights: tuple
	neuron_labels: int
	states: list

	def __init__(self, weights, neuron_labels, states, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.weights = weights
		self.neuron_labels = neuron_labels
		self.states = states
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'REV'

	def get_model_path(self):
		return read_utils.get_model_path()

	def read(self, file_id, data_variant='gcamp'):
		if data_variant != 'gcamp':
			print(f'Warning: Unavailable data variant \'{data_variant}\'.')

		path = os.path.join(read_utils.get_data_path(), 'exported_data_gcamp')

		# read
		(filename_gcamp,
		 discrete_states, eigenworms) = read_utils.get_data(path, file_id)

		# clear nan values
		mask = eigenworms.notna().all(axis=1)

		eigenworms = eigenworms[mask]
		discrete_states = discrete_states[mask]

		# preprocess
		pass

		# generate
		(filename,
		 modes,
		 traces_reconstructed,
		 traces_residual,
		 traces_concrete,
		 traces_other,
		 traces,
		 params,
		 params_manifold) = generate_utils.artificialize(self.weights, 'hgm',
														 filename_gcamp,
														 self.neuron_labels,
														 eigenworms,
														 discrete_states[self.states],
														 read_utils.get_sampling_rate(read_utils.get_file_handle(path, file_id)),
														 n_dimensions=len(self.states))

		# # kill all information in behavior
		# eigenworms = pd.DataFrame(np.random.randn(*eigenworms.values.shape), columns=eigenworms.columns)
		# eigenworms = pd.DataFrame(data=np.zeros(eigenworms.values.shape), columns=eigenworms.columns)

		# return
		return filename, traces, eigenworms, discrete_states, (modes, traces_reconstructed, traces_residual, traces_concrete, traces_other, params, params_manifold)
