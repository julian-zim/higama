import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_robot import read_utils
from utils_robot import generate_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from models import hgm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism


class Robot(Organism):
	weights: tuple
	n_neurons: int
	states: list

	def __init__(self, weights, n_neurons, states, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.subfolder = 'robot-2026-zimmermann'
		self.weights = weights
		self.n_neurons = n_neurons
		self.states = states
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'REV'

	def get_model_path(self):
		return read_utils.get_model_path()

	def read(self, file_id, data_variant='gcamp'):
		if data_variant != 'gcamp':
			print(f'Warning: Unavailable data variant \'{data_variant}\'.')

		# read
		path = os.path.join(read_utils.get_data_path(), 'exported_data_gcamp')
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
		 traces,
		 params,
		 params_cca) = generate_utils.artificialize(self.weights, hgm,
													filename_gcamp,
													[f'N{i}' for i in range(self.n_neurons)],
													eigenworms,
													discrete_states[self.states],
													n_dimensions=len(self.states))

		# return
		return filename, traces, eigenworms, discrete_states, (modes, traces_reconstructed, traces_residual, params, params_cca)
