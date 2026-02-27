import sys, os
import numpy as np
import scipy as sp
import pandas as pd

from utils_fly import read_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from organism import Organism


class Fly(Organism):

	def __init__(self, file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.subfolder = 'fly-2023-schaffer'
		super().__init__(file_ids, model_names, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'still'

	def get_model_path(self):
		return read_utils.get_model_path()

	def read(self, file_id, data_variant='gcamp'):
		if data_variant != 'gcamp':
			print(f'Warning: Unavailable data variant \'{data_variant}\'.')

		path = os.path.join(read_utils.get_data_path(), 'brain')

		# read
		(filename,
		 traces,
		 states, markers) = read_utils.get_data(path, file_id)

		# clear nan values
		mask = markers.notna().all(axis=1)

		traces = traces[mask]
		markers = markers[mask]
		states = states[mask]

		# naive ID unknown neurons
		traces.columns = pd.Index([
			f'{file_id:02d}_{n_id:03d}' if empty else label for n_id, (label, empty) in
			enumerate(zip(traces.columns, traces.columns == ''))
		])

		# preprocess
		def highpass_butter(x, fs, cutoff_hz, order=4, axis=0):
			return sp.signal.sosfiltfilt(sp.signal.butter(order, cutoff_hz, btype='highpass', fs=fs, output='sos'), x,
										 axis=axis)

		sampling_rate = read_utils.get_sampling_rate(os.path.join(path, filename))
		traces = pd.DataFrame(highpass_butter(traces.values, sampling_rate, 1 / 100), columns=traces.columns)
		# markers_grad = pd.DataFrame(np.gradient(markers.values, axis=0), columns=markers.columns)

		# add shifted markers in time
		# todo

		return filename, traces, markers, states, None
