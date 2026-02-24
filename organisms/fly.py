import sys, os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
from organisms.organism import Organism


class Fly(Organism):

	def __init__(self, file_ids, model_variants, hyperparameters, sweep_config, data_variant='gcamp'):
		self.subfolder = 'fly-2023-schaffer'
		super().__init__(file_ids, model_variants, hyperparameters, sweep_config, data_variant)

	def get_split_class(self):
		return 'still'

	def read(self, file_id, data_variant='gcamp'):
		if data_variant != 'gcamp':
			print(f'Warning: Unavailable data variant \'{data_variant}\'.')

		sys.path.append(os.path.join(os.getcwd(), f'./{self.subfolder}/src/'))
		# noinspection PyUnresolvedReferences
		from utils_fly import read_utils
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

		return filename, traces, markers, states, None
