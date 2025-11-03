import os
import pandas as pd
import pynwb
import h5py


def get_data_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/DATA')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/scratch/neurobiology/zimmer/zimmermann/DATA/'


def get_data(path, ids=None):
	file_ids = ids if ids is not None else list(range(len(os.listdir(path))))
	hdf5files = get_hdf5_files(path, file_ids)
	filenames = [get_filename(file) for file in hdf5files]
	traces = [get_neuronal_traces(file) for file in hdf5files]
	ball_motions = [get_ball_motions(file) for file in hdf5files]
	behavioral_states = [get_behavioral_states(file) for file in hdf5files]
	return file_ids, filenames, traces, ball_motions, behavioral_states


def get_hdf5_files(path, file_ids):
	filenames = [sorted(os.listdir(path))[file_id] for file_id in file_ids]
	return [h5py.File(os.path.join(path, filename), 'r') for filename in filenames]


def get_filename(file_handle):
	filename = file_handle.filename.split('/')[-1].split('.')[0]
	return filename


def get_neuronal_traces(file_handle):
	data = pd.DataFrame(file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/data'][:])
	try:
		data.columns = [str(n_id) for n_id in file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/rois'][:]]
	except ValueError:
		data.columns = ['?' for _ in range(data.shape[1])]
	return data


def get_ball_motions(file_handle):
	return pd.DataFrame(file_handle[f'/processing/behavior/ball_motion/data'][:],
						columns=['?'])


def get_behavioral_states(file_handle):
	data = pd.DataFrame(file_handle[f'/processing/behavioral state/behavioral state/data'][:].round(0).astype(int))
	try:
		data.columns = ['undefined', 'quiescence', 'ab_bending', 'grooming_b', 'grooming_f', 'running']
	except ValueError:
		data.columns = ['?']
	return data
