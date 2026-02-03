import os
import numpy as np
import pandas as pd
import pickle
import h5py
import pynwb


def get_data_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/DATA/fly-2023-schaffer')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/data/scratch/neurobiology/zimmer/zimmermann/DATA//fly-2023-schaffer'


def get_datas(path, file_ids=None, with_markers=True):
	if file_ids is None:
		file_ids = list(range(len(sorted(os.listdir(path)))))

	filepaths = [get_filepath(path, file_id) for file_id in file_ids]
	filenames = [get_filename(filepath) for filepath in filepaths]
	traces = [get_traces(filepath) for filepath in filepaths]
	states = [get_states(filepath) for filepath in filepaths]
	markers = [get_markers(filepath) for filepath in filepaths] if with_markers else None
	return file_ids, filenames, traces, states, markers


def get_data(path, file_id, with_markers=True):
	filepath = get_filepath(path, file_id)
	filename = get_filename(filepath)
	traces = get_traces(filepath)
	states = get_states(filepath)
	markers = get_markers(filepath) if with_markers else None
	return filename, traces, states, markers


def get_filepath(path, file_id):
	filenames = sorted([filename for filename in os.listdir(path) if not os.path.isdir(os.path.join(path, filename))])[::2]
	return os.path.join(path, '.'.join(filenames[file_id].split('.')[:-1]))


def get_filename(filepath):
	return os.path.basename(filepath)


def get_sampling_rate(filepath):
	with h5py.File(filepath + '.nwb', 'r') as file_handle:
		ts = file_handle['processing/behavioral state/behavioral state/timestamps'][:]
	return 1 / np.mean(ts[1:] - ts[:-1])


def get_timestamps(filepath):
	with h5py.File(filepath + '.nwb', 'r') as file_handle:
		timestamps = file_handle['processing/behavioral state/behavioral state/timestamps'][:]
	return timestamps


def get_traces(filepath):
	with h5py.File(filepath + '.nwb', 'r') as file_handle:
		data = pd.DataFrame(file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/data'][:])
		try:
			# data.columns = [str(n_id) for n_id in file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/rois'][:]]
			data.columns = ['' for _ in file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/rois'][:]]
		except ValueError:
			data.columns = ['?' for _ in range(data.shape[1])]
	return data


def get_states(filepath):
	with h5py.File(filepath + '.nwb', 'r') as file_handle:
		data = pd.DataFrame(file_handle[f'/processing/behavioral state/behavioral state/data'][:].round(0).astype(int))
		try:
			data.columns = ['background', 'still', 'walk', 'front-groom', 'back-groom', 'abdomen-move']  # labels from pickle file
		except ValueError:
			data.columns = ['?']
	return data


def get_markers(filepath):
	with open(filepath + '_dict.pkl', 'rb') as file:
		data = pickle.load(file)

	marker_names = list(data['marker_names'])

	# fix typo -.-
	for i in range(len(marker_names)):
		if marker_names[i].startswith('addomen'):
			marker_names[i] = marker_names[i].replace('addomen', 'abdomen')

	markers_x, markers_y = data['beh_markers_x'], data['beh_markers_y']
	markers_xy = np.stack((markers_x, markers_y), axis=2).reshape(markers_x.shape[0], -1)
	columns = [f'{name}_{ax}' for name in marker_names for ax in ['x', 'y']]
	behavior_df = pd.DataFrame(data=markers_xy, columns=columns)

	return behavior_df


# def get_states_pickle(filepath):
# 	with open(filepath, 'rb') as file:
# 		data = pickle.load(file)
# 	labels = data['beh_labels']
# 	names = data['state_labels'].item()
# 	states_alt = pd.DataFrame(data=(labels == labels.max(axis=1, keepdims=True)).astype(int),
# 							   columns=[names[i] for i in range(len(names))])
# 	return states_alt
