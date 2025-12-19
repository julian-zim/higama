import os
import numpy as np
import pandas as pd
import pickle
import h5py
import pynwb


def get_data_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/DATA')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/scratch/neurobiology/zimmer/zimmermann/DATA/'


def get_data(path, ids=None):
	# filenames_all = sorted('_'.join(filename.split('.')[0].split('_')[:-1]) for filename in os.listdir(os.path.join(path, 'pickles')))
	filenames_all = sorted(os.listdir(os.path.join(path, 'nwb')))
	if ids is not None:
		file_ids = ids
		filenames = [filenames_all[file_id] for file_id in file_ids]
	else:
		file_ids = list(range(len(filenames_all)))
		filenames = filenames_all

	traces = [get_traces(os.path.join(path, 'nwb', filenames[file_ids.index(file_id)])) for file_id in file_ids]
	states = [get_states(os.path.join(path, 'nwb', filenames[file_ids.index(file_id)])) for file_id in file_ids]
	markers = [get_markers(os.path.join(path, 'pickle', f'{filenames[file_ids.index(file_id)].split('.')[0]}_dict.pkl')) for file_id in file_ids]

	return file_ids, filenames, traces, states, markers


def get_traces(filepath):
	with h5py.File(filepath, 'r') as file_handle:
		data = pd.DataFrame(file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/data'][:])
		try:
			# data.columns = [str(n_id) for n_id in file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/rois'][:]]
			data.columns = ['' for _ in file_handle[f'/processing/ophys/DfOverF/RoiResponseSeries/rois'][:]]
		except ValueError:
			data.columns = ['?' for _ in range(data.shape[1])]
	return data


def get_states(filepath):
	with h5py.File(filepath, 'r') as file_handle:
		data = pd.DataFrame(file_handle[f'/processing/behavioral state/behavioral state/data'][:].round(0).astype(int))
		try:
			data.columns = ['background', 'still', 'walk', 'front-groom', 'back-groom', 'abdomen-move']  # labels from pickle file
		except ValueError:
			data.columns = ['?']
	return data


def get_markers(filepath):
	with open(filepath, 'rb') as file:
		data = pickle.load(file)

	marker_names = list(data['marker_names'])

	# fix typo
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
