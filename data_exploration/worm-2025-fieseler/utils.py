import os
import pandas as pd
import pynwb
import h5py


def get_hdf5_files(path):
	return [h5py.File(path + filename, 'r') for filename in os.listdir(path)]


def get_filename(file_handle):
	return file_handle.filename.split('/')[-1].split('.')[0]


def get_continuous_behaviors(file_handle):
	try:
		data = pd.DataFrame({key: file_handle[f'/processing/Behavior/continuous_behaviors/{key}/data'][:] for key in file_handle[f'/processing/Behavior/continuous_behaviors'].keys()})
	except KeyError:
		print(f'No continuous behavior data for {os.path.basename(file_handle.filename)}')
		data = None
	return data


def get_discrete_states(file_handle):
	try:
		data = pd.DataFrame({key: file_handle[f'/processing/Behavior/discrete_states/{key}/data'][:] for key in file_handle[f'/processing/Behavior/discrete_states'].keys()})
	except KeyError:
		print(f'No discrete state data for {os.path.basename(file_handle.filename)}')
		data = None
	return data


def get_eigenworms(file_handle):
	try:
		data = pd.DataFrame({key: file_handle[f'/processing/Behavior/eigenworms/{key}/data'][:] for key in file_handle[f'/processing/Behavior/eigenworms'].keys()})
	except KeyError:
		print(f'No eigenworm data for {os.path.basename(file_handle.filename)}')
		data = None
	return data


def get_kymograph(file_handle):
	try:
		data = pd.DataFrame({key: file_handle[f'/processing/Behavior/kymograph/{key}/data'][:] for key in file_handle[f'/processing/Behavior/kymograph'].keys()})
	except KeyError:
		print(f'No kymograph data for {os.path.basename(file_handle.filename)}')
		data = None
	return data


def get_neuronal_traces(file_handle):
	try:
		data = pd.DataFrame(file_handle[f'/processing/CalciumActivity/SignalDFoF/SignalCalciumImResponseSeries/data'][:],
							 columns=[n_id.decode('utf-8') for n_id in file_handle[f'/processing/CalciumActivity/NeuronIDs/labels'][:]])
	except KeyError:
		print(f'No neuronal trace data for {os.path.basename(file_handle.filename)}')
		data = None
	return data


def get_variance_explained(X, X_rec):
	X_res = X - X_rec

	X_var = X.var().sum()
	X_res_var = X_res.var().sum()

	X_var_ratio = 1 - X_res_var / X_var

	return X_var_ratio


def get_good_neuron_labels():
	return [
		"VB02",
		"DB01",
		"DD01",
		"RMDDR",
		"RMDDL",
		"SMDDR",
		"SMDDL",
		"RMED",
		"RMEV",
		"RMDVR",
		"DB02",
		"SIADR",
		"URXL",
		"URYVR",
		"RID",
	]
