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


def get_data(path, slice=None):
	hdf5files = get_hdf5_files(path) if slice is None else get_hdf5_files(path)[slice[0]:slice[1]]
	filenames = [get_filename(file) for file in hdf5files]
	continuous_behaviors = [get_continuous_behaviors(file) for file in hdf5files]
	discrete_states = [get_discrete_states(file) for file in hdf5files]
	eigenworms = [get_eigenworms(file) for file in hdf5files]
	kymographs = [get_kymograph(file) for file in hdf5files]
	traces = [get_neuronal_traces(file) for file in hdf5files]
	return filenames, continuous_behaviors, discrete_states, eigenworms, kymographs, traces


def get_hdf5_files(path):
	return [h5py.File(os.path.join(path, filename), 'r') for filename in os.listdir(path)]


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
