import os
import pandas as pd
import h5py
import pynwb


def get_data_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/DATA/worm-2025-fieseler')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/data/scratch/neurobiology/zimmer/zimmermann/DATA/worm-2025-fieseler'


def get_datas(path, file_ids=None):
	if file_ids is None:
		file_ids = list(range(len(sorted(os.listdir(path)))))

	file_handles = [get_file_handle(path, file_id) for file_id in file_ids]
	filenames = [get_filename(file) for file in file_handles]
	traces = [get_traces(file) for file in file_handles]
	continuous_behaviors = [get_continuous_behaviors(file) for file in file_handles]
	discrete_states = [get_discrete_states(file) for file in file_handles]
	eigenworms = [get_eigenworms(file) for file in file_handles]
	kymographs = [get_kymograph(file) for file in file_handles]
	return file_ids, filenames, traces, continuous_behaviors, discrete_states, eigenworms, kymographs,


def get_data(path, file_id):
	file_handle = get_file_handle(path, file_id)
	filename = get_filename(file_handle)
	traces = get_traces(file_handle)
	continuous_behaviors = get_continuous_behaviors(file_handle)
	discrete_states = get_discrete_states(file_handle)
	eigenworms = get_eigenworms(file_handle)
	kymograph = get_kymograph(file_handle)
	return filename, traces, continuous_behaviors, discrete_states, eigenworms, kymograph


def get_file_handle(path, file_id):
	return h5py.File(os.path.join(path, sorted(os.listdir(path))[file_id]), 'r')


def get_filename(file_handle):
	filename = file_handle.filename.split('/')[-1].split('.')[0]
	if filename.endswith('_no_image_data'):
		filename = filename[:-len('_no_image_data')]
	return filename


def get_sampling_rate(file_handle):
	raise NotImplementedError('ToDo')


def get_timestamps(file_handle):
	raise NotImplementedError('ToDo')


def get_traces(file_handle):
	try:
		data = pd.DataFrame(file_handle[f'/processing/CalciumActivity/SignalDFoF/SignalCalciumImResponseSeries/data'][:],
							 columns=[n_id.decode('utf-8') for n_id in file_handle[f'/processing/CalciumActivity/NeuronIDs/labels'][:]])
	except KeyError:
		print(f'No neuronal trace data for {os.path.basename(file_handle.filename)}')
		data = None
	return data


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




# temp
def read_filename(path, file_id):
	file_handle = get_file_handle(path, file_id)
	filename = get_filename(file_handle)
	return filename


def check_neuron_contents(file_ids, traces, selection_n, sort=True):
	scores = list()
	for i in range(len(traces)):
		scores.append(list())
		for label in selection_n:
			if label in traces[i].columns:
				scores[i].append(label)

	if sort:
		ids_sorted = sorted(file_ids, key=lambda _id: len(scores[_id]), reverse=True)
		scores_sorted = [scores[_id] for _id in ids_sorted]
		return ids_sorted, scores_sorted

	else:
		return file_ids, scores
