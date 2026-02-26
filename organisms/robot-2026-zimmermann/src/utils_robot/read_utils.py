import os
import pandas as pd
import h5py
import pynwb


def get_model_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/MODEL/robot-2026-zimmermann')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/data/scratch/neurobiology/zimmer/zimmermann/MODEL/robot-2026-zimmermann'


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
	discrete_states = [get_discrete_states(file) for file in file_handles]
	eigenworms = [get_eigenworms(file) for file in file_handles]
	return file_ids, filenames, discrete_states, eigenworms


def get_data(path, file_id):
	file_handle = get_file_handle(path, file_id)
	filename = get_filename(file_handle)
	discrete_states = get_discrete_states(file_handle)
	eigenworms = get_eigenworms(file_handle)
	return filename, discrete_states, eigenworms


def get_file_handle(path, file_id):
	return h5py.File(os.path.join(path, sorted(os.listdir(path))[file_id]), 'r')


def get_filename(file_handle):
	filename = file_handle.filename.split('/')[-1].split('.')[0]
	if filename.endswith('_no_image_data'):
		filename = filename[:-len('_no_image_data')]
	return filename


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
