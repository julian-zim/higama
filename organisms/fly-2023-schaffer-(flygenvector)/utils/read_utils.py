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
	filenames_all = sorted(os.listdir(path))
	if ids is not None:
		file_ids = ids
		filenames = [filenames_all[file_id] for file_id in file_ids]
	else:
		file_ids = list(range(len(filenames_all)))
		filenames = filenames_all

	dicts = [get_pickle_file(os.path.join(path, filenames_all[file_id])) for file_id in file_ids]

	return file_ids, filenames, dicts


def get_pickle_file(filepath):
	with open(filepath, 'rb') as file:
		dictionary = pickle.load(file)
	return dictionary


# def get_hdf5_file(filepath):
# 	file_handle = h5py.File(filepath, 'r')
# 	return file_handle
