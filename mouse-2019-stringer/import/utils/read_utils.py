import os
import numpy as np
import pandas as pd
import scipy as sp


def get_data_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/DATA')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/scratch/neurobiology/zimmer/zimmermann/DATA/'


def get_data(path, ids=None):
	filenames_all = sorted(os.listdir(path))
	if ids is not None:
		file_ids = ids
		filenames = [filenames_all[file_id] for file_id in file_ids]
	else:
		file_ids = list(range(len(filenames_all)))
		filenames = filenames_all

	traces = [get_traces(os.path.join(path, filenames_all[file_id])) for file_id in file_ids]
	behaviors = [get_behaviors(os.path.join(path, filenames_all[file_id])) for file_id in file_ids]

	return file_ids, filenames, traces, [b[0] for b in behaviors], [b[1] for b in behaviors]


def get_traces(filepath):
	traces = sp.io.loadmat(filepath, variable_names=['Fsp'])['Fsp'].T
	traces_df = pd.DataFrame(data=traces)
	return traces_df


def get_behaviors(filepath):
	behaviors = sp.io.loadmat(filepath, variable_names=['beh'])['beh']

	motion_pcs = get_motion_pcs(behaviors)
	continuous_behaviors = get_continuous_behaviors(behaviors)

	# TODO: get whisking signal

	return motion_pcs, continuous_behaviors


def get_motion_pcs(behaviors):
	face = behaviors['face'][0][0]['motionSVD'][0][0]
	face_df = pd.DataFrame(data=face, columns=[f'PC{i+1}' for i in range(face.shape[1])])

	whisker = behaviors['whisker'][0][0]['motionSVD'][0][0]
	whisker_df = pd.DataFrame(data=whisker, columns=[f'PC{i+1}' for i in range(whisker.shape[1])])

	eye = behaviors['eye'][0][0]['motionSVD'][0][0]
	eye_df = pd.DataFrame(data=eye, columns=[f'PC{i+1}' for i in range(eye.shape[1])])

	return [face_df, whisker_df, eye_df]


def get_continuous_behaviors(behaviors):
	area = behaviors['pupil'][0][0]['area'][0][0]
	com = behaviors['pupil'][0][0]['com'][0][0]
	speed = behaviors['runSpeed'][0][0]
	# angle = behaviors['runAngle'][0][0]

	behaviors_df = pd.DataFrame(data=np.concat([area, com, speed], axis=1), columns=['pupil_area', 'pupil_com_x', 'pupil_com_y', 'running_speed'])
	# angle_df = pd.DataFrame(data=angle, columns=[])

	return behaviors_df
