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


def get_data(path, ids=None, with_whisking=True, median_center=True):
	filenames_all = sorted(os.listdir(path))
	if ids is not None:
		file_ids = ids
		filenames = [filenames_all[file_id] for file_id in file_ids]
	else:
		file_ids = list(range(len(filenames_all)))
		filenames = filenames_all

	traces = [get_traces(os.path.join(path, filenames_all[file_id]), median_center) for file_id in file_ids]
	behaviors = [get_behaviors(os.path.join(path, filenames_all[file_id]), with_whisking) for file_id in file_ids]

	return file_ids, filenames, traces, *([b[i] for b in behaviors] for i in range(3))


def get_traces(filepath, median_center=False):
	traces = sp.io.loadmat(filepath, variable_names=['Fsp'])['Fsp'].T
	if median_center:
		traces -= np.median(traces, axis=0)
	traces_df = pd.DataFrame(data=traces)
	return traces_df


def get_behaviors(filepath, with_whisking=False):
	behaviors = sp.io.loadmat(filepath, variable_names=['beh'])['beh']

	continuous_behaviors = get_continuous_behaviors(behaviors)
	motion_pcs = get_motion_pcs(behaviors)
	motion_masks = get_motion_masks(behaviors)
	if with_whisking:
		whisking = compute_whisking(motion_pcs[1].values, motion_masks[1][0].values)
		continuous_behaviors['whisking'] = whisking

	return continuous_behaviors, motion_pcs, motion_masks


def get_continuous_behaviors(behaviors):
	area = behaviors['pupil'][0][0]['area'][0][0]
	com = behaviors['pupil'][0][0]['com'][0][0]
	speed = behaviors['runSpeed'][0][0]
	# angle = behaviors['runAngle'][0][0]

	behaviors_df = pd.DataFrame(data=np.concat([com, area, speed], axis=1), columns=['pupil_com_x', 'pupil_com_y', 'pupil_area', 'running_speed'])
	# angle_df = pd.DataFrame(data=angle, columns=[])

	return behaviors_df


def get_motion_pcs(behaviors):
	face = behaviors['face'][0][0]['motionSVD'][0][0]
	face_df = pd.DataFrame(data=face, columns=[f'PC{i+1}' for i in range(face.shape[1])])

	whisker = behaviors['whisker'][0][0]['motionSVD'][0][0]
	whisker_df = pd.DataFrame(data=whisker, columns=[f'PC{i+1}' for i in range(whisker.shape[1])])

	eye = behaviors['eye'][0][0]['motionSVD'][0][0]
	eye_df = pd.DataFrame(data=eye, columns=[f'PC{i+1}' for i in range(eye.shape[1])])

	return [face_df, whisker_df, eye_df]


def get_motion_masks(behaviors):
	face_2d = behaviors['face'][0][0]['motionMask'][0][0].T
	face_aspect_ratio = face_2d.shape[1], face_2d.shape[2]
	face = face_2d.reshape(-1, face_aspect_ratio[0] * face_aspect_ratio[1])
	face_df = pd.DataFrame(data=face, columns=[f'L{i+1}' for i in range(face.shape[1])])

	whisker_2d = behaviors['whisker'][0][0]['motionMask'][0][0].T
	whisker_aspect_ratio = whisker_2d.shape[1], whisker_2d.shape[2]
	whisker = whisker_2d.reshape(-1, whisker_aspect_ratio[0] * whisker_aspect_ratio[1])
	whisker_df = pd.DataFrame(data=whisker, columns=[f'L{i+1}' for i in range(whisker.shape[1])])

	# eye = behaviors['eye'][0][0]['motionSVD'][0][0]
	# eye_df = pd.DataFrame(data=eye, columns=[f'PC{i+1}' for i in range(eye.shape[1])])

	return [(face_df, face_aspect_ratio), (whisker_df, whisker_aspect_ratio)]


def compute_whisking(whisk_weights, whisk_loadings):
	whisking_roi = whisk_weights @ whisk_loadings
	return np.sum(whisking_roi, axis=1)
