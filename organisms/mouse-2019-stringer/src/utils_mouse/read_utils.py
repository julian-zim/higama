import os
import numpy as np
import pandas as pd
import scipy as sp


def get_data_path():
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/DATA/mouse-2019-stringer')
	if os.path.exists(local_path):
		return local_path
	else:
		return '/lisc/data/scratch/neurobiology/zimmer/zimmermann/DATA/mouse-2019-stringer'


def get_datas(path, file_ids=None, with_whisking=True):
	if file_ids is None:
		file_ids = list(range(len(sorted(os.listdir(path)))))

	filepaths = [get_filepath(path, file_id) for file_id in file_ids]
	filenames = [get_filename(filepath) for filepath in filepaths]
	traces = [get_traces(filepath) for filepath in filepaths]
	behaviors = [get_behaviors(filepath, with_whisking) for filepath in filepaths]
	return file_ids, filenames, traces, *([b[i] for b in behaviors] for i in range(3))


def get_data(path, file_id, with_whisking=True):
	filepath = get_filepath(path, file_id)
	filename = get_filename(filepath)
	traces = get_traces(filepath)
	behaviors = get_behaviors(filepath, with_whisking)
	return filename, traces, *behaviors


def get_filepath(path, file_id):
	filenames = sorted([filename for filename in os.listdir(path) if not os.path.isdir(os.path.join(path, filename))])
	return os.path.join(path, '.'.join(filenames[file_id].split('.')[:-1]))


def get_filename(filepath):
	return os.path.basename(filepath)


def get_sampling_rate(filepath):
	planes = sp.io.loadmat(filepath, variable_names=['stat'])['stat']['iplane'].max().item()
	return {10: 3, 12: 2.5}[planes]


def get_traces(filepath):
	traces = pd.DataFrame(data=sp.io.loadmat(filepath, variable_names=['Fsp'])['Fsp'].T)
	return traces


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
