import os


def get_model_path(shuffle_id):
	local_path = os.path.join(os.path.expanduser('~'), 'Desktop/MODEL/mouse-control')
	if os.path.exists(local_path):
		return os.path.join(local_path, f'shuffle_{shuffle_id}')
	else:
		return os.path.join('/lisc/data/scratch/neurobiology/zimmer/zimmermann/MODEL/mouse-control', f'shuffle_{shuffle_id}')
