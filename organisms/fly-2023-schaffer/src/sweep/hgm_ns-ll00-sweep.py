#region imports
import sys
import os
import subprocess

import jax


# jax.config.update("jax_enable_x64", True)
#endregion


manual_hyperparameters = {
	'max_epochs': 4000,
	'max_patience': 4000 / 4,
	'learning_rate': 1E-0,
	'dimensions': 2,
	'ccadiag_lambda': 0.6,
	'ccaunit_lambda': 1.0 - 0.6,
	'loss_man_lambda': 0.0,
	'loss_res_lambda': 1.0 - 0.0,
	'wb_reg_lambda': 0.0,
	'wm_reg_lambda': 0.0,
}


def main():
	log_path = '/lisc/data/scratch/neurobiology/zimmer/zimmermann/Sweeps/'
	if not (os.path.exists(log_path) and os.access(log_path, os.W_OK)):
		log_path = '../../out/'

	print('Visible devices: ', jax.devices())
	sweep_config = {
		'method': 'bayes',
		'metric': {
			'name': 'mpe_loss',
			'goal': 'minimize',
		},
		'parameters': {
			'wb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
			'wm_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
		},
		'run_cap': 400,
	}
	prefix = f'{os.path.basename(__file__).split('.')[0].replace('-sweep', '')}'

	file_ids = list(range(0, 33+1))
	# file_ids = [0,   # minrevs
	# 			1,   # minswitch
	# 			7,   # maxswitch
	# 			11,  # median
	# 			12,  # maxrevs
	# 			17,  # mean
	# 			19,  # paper
	# 			21]  # clean
	# file_ids = [19]

	for file_id in file_ids:
		print(f'Running sweep for file with ID {file_id}')
		subprocess.run(
			[
				sys.executable,
				"-c",
				f"""
import sys, os

import sweep_model_ns_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from models import hgm

sweep_model_ns_utils.sweep(
	{file_id},
	{prefix!r},
	hgm,
	{sweep_config!r},
	{manual_hyperparameters!r},
	{log_path!r}
)
"""
			],
			check=True,
			env=os.environ,
		)


if __name__ == '__main__':
	main()
