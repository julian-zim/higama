#region imports
import sys
import os
import subprocess

import jax


# jax.config.update("jax_enable_x64", True)
#endregion


manual_hyperparameters = {
	'max_epochs': 4000,
	'max_patience': 4000 / 1,
	'learning_rate': 1E-0,
	'dimensions': 2,
	'ccadiag_lambda': 0.6,
	'ccaunit_lambda': 1.0 - 0.6,
	'loss_man_lambda': 0.0,
	'loss_res_lambda': 1.0 - 0.0,
	'vb_reg_lambda': 0.0,
	'wm_reg_lambda': -1.0,  # unused by the bom
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
			'vb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
		},
		'run_cap': 600,
	}
	prefix = f'{os.path.basename(__file__).split('.')[0].replace('-sweep', '')}'

	file_ids = list(range(0, 12+1))
	# file_ids = [0,	# paper
	# 			1,	# nice states
	# 			2,	# nice manifold
	# 			5,	# variety
	# 			6,	# nice quiescence
	# 			7,	# nice states
	# 			9,	# nice heatmap
	# 			11]	# variety
	# file_ids = [0]

	for file_id in file_ids:
		subprocess.run(
			[
				sys.executable,
				"-c",
				f"""
import sys, os

import sweep_model_ss_utils

sys.path.append(os.path.join(os.getcwd(), '../../../../'))
from models import bom_ss as bom

sweep_model_ss_utils.sweep(
	{file_id},
	{prefix!r},
	bom,
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
