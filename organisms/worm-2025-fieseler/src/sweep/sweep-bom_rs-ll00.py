#region imports
import sys
import os

import jax

import sweep_utils_rs

sys.path.append(os.path.join(os.getcwd(), '../../'))
from models import bom_rs as bom
from models import hgm_rs as hgm


jax.config.update("jax_enable_x64", True)
#endregion


default_hyperparameters = {
	'max_epochs': 3000,
	'max_patience': 3000 / 4,
	'learning_rate': 1E-0,
	'dimensions': 2,
	'ccadiag_lambda': 0.6,
	'ccaunit_lambda': 1.0 - 0.6,
	'loss_man_lambda': 0.0,
	'loss_res_lambda': 1.0 - 0.0,
	'vb_reg_lambda': 0.0,
	'wb_reg_lambda': 0.0,
	'wm_reg_lambda': -1.0,  # unused by the bom
}

if __name__ == '__main__':
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
			'wb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
		},
		'run_cap': 400,
	}
	prefix = f'{os.path.basename(__file__).split('.')[0].replace('sweep-', '')}'
	sweep_utils_rs.sweep(prefix, bom, sweep_config, default_hyperparameters, log_path)
