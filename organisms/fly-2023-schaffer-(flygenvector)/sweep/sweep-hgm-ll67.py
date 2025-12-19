#region imports
import sys
import os

import jax

import sweep_utils

sys.path.append(os.path.join(os.getcwd(), '../'))
from models import bom
from models import hgm
#endregion


default_hyperparameters = {
	'max_epochs': 3000,
	'max_patience': 3000 / 4,
	'learning_rate': 1E-0,
	'dimensions': 1,
	'ccadiag_lambda': 0.6,
	'ccaunit_lambda': 1.0 - 0.6,
	'loss1_lambda': 0.67,
	'loss2_lambda': 1.0 - 0.67,
	'w1_reg_lambda': 0.0,	#1E-3,
	'w2_reg_lambda': 0.0,	#1E-5,
}

if __name__ == '__main__':
	log_path = '/lisc/scratch/neurobiology/zimmer/zimmermann/Sweeps/'
	if not (os.path.exists(log_path) and os.access(log_path, os.W_OK)):
		log_path = './out/'

	print('Visible devices: ', jax.devices())
	sweep_config = {
		'method': 'bayes',
		'metric': {
			'name': 'mpe_loss',
			'goal': 'minimize',
		},
		'parameters': {
			'w1_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
			'w2_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
		},
		'run_cap': 1000,
	}
	prefix = f'{os.path.basename(__file__).split('.')[0].replace('sweep-', '')}'
	sweep_utils.sweep(prefix, hgm, sweep_config, default_hyperparameters, log_path)
