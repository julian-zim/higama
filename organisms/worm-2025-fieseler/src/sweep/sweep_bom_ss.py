import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import worm
Worm = worm.Worm


def main():
	worm_sweep_parameters = {
		'ns': {
			'bom': {
				'parameters': {
					'wb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
				},
				'run_cap': 100,
			},
			'hgm': {
				'parameters': {
					'wb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
					'wm_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
				},
				'run_cap': 400,
			}
		},
		'ss': {
			'bom': {
				'parameters': {
					'vb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
				},
				'run_cap': 600,
			},
			'hgm': {
				'parameters': {
					'vb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
					'wm_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
				},
				'run_cap': 1000,
			}
		}
	}
	manual_reg_lambdas = {}
	worm_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 8,
		'reg_lambdas': manual_reg_lambdas
	}

	names = ['bom']
	variants = ['ss']

	# file_ids = list(range(0, 33+1))
	file_ids = [19]

	worm = Worm(file_ids, names, variants, worm_hyperparameters, worm_sweep_parameters)
	worm.sweep()
	# worm.train()
	# worm.run()


if __name__ == '__main__':
	main()
