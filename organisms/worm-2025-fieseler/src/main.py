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
	manual_reg_lambdas = {
		'ns': {
			'bom': {
				'wb_reg_lambda': 0
			}
		}
	}
	worm_hyperparameters = {
		'max_epochs': 3000,
		'max_patience': 3000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 2,
		'reg_lambdas': manual_reg_lambdas
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	file_ids_gcamp = list(range(0, 33+1))
	# file_ids_gcamp = [19]

	worm_gcamp = Worm(file_ids_gcamp, names, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='gcamp')
	# worm_gcamp.sweep()
	worm_gcamp.train()
	worm_gcamp.run()


	# file_ids_gfp = list(range(0, 7+1))
	# # file_ids_gfp = [2]
	# worm_gfp = Worm(file_ids_gfp, names, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='gfp')
	# # worm_gfp.sweep()
	# worm_gfp.train()
	# worm_gfp.run()
	#
	# file_ids_art = file_ids_gfp.copy()
	# worm_art = Worm(file_ids_art, names, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='art')
	# # worm_art.sweep()
	# worm_art.train()
	# worm_art.run()


if __name__ == '__main__':
	main()
