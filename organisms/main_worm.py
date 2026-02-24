import worm
Worm = worm.Worm


def main():
	variants = ['ns', 'ss']

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
	worm_hyperparameters = {
		'max_epochs': 3000,
		'max_patience': 3000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 2,
		'reg_lambdas': dict()
	}

	file_ids_gcamp = list(range(0, 33+1))
	# file_ids_gcamp = [0,   # minrevs
	# 			1,   # minswitch
	# 			7,   # maxswitch
	# 			11,  # median
	# 			12,  # maxrevs
	# 			17,  # mean
	# 			19,  # paper
	# 			21]  # clean
	worm_gcamp = Worm(file_ids_gcamp, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='gcamp')
	# worm_gcamp.sweep()
	# worm_gcamp.train()
	# worm_gcamp.run()

	file_ids_gfp = list(range(0, 7+1))
	# file_ids_gfp = [2, 4, 6]
	worm_gfp = Worm(file_ids_gfp, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='gfp')
	# worm_gfp.sweep()
	# worm_gfp.train()
	# worm_gfp.run()

	file_ids_art = file_ids_gfp.copy()
	worm_art = Worm(file_ids_art, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='art')
	# worm_art.sweep()
	# worm_art.train()
	# worm_art.run()


if __name__ == '__main__':
	main()
