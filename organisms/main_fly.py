import fly
Fly = fly.Fly


def main():
	variants = ['ns', 'ss']

	fly_sweep_parameters = {
		'ns': {
			'bom': {
				'parameters': {
					'wb_reg_lambda': {'distribution': 'log_uniform_values', 'min': 1E-9, 'max': 1E-0},
				},
				'run_cap': 150,
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
	fly_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 4,
		'reg_lambdas': dict()
	}

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

	fly = Fly(file_ids, variants, fly_hyperparameters, fly_sweep_parameters)
	# fly.sweep()
	# fly.train()
	# fly.run()


if __name__ == '__main__':
	main()
