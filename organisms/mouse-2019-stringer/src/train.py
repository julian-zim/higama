import mouse
Mouse = mouse.Mouse


def main():
	mouse_sweep_parameters = {
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
	mouse_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 8,
		'reg_lambdas': 'manual'
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	# file_ids = list(range(0, 8+1))
	file_ids = [2]

	mouse = Mouse(file_ids, names, variants, mouse_hyperparameters, mouse_sweep_parameters)
	# mouse.sweep()
	mouse.train()
	mouse.run()


if __name__ == '__main__':
	main()
