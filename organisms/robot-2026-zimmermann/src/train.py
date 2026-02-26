import robot
Robot = robot.Robot


def main():
	robot_sweep_parameters = {
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
				'wb_reg_lambda': 1E-9,
			},
			'hgm': {
				'wb_reg_lambda': 1E-9,
				'wm_reg_lambda': 1E-9,
			}
		},
		'ss': {
			'bom': {
				'vb_reg_lambda': 1E-9,
			},
			'hgm': {
				'vb_reg_lambda': 1E-9,
				'wm_reg_lambda': 1E-9,
			}
		}
	}
	robot_hyperparameters = {
		'max_epochs': 3000,
		'max_patience': 3000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 2,
		'reg_lambdas': manual_reg_lambdas
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	# file_id = list(range(0, 33+1))
	file_id = [19]

	weights = (0.1, -0.25)
	n_neurons = 6
	# states = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']
	states = ['REV', 'VENTRAL_TURN']

	robot = Robot(weights, n_neurons, states, file_id, names, variants, robot_hyperparameters, robot_sweep_parameters, data_variant='gcamp')
	# robot.sweep()
	robot.train()
	robot.run()


if __name__ == '__main__':
	main()
