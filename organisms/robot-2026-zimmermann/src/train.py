import robot
Robot = robot.Robot


def main():
	robot_sweep_parameters = {}
	robot_hyperparameters = {
		'max_epochs': 3000,
		'max_patience': 3000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 2,
		'reg_lambdas': 'manual'
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
