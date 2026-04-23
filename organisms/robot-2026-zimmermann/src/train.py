import robot
Robot = robot.Robot


def main():
	robot_sweep_parameters = {}
	robot_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 1,
		'reg_lambdas': 'manual'
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	# file_ids = list(range(0, 33+1))
	# file_ids = [19, 20, 21, 22, 23]
	file_ids = [19]

	weights = (0.25, -0.25, 1)

	neuron_labels = [f'N{i}' for i in range(100)]
	# states = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']
	states = ['REV']

	robot = Robot(weights, neuron_labels, states,
				  file_ids, names, variants, robot_hyperparameters, robot_sweep_parameters, data_variant='gcamp')
	# robot.do(action='train')
	# robot.do(action='run')
	robot.do(action='train', full=True)
	robot.do(action='run', full=True)


if __name__ == '__main__':
	main()
