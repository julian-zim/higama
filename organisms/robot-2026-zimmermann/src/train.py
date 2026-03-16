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

	# file_ids = list(range(0, 33+1))
	file_ids = [19, 20, 21, 22, 23]

	weights = (0.1, -0.25, 1.)

	neuron_labels = [f'N{i}' for i in range(50)]
	states = ['REV', 'VENTRAL_TURN', 'DORSAL_TURN', 'PAUSE']

	robot = Robot(weights, neuron_labels, states,
				  file_ids, names, variants, robot_hyperparameters, robot_sweep_parameters, data_variant='gcamp')
	robot.do(action='train')
	robot.do(action='run')
	robot.do(action='train', full=True)
	robot.do(action='run', full=True)


if __name__ == '__main__':
	main()
