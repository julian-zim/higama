import mouse_control
Mouse = mouse_control.Mouse


def main():
	mouse_sweep_parameters = {}
	mouse_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 16,
		'reg_lambdas': 'swept'
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	file_ids = [0, 1, 3, 4, 5, 6, 7, 8]
	# file_ids = [2]

	for shuffle_id in range(100):
		print(f'\n=================== SHUFFLE {shuffle_id} ===================')
		mouse = Mouse(shuffle_id, file_ids, names, variants, mouse_hyperparameters, mouse_sweep_parameters)
		mouse.do(action='train')
		mouse.do(action='run')


if __name__ == '__main__':
	main()
