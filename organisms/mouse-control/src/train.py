import mouse_control
Mouse = mouse_control.Mouse


def main():
	mouse_sweep_parameters = {}
	mouse_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 16,
		'reg_lambdas': 'swept'
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	file_ids = list(range(0, 8+1))
	# file_ids = [2]

	for file_id in file_ids:
		for shuffle_id in range(100):
			print(f'\n=================== SHUFFLE {shuffle_id} ===================')
			mouse = Mouse(shuffle_id, [file_id], names, variants, mouse_hyperparameters, mouse_sweep_parameters)
			mouse.do(action='train')
			mouse.do(action='run')


if __name__ == '__main__':
	main()
