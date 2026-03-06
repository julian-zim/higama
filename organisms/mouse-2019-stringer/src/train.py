import mouse
Mouse = mouse.Mouse


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

	# file_ids = list(range(0, 8+1))
	file_ids = [2]

	mouse = Mouse(file_ids, names, variants, mouse_hyperparameters, mouse_sweep_parameters)
	# mouse.sweep()
	mouse.train()
	mouse.run()


if __name__ == '__main__':
	main()
