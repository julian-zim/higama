import fly
Fly = fly.Fly


def main():
	fly_sweep_parameters = {}
	fly_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 8,
		'reg_lambdas': 'manual'
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	# file_ids = list(range(0, 12+1))
	file_ids = [0]

	fly = Fly(file_ids, names, variants, fly_hyperparameters, fly_sweep_parameters)
	# fly.sweep()
	fly.train()
	fly.run()
	fly.train_on_all()
	fly.run_on_all()


if __name__ == '__main__':
	main()
