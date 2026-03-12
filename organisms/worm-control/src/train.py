import worm_control
Worm = worm_control.Worm


def main():
	worm_sweep_parameters = {}
	worm_hyperparameters = {
		'max_epochs': 3000,
		'max_patience': 3000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 2,
		'reg_lambdas': 'swept'
	}

	names = ['bom', 'hgm']
	variants = ['ns', 'ss']

	# file_ids_gcamp = list(range(0, 33+1))
	file_ids_gcamp = [19]

	for shuffle_id in range(100):
		print(f'\n=================== SHUFFLE {shuffle_id} ===================')
		worm_gcamp = Worm(shuffle_id, file_ids_gcamp, names, variants, worm_hyperparameters, worm_sweep_parameters, data_variant='gcamp')
		worm_gcamp.train()
		worm_gcamp.run()


if __name__ == '__main__':
	main()
