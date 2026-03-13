import sys, os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import worm
Worm = worm.Worm


def main():
	with open('../sweep-config.json', 'r') as f:
		worm_sweep_parameters = json.load(f)
	worm_hyperparameters = {
		'max_epochs': 3000,
		'max_patience': 3000 / 4,
		'learning_rate': 1E-0,
		'dimensions': 2,
		'reg_lambdas': 'swept'
	}

	names = ['hgm']
	variants = ['ns']

	file_ids = list(range(0, 33+1))
	# file_ids = [19]

	worm = Worm(file_ids, names, variants, worm_hyperparameters, worm_sweep_parameters)
	worm.do(action='sweep')


if __name__ == '__main__':
	main()
