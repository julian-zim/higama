import sys, os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import fly
Fly = fly.Fly


def main():
	with open('../sweep-config.json', 'r') as f:
		fly_sweep_parameters = json.load(f)
	fly_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 8,
		'reg_lambdas': 'manual'
	}

	names = ['hgm']
	variants = ['ns']

	# file_ids = list(range(0, 12+1))
	file_ids = [0]

	fly = Fly(file_ids, names, variants, fly_hyperparameters, fly_sweep_parameters)
	fly.sweep()
	# fly.train()
	# fly.run()


if __name__ == '__main__':
	main()
