import sys, os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import mouse
Mouse = mouse.Mouse


def main():
	with open('../sweep-config.json', 'r') as f:
		mouse_sweep_parameters = json.load(f)
	mouse_hyperparameters = {
		'max_epochs': 4000,
		'max_patience': 4000 / 1,
		'learning_rate': 1E-0,
		'dimensions': 16,
		'reg_lambdas': 'swept'
	}

	names = ['hgm']
	variants = ['ss']

	# file_ids = list(range(0, 8+1))
	file_ids = [2]

	mouse = Mouse(file_ids, names, variants, mouse_hyperparameters, mouse_sweep_parameters)
	mouse.sweep()
	# mouse.train()
	# mouse.run()


if __name__ == '__main__':
	main()
