#region imports
import os, sys
import numpy as np
import pandas as pd
import json
sys.path.append(os.path.join(os.getcwd(), '../../'))
from utils import train_utils
from models import model_utils
#endregion


def get_plot_class(organism):
	match organism:
		case 'robot':
			return 'REV', 'reversal'
		case 'worm':
			return 'REV', 'reversal'
		case 'fly':
			return 'still', 'quiescent'
		case 'mouse':
			return 'run', 'running'
		case _:
			raise ValueError()


def get_model_folders(manual, model_variant, data_variant='gcamp', ll='ll00'):
	if manual:
		with open('../../hyperparams_manual.json', 'r') as f:
			reg_lambdas = json.load(f)
	else:
		with open('../../hyperparams.json', 'r') as f:
			reg_lambdas = json.load(f)['geometric'][ll]
	bom_folder = (f'{data_variant}_bom_{model_variant}_'
				  + '_'.join([f'{param.split('_')[0]}{reg_lambdas[model_variant]['bom'][param]:.2E}' for param in
							  reg_lambdas[model_variant]['bom'].keys()]))
	hgm_folder = (f'{data_variant}_hgm_{model_variant}_'
				  + '_'.join([f'{param.split('_')[0]}{reg_lambdas[model_variant]['hgm'][param]:.2E}' for param in
							  reg_lambdas[model_variant]['hgm'].keys()]))

	return bom_folder, hgm_folder


def get_significant_neurons(filename, model_variant):
	significant_neurons = pd.read_csv(f'../../out/data/significant_neurons/{filename}/{model_variant}_model.csv', index_col=0)
	return list(significant_neurons.columns)


def assemble_scoress(boms_scores_per_neuron, hgms_scores_per_neuron):
	neurons_labels = [scores_per_neuron[0].columns for scores_per_neuron in hgms_scores_per_neuron]

	boms_scores_per_neuron = [np.array([split.values[0] for split in scores]) for scores in boms_scores_per_neuron]
	hgms_scores_per_neuron = [np.array([split.values[0] for split in scores]) for scores in hgms_scores_per_neuron]

	neuron_scores = dict()
	boms_scores_per_neuron_avg = [np.mean(scores, axis=0) for scores in boms_scores_per_neuron]
	hgms_scores_per_neuron_avg = [np.mean(scores, axis=0) for scores in hgms_scores_per_neuron]
	for label_list, score_bom_list, score_hgm_list in zip(neurons_labels, boms_scores_per_neuron_avg, hgms_scores_per_neuron_avg):
		sort_indices = np.argsort(score_hgm_list)[::-1]
		label_list = label_list[sort_indices]
		score_bom_list = score_bom_list[sort_indices]
		score_hgm_list = score_hgm_list[sort_indices]

		for i in range(len(label_list)):
			label = label_list[i]
			if label != '' and label[-1] != '_':
				score_bom_new, score_hgm_new = np.array(score_bom_list[i]), np.array(score_hgm_list[i])
				if len(score_hgm_new.shape) == 0:
					score_bom_new = score_bom_new.reshape(1, 1)
					score_hgm_new = score_hgm_new.reshape(1, 1)
				if label not in neuron_scores.keys():
					neuron_scores[label] = (score_bom_new, score_hgm_new)
				else:
					score_bom_prev, score_hgm_prev = neuron_scores[label]
					neuron_scores[label] = (np.vstack([score_bom_prev, score_bom_new]), np.vstack([score_hgm_prev, score_hgm_new]))

	return neuron_scores


# single

def read_filename(organism, file_id, data_variant='gcamp'):
	match organism:
		case 'robot':
			from utils_robot import read_utils
			path = os.path.join(read_utils.get_data_path(), f'exported_data_gcamp')
			filename = read_utils.get_filename(read_utils.get_file_handle(path, file_id)).replace('worm', 'wyrm')
			return filename
		case 'worm':
			from utils_worm import read_utils
			path = os.path.join(read_utils.get_data_path(), f'exported_data_{data_variant}')
			return read_utils.get_filename(read_utils.get_file_handle(path, file_id))
		case 'fly':
			from utils_fly import read_utils
			path = os.path.join(read_utils.get_data_path(), 'brain')
			return read_utils.get_filename(read_utils.get_file_handle(path, file_id))
		case 'mouse':
			from utils_mouse import read_utils
			path = os.path.join(read_utils.get_data_path(), '6163622/sessions')
			filename = read_utils.get_filename(read_utils.get_filepath(path, file_id))
			return filename
		case _:
			raise ValueError


def read_data(organism, file_id, data_variant='gcamp', generate_data=None):
	match organism:
		case 'robot':
			from utils_robot import read_utils, generate_utils

			path = os.path.join(read_utils.get_data_path(), 'exported_data_gcamp')

			# read
			(filename_gcamp,
			 discrete_states, eigenworms) = read_utils.get_data(path, file_id)

			# clear nan values
			mask = eigenworms.notna().all(axis=1)

			eigenworms = eigenworms[mask]
			discrete_states = discrete_states[mask]

			# preprocess
			pass

			# generate
			weights, neurons, states = generate_data

			(filename,
			 modes,
			 traces_reconstructed,
			 traces_residual,
			 traces,
			 params,
			 params_manifold) = generate_utils.artificialize(weights, 'hgm',
															 filename_gcamp,
															 neurons,
															 eigenworms,
															 discrete_states[states],
															 n_dimensions=len(states))

			# # kill all information in behavior
			# eigenworms = pd.DataFrame(np.random.randn(*eigenworms.values.shape), columns=eigenworms.columns)
			# eigenworms = pd.DataFrame(data=np.zeros(eigenworms.values.shape), columns=eigenworms.columns)

			return filename, traces, eigenworms, discrete_states, (modes, traces_reconstructed, traces_residual, params,
																   params_manifold)

		case 'worm':
			from utils_worm import read_utils

			path = os.path.join(read_utils.get_data_path(), f'exported_data_{data_variant}')

			# read
			(filename,
			 traces,
			 continuous_behaviors,
			 discrete_states, eigenworms,
			 kymograph) = read_utils.get_data(path, file_id)

			# clear nan values
			mask = eigenworms.notna().all(axis=1)

			traces = traces[mask]
			continuous_behaviors = continuous_behaviors[mask]
			eigenworms = eigenworms[mask]
			discrete_states = discrete_states[mask]
			kymograph = kymograph[mask]

			# naive ID unknown neurons
			traces.columns = pd.Index([
				f'{file_id:02d}_{n_id:03d}' if empty or type(label) is not str else label for n_id, (label, empty) in
				enumerate(zip(traces.columns, traces.columns == ''))
			])

			# preprocess
			pass

			return filename, traces, eigenworms, discrete_states, (continuous_behaviors, kymograph)

		case 'fly':
			from utils_fly import read_utils

			raise NotImplementedError

		case 'mouse':
			from utils_mouse import read_utils

			path = os.path.join(read_utils.get_data_path(), '6163622/sessions')

			(filename,
			 traces,
			 continuous_behaviors,
			 motion_pcs,
			 motion_masks) = read_utils.get_data(path, file_id, with_whisking=True)

			# clear nan values
			# mask = motion_pcs[0].notna().all(axis=1)
			mask = continuous_behaviors.notna().all(axis=1)

			traces = traces[mask]
			motion_pcs[0] = motion_pcs[0][mask]
			motion_pcs[1] = motion_pcs[1][mask]
			continuous_behaviors = continuous_behaviors[mask]

			# naive ID unknown neurons
			traces.columns = pd.Index([
				f'{file_id:02d}_{n_id:03d}' if empty or type(label) is not str else label for n_id, (label, empty) in
				enumerate(zip(traces.columns, traces.columns == ''))
			])

			# preprocess
			def reconstruct_calcium(spikes, fs, decay=2.0):
				dt = 1.0 / fs
				gamma = np.exp(-dt / decay)
				trace = np.zeros(spikes.shape, dtype=np.float32)
				for t in range(1, spikes.shape[0]):
					trace[t, :] = gamma * trace[t - 1, :] + spikes[t, :]
				return trace

			sampling_rate = read_utils.get_sampling_rate(os.path.join(path, filename))
			traces = pd.DataFrame(reconstruct_calcium(traces.values, fs=sampling_rate), columns=traces.columns)
			classes = pd.DataFrame(
				continuous_behaviors[['running_speed']].values > np.quantile(continuous_behaviors[['running_speed']].values,
																			 0.05),
				columns=['run']
			)

			return filename, traces, motion_pcs[0], classes, (continuous_behaviors, motion_masks, motion_pcs[1])

		case _:
			raise ValueError


def read_models(filename, model_path, model_variant, data_variant='gcamp', manual=False, full=True):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant, ll='ll00')

	bom_params = train_utils.load_params(
		os.path.join(model_path, 'Full' if full else 'Splits', 'params', bom_folder), filename)
	hgm_params = train_utils.load_params(
		os.path.join(model_path, 'Full' if full else 'Splits', 'params', hgm_folder), filename)

	return bom_params, hgm_params


def read_predictions(filename, model_path, model_variant, data_variant='gcamp', manual=False, full=True):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant, ll='ll00')

	bom_spn, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', bom_folder), filename)
	hgm_spn, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', hgm_folder), filename)

	return ((bom_spn, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss),
			(hgm_spn, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss))


def read_scores(filename, model_path, model_variant, data_variant='gcamp', manual=False, full=False):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant, ll='ll00')

	bom_spn, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', bom_folder), filename)
	hgm_spn, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', hgm_folder), filename)

	return ((bom_spn, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss),
			(hgm_spn, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss))


def get_losses(filename, model_path, model_variant, data_variant='gcamp', manual=False, full=False):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant, ll='ll00')

	(bom_losses_train,
	 bom_man_losses_train,
	 bom_res_losses_train,
	 bom_mpe_losses_train,
	 bom_losses_test,
	 bom_man_losses_test,
	 bom_res_losses_test,
	 bom_mpe_losses_test) = train_utils.load_losses(
		os.path.join(model_path, 'Full' if full else 'Splits', 'history', bom_folder), filename)
	(hgm_losses_train,
	 hgm_man_losses_train,
	 hgm_res_losses_train,
	 hgm_mpe_losses_train,
	 hgm_losses_test,
	 hgm_man_losses_test,
	 hgm_res_losses_test,
	 hgm_mpe_losses_test) = train_utils.load_losses(
		os.path.join(model_path, 'Full' if full else 'Splits', 'history', hgm_folder), filename)

	return ((bom_losses_train,
			 bom_man_losses_train,
			 bom_res_losses_train,
			 bom_mpe_losses_train,
			 bom_losses_test,
			 bom_man_losses_test,
			 bom_res_losses_test,
			 bom_mpe_losses_test),
			(hgm_losses_train,
			 hgm_man_losses_train,
			 hgm_res_losses_train,
			 hgm_mpe_losses_train,
			 hgm_losses_test,
			 hgm_man_losses_test,
			 hgm_res_losses_test,
			 hgm_mpe_losses_test))


def postprocess_manifold(man, traces, params):
	manifolds = list()
	manifold_vars_expl = list()
	for s in range(len(params)):
		decode = lambda X: X @ model_utils.orthonormalize_qr(params[s]['model']['UN']).T
		manifold, var_explained = train_utils.postprocess_manifold(man[s].values, traces.values, decode,
																   scale=False, align=True)
		manifolds.append(pd.DataFrame(data=manifold, columns=man[s].columns))
		manifold_vars_expl.append(var_explained)

	return manifolds, manifold_vars_expl


# multiple

def read_filenames(organism, file_ids, data_variant='gcamp'):
	filenames = [read_filename(organism, file_id, data_variant) for file_id in file_ids]
	return filenames


def read_datas(organism, file_ids, generate_data=None):
	datas = [read_data(organism, file_id, generate_data) for file_id in file_ids]
	return list(zip(*datas))


def read_modelss(filenames, model_path, model_variant, data_variant='gcamp', manual=False, full=True):
	modelss = [read_models(filename, model_path, model_variant, data_variant, manual, full) for filename in filenames]
	return list(zip(*modelss))


def read_predictionss(filenames, model_path, model_variant, data_variant='gcamp', manual=False, full=True):
	predictionss = [read_predictions(filename, model_path, model_variant, data_variant, manual, full) for filename in filenames]
	boms_params, hgms_params = list(zip(*predictionss))
	return list(zip(*boms_params)), list(zip(*hgms_params))


def read_scoress(filenames, model_path, model_variant, data_variant='gcamp', manual=False, full=True):
	scoress = [read_scores(filename, model_path, model_variant, data_variant, manual, full) for filename in filenames]
	boms_spn, hgms_spn = list(zip(*scoress))
	return list(zip(*boms_spn)), list(zip(*hgms_spn))


def get_lossess(filenames, model_path, model_variant, data_variant='gcamp', manual=False, full=True):
	lossess = [get_losses(filename, model_path, model_variant, data_variant, manual, full) for filename in filenames]
	boms_losses, hgms_losses = list(zip(*lossess))
	return list(zip(*boms_losses)), list(zip(*hgms_losses))


def postprocess_manifolds(mans, tracess, paramss):
	manifolds = [postprocess_manifold(man, traces, params) for man, traces, params in zip(mans, tracess, paramss)]
	return manifolds
