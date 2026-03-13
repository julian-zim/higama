import os, sys
import numpy as np
import pandas as pd
import json
sys.path.append(os.path.join(os.getcwd(), '../../'))
from utils import train_utils
from models import model_utils


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


def read_data(organism, file_id):
	match organism:
		case 'worm':
			from utils_worm import read_utils

			path = os.path.join(read_utils.get_data_path(), 'exported_data_gcamp')

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


def read_models(model_path, filename, model_variant, manual, full=True):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant='gcamp', ll='ll00')

	bom_params = train_utils.load_params(os.path.join(model_path, 'Full' if full else 'Splits', 'params', bom_folder),
										 filename)
	hgm_params = train_utils.load_params(os.path.join(model_path, 'Full' if full else 'Splits', 'params', hgm_folder),
										 filename)

	return bom_params, hgm_params


def read_predictions(model_path, filename, model_variant, manual, full=True):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant='gcamp', ll='ll00')

	_, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', bom_folder), filename)
	_, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', hgm_folder), filename)

	return ((None, bom_man, bom_rec, bom_res, bom_pres, bom_resss, bom_bss),
			(None, hgm_man, hgm_rec, hgm_res, hgm_pres, hgm_resss, hgm_bss))


def read_scores(model_path, filename, model_variant, manual, full=False):
	bom_folder, hgm_folder = get_model_folders(manual, model_variant, data_variant='gcamp', ll='ll00')

	bom_spn, _, _, _, _, _, _ = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', bom_folder), filename)
	hgm_spn, _, _, _, _, _, _ = train_utils.load_predictions(
		os.path.join(model_path, 'Full' if full else 'Splits', 'predictions', hgm_folder), filename)

	return ((bom_spn, None, None, None, None, None, None),
			(hgm_spn, None, None, None, None, None, None))


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
