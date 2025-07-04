import math
import numpy as np
import pandas as pd
from matplotlib import patches, pyplot as plt


def plot_performance_plot(neuron_scores, neuron_scores_gfp=None, limits=None, save=False):
	if neuron_scores_gfp is None: neuron_scores_gfp = dict()

	threshold = 0.95

	plt.figure(figsize=(20, 20))
	plt.xlabel('GLM r2 score (median)', fontsize=20, labelpad=50)
	plt.ylabel('HGM r2 score (median)', fontsize=20)

	if limits is None:
		gcamp_x = np.array([score[0] for scores in neuron_scores.values() for score in scores])
		if len(neuron_scores_gfp) > 0:
			gfp_x = np.array([score[0] for scores in neuron_scores_gfp.values() for score in scores])
			full_x = np.array(np.concatenate([gcamp_x, gfp_x]))
		else:
			full_x = np.array(gcamp_x)

		gcamp_y = np.array([score[1] for scores in neuron_scores.values() for score in scores])
		if len(neuron_scores_gfp) > 0:
			gfp_y = np.array([score[1] for scores in neuron_scores_gfp.values() for score in scores])
			full_y = np.array(np.concatenate([gcamp_y, gfp_y]))
		else:
			full_y = np.array(gcamp_y)

		min_x, max_x = full_x.min(), full_x.max()
		min_y, max_y = full_y.min(), full_y.max()
	else:
		min_x, max_x, min_y, max_y = limits
	xstep = (max_x - min_x) / 10
	ystep = (max_y - min_y) / 10
	min_x -= 2 * xstep
	max_x += 2 * xstep
	min_y -= 2 * ystep
	max_y += 2 * ystep
	bp_width = 0.01

	plt.xlim(min_x, max_x)
	plt.ylim(min_y, max_y)

	plt.plot([-10, 10], [-10, 10], color='black', linestyle='--')

	# GFP =============================================================================================================
	gfp_boxplot_xticks, gfp_boxplot_xtick_labels = list(), list()

	keys = neuron_scores_gfp.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores_gfp[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm, s=10, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm)

		gfp_boxplot_xticks.append(x)
		gfp_boxplot_xtick_labels.append(f'{key}')

		boxplot_h = plt.boxplot(scores_hgm, positions=[x], widths=bp_width, patch_artist=True,
								showfliers=True)
		boxplot_v = plt.boxplot(scores_glm, positions=[y], widths=bp_width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_h, boxplot_v]:
			for patch in box['boxes']:
				patch.set_facecolor('tomato')
				patch.set_alpha(0.5)

	for x, label in zip(gfp_boxplot_xticks, gfp_boxplot_xtick_labels):
		plt.text(x, min_y - 0.01, label, color='red', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	if len(neuron_scores_gfp) > 0:
		# gfp_high_score_hgm_impr = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_gfp.values()]), threshold)
		# gfp_high_score_glm = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_gfp.values()]), threshold)
		gfp_high_score_hgm = np.quantile(np.concatenate([pair[1] for pair in neuron_scores_gfp.values()]), threshold)
		plt.plot([min_x, max_x + 1], [gfp_high_score_hgm, gfp_high_score_hgm], color='black', linestyle='--')
	else:
		gfp_high_score_hgm_impr = -1E3
		gfp_high_score_hgm = -1E3

	# GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	keys = neuron_scores.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm, s=10, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		# if x < gfp_high_score_glm: continue
		y = np.median(scores_hgm)
		if y < gfp_high_score_hgm: continue

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		boxplot_h = plt.boxplot(scores_hgm, positions=[x], widths=bp_width, patch_artist=True,
								showfliers=True)
		boxplot_v = plt.boxplot(scores_glm, positions=[y], widths=bp_width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_h, boxplot_v]:
			for patch in box['boxes']:
				patch.set_facecolor('skyblue')
				patch.set_alpha(0.5)

	for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
		plt.text(x, min_y - 0.01, label, color='blue', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	xtickspos = np.arange(min_x, max_x, xstep)
	plt.xticks(xtickspos, [f'{val:.2f}' for val in xtickspos])
	ytickspos = np.arange(min_y, max_y, ystep)
	plt.yticks(ytickspos, [f'{val:.2f}' for val in ytickspos])

	if save: plt.savefig('out/sep-hgm-vs-glm-scores', dpi=300)


def plot_improvement_plot(neuron_scores, neuron_scores_gfp=None, limits=None, save=False):
	# TODO: automatic limit finding in this plot too
	if neuron_scores_gfp is None: neuron_scores_gfp = dict()

	threshold = 0.95

	plt.figure(figsize=(20, 20))
	plt.xlabel('GLM r2 score (median)', fontsize=20, labelpad=25)
	plt.ylabel('HGM-GLM r2 score (median)', fontsize=20)

	if limits is None:
		gcamp_x = np.array([score[0] for scores in neuron_scores.values() for score in scores])
		if len(neuron_scores_gfp) > 0:
			gfp_x = np.array([score[0] for scores in neuron_scores_gfp.values() for score in scores])
			full_x = np.array(np.concatenate([gcamp_x, gfp_x]))
		else:
			full_x = np.array(gcamp_x)

		gcamp_y = np.array([score[1] for scores in neuron_scores.values() for score in scores])
		if len(neuron_scores_gfp) > 0:
			gfp_y = np.array([score[1] for scores in neuron_scores_gfp.values() for score in scores])
			full_y = np.array(np.concatenate([gcamp_y, gfp_y]))
		else:
			full_y = np.array(gcamp_y)

		min_x, max_x = full_x.min(), full_x.max()
		min_y, max_y = full_y.min(), full_y.max()
	else:
		min_x, max_x, min_y, max_y = limits
	xstep = (max_x - min_x) / 10
	ystep = (max_y - min_y) / 10
	min_x -= 2 * xstep
	max_x += 2 * xstep
	min_y -= 2 * ystep
	max_y += 2 * ystep
	bp_width = 0.01

	plt.xlim(min_x, max_x)
	plt.ylim(min_y, max_y)

	# plt.plot([min_x, max_x + 1], [min_y, max_y + 1], color='black', linestyle='--')

	# GFP =============================================================================================================
	gfp_boxplot_xticks, gfp_boxplot_xtick_labels = list(), list()

	keys = neuron_scores_gfp.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores_gfp[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_glm)
		y = np.median(scores_hgm - scores_glm)

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=10, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		gfp_boxplot_xticks.append(x)
		gfp_boxplot_xtick_labels.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm - scores_glm, positions=[x], widths=bp_width, patch_artist=True,
								showfliers=True)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=bp_width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('tomato')
				patch.set_alpha(0.5)

	for x, label in zip(gfp_boxplot_xticks, gfp_boxplot_xtick_labels):
		# if x >= 0:
		plt.text(x, min_y - 0.01, label, color='red', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	if len(neuron_scores_gfp) > 0:
		gfp_high_score_hgm_impr = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_gfp.values()]),
											  threshold)
		plt.plot([min_x, max_x + 1], [gfp_high_score_hgm_impr, gfp_high_score_hgm_impr], color='black', linestyle='--')
	else:
		gfp_high_score_hgm_impr = -1E3
		gfp_high_score_hgm = -1E3

	# GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	keys = neuron_scores.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=10, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		# if x < gfp_high_score_glm: continue
		y = np.median(scores_hgm - scores_glm)
		if y < gfp_high_score_hgm_impr: continue

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm - scores_glm, positions=[x], widths=bp_width, patch_artist=True,
								showfliers=True)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=bp_width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('skyblue')
				patch.set_alpha(0.5)

	for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
		# if x >= 0:
		plt.text(x, min_y - 0.01, label, color='blue', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	xtickspos = np.arange(min_x, max_x, xstep)
	plt.xticks(xtickspos, [f'{val:.2f}' for val in xtickspos])
	ytickspos = np.arange(min_y, max_y, ystep)
	plt.yticks(ytickspos, [f'{val:.2f}' for val in ytickspos])

	if save: plt.savefig('out/sep-hgm-vs-glm-scores-rel', dpi=300)


def plot_losses(filenames, losses_m1, losses_m2, title, n_plots=None, save_name=None):
	if n_plots is None:
		n_plots = len(filenames)

	dim = math.ceil(math.sqrt(n_plots))
	fig, axes = plt.subplots(dim, dim, figsize=(24, 20))
	for i in range(n_plots):
		# axes[i // dim, i % dim].set_xticklabels([])
		# axes[i // dim, i % dim].set_yticklabels([])
		axes[i // dim, i % dim].grid(False)

	for i, (losses_per_split_m1, losses_per_split_m2) in enumerate(zip(losses_m1[:n_plots], losses_m2[:n_plots])):
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		for j, (losses_m1, losses_m2) in enumerate(zip(losses_per_split_m1, losses_per_split_m2)):
			axes[i // dim, i % dim].plot(losses_m1, color=colors[j])
			axes[i // dim, i % dim].plot(losses_m2, color=colors[j], alpha=0.5)
		axes[i // dim, i % dim].grid(True)
		axes[i // dim, i % dim].set_title(f'{filenames[i][:-14]}\n'
										  f'(model: {np.average(losses_per_split_m1[:, -1], axis=0) * 100:.2f}E-2, '
										  f'comp: {np.average(losses_per_split_m2[:, -1], axis=0) * 100:.2f}E-2)',
										  fontsize=10)

		yticks = axes[i // dim, i % dim].get_yticks()
		axes[i // dim, i % dim].set_yticks(yticks)
	# axes[i // dim, i % dim].set_yticklabels([f'{tick * 100:.2f}E-2' for tick in yticks])

	fig.subplots_adjust(hspace=0.4, wspace=0.5)
	fig.suptitle(title, fontsize=20, y=0.95)
	if save_name is not None: plt.savefig(f'./out/{save_name}')


def plot_manifolds(filenames, modes, vars_expl, vars_expl_rs=None):
	n_plots = len(modes)
	dim = math.ceil(math.sqrt(n_plots))
	fig, axes = plt.subplots(dim, dim, figsize=(20, 24))
	for i in range(n_plots):
		axes[i // dim, i % dim].set_xticklabels([])
		axes[i // dim, i % dim].set_yticklabels([])
		axes[i // dim, i % dim].grid(False)

	scale = [-1, 0, 1]
	for i, modes_ in enumerate(modes):
		axes[i // dim, i % dim].plot(modes_['CC1'] if 'CC1' in modes_.columns else modes_['PC1'],
									 modes_['CC2'] if 'CC2' in modes_.columns else modes_['PC2'])
		axes[i // dim, i % dim].set_xticks(scale)
		axes[i // dim, i % dim].set_xticklabels(scale)
		axes[i // dim, i % dim].set_yticks(scale)
		axes[i // dim, i % dim].set_yticklabels(scale)
		axes[i // dim, i % dim].grid(True)
		axes[i // dim, i % dim].set_title(f'{filenames[i][:-14]}\n{i}: {vars_expl[i]:.2f}' +
										  (f', ({vars_expl_rs[i]:.2f})' if vars_expl_rs is not None else ''),
										  fontsize=8)

	plt.subplots_adjust(hspace=0.5)
	if vars_expl_rs is not None:
		print(np.mean(np.array(vars_expl)), np.mean(np.array(vars_expl_rs)))
	else:
		print(np.mean(np.array(vars_expl)))


def plot_traces(traces, states, title, max_plots=4, label='reversal', limits=None, save=False):
	if max_plots < 0:
		max_plots = len(traces.columns)
	if limits == None: limits = (traces.values.min(), traces.values.max())

	heatmap = states.values.reshape(-1, 1) * 0.1

	dim1, dim2 = math.ceil(max_plots / 2), 2
	fig, axes = plt.subplots(dim1, dim2, figsize=(10 * dim2, 2 * (dim1 + 0.25)), constrained_layout=True)
	for i in range(dim1 * dim2):
		#axes[i // dim2, i % dim2].set_xticklabels([])
		#axes[i // dim2, i % dim2].set_yticklabels([])
		axes[i // dim2, i % dim2].grid(False)
		axes[i // dim2, i % dim2].set_ylim(limits[0], limits[1])

	keys = [key for key in traces.columns if key != ''][:max_plots]
	i = 0
	for key in keys:
		if key != '':
			# min_val, max_val = min(traces[key]), max(traces[key])
			min_val, max_val = limits
			axes[i // dim2, i % dim2].imshow(
				heatmap.T, aspect='auto', cmap='rainbow_r', interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			axes[i // dim2, i % dim2].plot(traces[key], color='black', linewidth=1.0)
			axes[i // dim2, i % dim2].grid(False)
			axes[i // dim2, i % dim2].set_title(f'{key}', fontsize=12)
			i += 1

	fig.text(0.5, 1.15, title, ha='center', fontsize=20)
	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('rainbow_r')(1.0), alpha=0.3, label=label)],
		loc='upper center',
		bbox_to_anchor=(0.5, 1.1),
		ncol=1,
		fontsize='large'
	)

	if save:
		plt.savefig('out/traces.png', dpi=300)
