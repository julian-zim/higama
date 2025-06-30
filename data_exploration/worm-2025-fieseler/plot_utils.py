import math
import numpy as np
from matplotlib import pyplot as plt


def plot_performance_plot(neuron_scores, neuron_scores_gfp, save=False):
	threshold = 0.95

	plt.figure(figsize=(20, 20))
	plt.xlabel('GLM r2 score (median)', fontsize=20, labelpad=50)
	plt.ylabel('HGM r2 score (median)', fontsize=20)

	min_x = -0.5
	max_x = 0.5
	min_y = -0.5
	max_y = 0.5
	width = 0.01

	plt.xlim(min_x, max_x)
	plt.ylim(min_y, max_y)

	plt.plot([min_x, max_x + 1], [min_y, max_y + 1], color='black', linestyle='--')

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

		boxplot_h = plt.boxplot(scores_hgm, positions=[x], widths=width, patch_artist=True,
								showfliers=True)
		boxplot_v = plt.boxplot(scores_glm, positions=[y], widths=width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_h, boxplot_v]:
			for patch in box['boxes']:
				patch.set_facecolor('tomato')
				patch.set_alpha(0.5)

	for x, label in zip(gfp_boxplot_xticks, gfp_boxplot_xtick_labels):
		plt.text(x, min_y - 0.02, label, color='red', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	gfp_high_score_hgm_impr = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_gfp.values()]),
										  threshold)
	gfp_high_score_glm = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_gfp.values()]), threshold)
	gfp_high_score_hgm = np.quantile(np.concatenate([pair[1] for pair in neuron_scores_gfp.values()]), threshold)
	plt.plot([min_x, max_x + 1], [gfp_high_score_hgm_impr, gfp_high_score_hgm_impr], color='black', linestyle='--')

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

		boxplot_h = plt.boxplot(scores_hgm, positions=[x], widths=width, patch_artist=True,
								showfliers=True)
		boxplot_v = plt.boxplot(scores_glm, positions=[y], widths=width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_h, boxplot_v]:
			for patch in box['boxes']:
				patch.set_facecolor('skyblue')
				patch.set_alpha(0.5)

	for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
		plt.text(x, min_y - 0.02, label, color='blue', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	plt.xticks(np.arange(min_x, max_x, 0.1), [f'{val:.1f}' for val in np.arange(min_x, max_x, 0.1)])
	plt.yticks(np.arange(min_y, max_y, 0.1), [f'{val:.1f}' for val in np.arange(min_y, max_y, 0.1)])

	if save: plt.savefig('out/sep-hgm-vs-glm-scores', dpi=300)


def plot_improvement_plot(neuron_scores, neuron_scores_gfp, save=False):
	threshold = 0.95

	plt.figure(figsize=(20, 20))
	plt.xlabel('GLM r2 score (median)', fontsize=20, labelpad=25)
	plt.ylabel('HGM-GLM r2 score (median)', fontsize=20)

	min_x = -0.6
	max_x = 0.6
	min_y = 0
	max_y = 0.16
	width = 0.01

	plt.xlim(min_x, max_x)
	plt.ylim(0, max_y)

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

		boxplot_v = plt.boxplot(scores_hgm - scores_glm, positions=[x], widths=width, patch_artist=True,
								showfliers=True)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('tomato')
				patch.set_alpha(0.5)

	for x, label in zip(gfp_boxplot_xticks, gfp_boxplot_xtick_labels):
		# if x >= 0:
		plt.text(x, min_y - 0.004, label, color='red', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	# gfp_high_score_glm = np.quantile(np.concatenate([pair[0] for pair in gfp_neuron_scores.values()]), threshold)
	gfp_high_score_hgm_impr = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_gfp.values()]),
										  threshold)
	plt.plot([min_x, max_x + 1], [gfp_high_score_hgm_impr, gfp_high_score_hgm_impr], color='black', linestyle='--')

	# GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	keys = neuron_scores.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_glm)
		# if x < gfp_high_score_glm: continue
		y = np.median(scores_hgm - scores_glm)
		if y < gfp_high_score_hgm_impr: continue

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=10, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm - scores_glm, positions=[x], widths=width, patch_artist=True,
								showfliers=True)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=width * (max_y - min_y) / (max_x - min_x), patch_artist=True,
								showfliers=True, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('skyblue')
				patch.set_alpha(0.5)

	for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
		# if x >= 0:
		plt.text(x, -0.004, label, color='blue', ha='center', va='top', rotation=45, fontsize=5,
				 transform=plt.gca().transData)
	# =================================================================================================================

	plt.xticks(np.arange(min_x, max_x, 0.1), [f'{val:.1f}' for val in np.arange(min_x, max_x, 0.1)])
	plt.yticks(np.arange(min_y, max_y, 0.01), [f'{val:.2f}' for val in np.arange(min_y, max_y, 0.01)])

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
