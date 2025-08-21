import math
import numpy as np
import pandas as pd
from matplotlib import patches, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib import cm


def subplot(datas, n_plots, plots_per_row, top_space=0):
	if type(datas) in (pd.DataFrame, pd.Series):
		datas = [datas]

	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 4 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(4 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].set_xticklabels([])
		axes[i // plots_per_row, i % plots_per_row].set_yticklabels([])
		axes[i // plots_per_row, i % plots_per_row].grid(False)

	for data in datas:
		if data is not None:
			for i, key in enumerate(data):
				y = data[key]
				axes[i // plots_per_row, i % plots_per_row].grid(True)
				axes[i // plots_per_row, i % plots_per_row].plot(y)
				axes[i // plots_per_row, i % plots_per_row].set_title(key, fontsize=8)

	plt.show()


def plot_manifolds(filenames, modes, states, vars_expl, vars_expl_rs=None, n_plots=None, plots_per_row=5, save_name=None):
	if n_plots is None:
		n_plots = len(filenames)

	key = 'CC' if 'CC1' in modes[0].columns else 'PC'

	top_space = 1
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 4 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(4 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].set_xticklabels([])
		axes[i // plots_per_row, i % plots_per_row].set_yticklabels([])
		axes[i // plots_per_row, i % plots_per_row].grid(False)

	scale = [-1, 0, 1]
	for i, (modes_, states_) in enumerate(zip(modes[:n_plots], states[:n_plots])):
		points = np.array([modes_[f'{key}1'], modes_[f'{key}2']]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		colors = np.argmax(states_.values, axis=1)
		lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(np.min(colors), np.max(colors)))
		lc.set_array(colors)
		lc.set_linewidth(2)

		axes[i // plots_per_row, i % plots_per_row].add_collection(lc)

		# axes[i // plots_per_row, i % plots_per_row].set_xlim(modes_[f'{key}1'].min(), modes_[f'{key}1'].max())
		# axes[i // plots_per_row, i % plots_per_row].set_ylim(modes_[f'{key}2'].min(), modes_[f'{key}2'].max())
		axes[i // plots_per_row, i % plots_per_row].plot(modes_[f'{key}1'], modes_[f'{key}2'], alpha=0)
		axes[i // plots_per_row, i % plots_per_row].set_xticks(scale)
		axes[i // plots_per_row, i % plots_per_row].set_xticklabels(scale)
		axes[i // plots_per_row, i % plots_per_row].set_yticks(scale)
		axes[i // plots_per_row, i % plots_per_row].set_yticklabels(scale)
		axes[i // plots_per_row, i % plots_per_row].grid(True)
		axes[i // plots_per_row, i % plots_per_row].set_title(f'{filenames[i][:-14]}\n{i}: {vars_expl[i]:.2f}' +
										  (f', ({vars_expl_rs[i]:.2f})' if vars_expl_rs is not None else ''),
										  fontsize=12)

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel * 1 / 2, 'manifolds', ha='center', fontsize=20)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if vars_expl_rs is not None:
		print(np.mean(np.array(vars_expl)), np.mean(np.array(vars_expl_rs)))
	else:
		print(np.mean(np.array(vars_expl)))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300)
	plt.show()


# def plot_manifold_3d(filename, modes, states, var_expl, var_expl_rs=None):
# 	# Sample data
# 	x = modes[:, 0]
# 	y = modes[:, 1]
# 	z = modes[:, 2]
#
# 	# Prepare segments for LineCollection
# 	points = np.array([x, y, z]).T.reshape(-1, 1, 3)
# 	segments = np.concatenate([points[:-1], points[1:]], axis=1)
#
# 	# Use x or y or any scalar to map to color
# 	color_values = np.argmax(states.values, axis=1)
#
# 	# Create LineCollection
# 	lc = Line3DCollection(segments, cmap='viridis', norm=plt.Normalize(color_values.min(), color_values.max()))
# 	lc.set_array(states)
# 	lc.set_linewidth(2)
#
# 	# Create 3D plot
# 	fig = plt.figure(figsize=(12, 8))
# 	ax = fig.add_subplot(111, projection='3d')
#
# 	# Color points w.r.t. Y targets
# 	scatter = ax.plot(
# 		modes[:, 0],
# 		modes[:, 1],
# 		modes[:, 2],
# 		c=states,
# 		cmap='seismic',
# 		alpha=1,
# 		s=1,
# 	)
#
# 	# cbar = fig.colorbar(scatter, pad=0.1)
# 	# cbar.set_label('Delta Z (μm)', rotation=270, labelpad=20)
# 	#
# 	# ax.set_xlabel('PLS Component 1')
# 	# ax.set_ylabel('PLS Component 2')
# 	# ax.set_zlabel('PLS Component 3')
# 	# ax.set_title('PLS Regression - 3D Projection (Colored by Delta Z)')
#
# 	ax.view_init(elev=20, azim=45)
#
# 	plt.tight_layout()
# 	plt.show()


def plot_heatmap_stack(traces, states, label_bot, label_top, height_ratios=(1, 4), save_name=None):
	fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': height_ratios})

	if states is not None:
		im_top = ax_top.imshow(np.clip(states.values.T, -0.3, 0.8),
							   aspect='auto', cmap='rainbow', interpolation='nearest')
		ax_top.set_ylabel(label_top)
		ax_top.set_yticks(np.arange(len(states.values.T)))
		ax_top.set_yticklabels(states.columns, fontsize=8)
		ax_top.set_xticks(np.arange(0, len(states.values), 100))
		fig.colorbar(im_top, ax=ax_top, orientation='vertical', fraction=0.05)
		ax_top.grid(False)

	im_bottom = ax_bottom.imshow(np.clip(traces.values.T, -0.3, 0.8), aspect='auto',
								 cmap='rainbow', interpolation='nearest')
	ax_bottom.set_ylabel(label_bot)
	ax_bottom.set_yticks(np.arange(len(traces.values.T)))
	ax_bottom.set_yticklabels(traces.columns, fontsize=4)
	ax_bottom.set_xticks(np.arange(0, len(traces.values), 100))
	fig.colorbar(im_bottom, ax=ax_bottom, orientation='vertical', fraction=0.05)
	ax_bottom.grid(False)

	plt.tight_layout()
	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300)
	plt.show()


def plot_mcos(bars, bar_labels, labels, order, save_name=None):
	index_orders = [
		['m', 'c', 'o'].index(order[0]),
		['m', 'c', 'o'].index(order[1]),
		['m', 'c', 'o'].index(order[2])
	]
	rocket_cmap = cm.get_cmap('rocket')
	# color_pool = [rocket_cmap(0.4), rocket_cmap(0.9), rocket_cmap(0.1)]
	color_pool = [rocket_cmap(0.4), rocket_cmap(0.9), 'gray']
	mono_cmap = cm.get_cmap('Greys')
	bar_labels_color_pool = [mono_cmap(0.1), mono_cmap(0.9), mono_cmap(0.1)]

	colors = [color_pool[idx] for idx in index_orders]
	bar_labels_colors = [bar_labels_color_pool[idx] for idx in index_orders]
	bar_labels_positions = [
		# [np.mean(bars.loc[order[0]] / 2) for _ in bars.loc[order[0]]],
		# [(np.mean(bars.loc[order[0]] / 2) + 1 - np.mean(bars.loc[order[2]] / 2)) / 2 for _ in bars.loc[order[1]]],
		# [1 - np.mean(bars.loc[order[2]] / 2) for _ in bars.loc[order[2]]],

		# [y1 / 2 for y1 in bars.loc[order[0]]],
		# [y1 + y2 + y3 / 2 for y1, y2, y3 in zip(bars.loc[order[0]], bars.loc[order[1]], bars.loc[order[2]])]
		# [0.975 for _ in bars.loc[order[2]]],

		[0.025 for _ in bars.loc[order[0]]],
		[y1 + y2 - 0.025 * y2 for y1, y2 in zip(bars.loc[order[0]], bars.loc[order[1]])],
		[0.975 for _ in bars.loc[order[2]]]
	]
	x_positions = list((idx + 0.5) * 10. for idx in range(len(labels)))
	y_positions = [
		[0 for _ in bars.loc[order[0]]],
		[y1 for y1 in bars.loc[order[0]]],
		[y1 + y2 for y1, y2 in zip(bars.loc[order[0]], bars.loc[order[1]])]
	]

	plt.figure(figsize=(39, 13))

	for idx in range(3):
		plt.bar(x_positions, bars.loc[order[idx]], width=10, bottom=y_positions[idx], color=colors[idx], edgecolor='none')

	idx = 0
	for x, y, bar_label in zip(x_positions, bar_labels_positions[idx], bar_labels.loc[order[idx]]):
		# plt.text(x, y, f'±\n{bar_label * 100:.1f}%', color=tuple(1 - v for v in to_rgb(colors[i])), ha='center', va='center', fontsize=10)
		plt.text(x, y, f'±\n{bar_label * 100:.1f}%', color=bar_labels_colors[idx], ha='center', va='center', fontsize=10)
	idx = 1
	for x, y, bar_label in zip(x_positions, bar_labels_positions[idx], bar_labels.loc[order[idx]]):
		# plt.text(x, y, f'±\n{bar_label * 100:.1f}%', color=tuple(1 - v for v in to_rgb(colors[i])), ha='center', va='center', fontsize=10)
		plt.text(x, y, f'±\n{bar_label * 100:.1f}%', color=bar_labels_colors[idx], ha='center', va='top', fontsize=10)
	idx = 2
	for x, y, bar_label in zip(x_positions, bar_labels_positions[idx], bar_labels.loc[order[idx]]):
		# plt.text(x, y, f'±\n{bar_label * 100:.1f}%', color=tuple(1 - v for v in to_rgb(colors[i])), ha='center', va='center', fontsize=10)
		plt.text(x, y, f'±\n{bar_label * 100:.1f}%', color=bar_labels_colors[idx], ha='center', va='center', fontsize=10)

	plt.yticks(np.arange(0, 1.05, 0.05), labels=[f'{val}%' for val in np.arange(0, 105, 5)])
	plt.xticks(x_positions, labels)
	plt.xlim(0, x_positions[-1] + 5)
	plt.ylim(0, 1)

	plt.gca().set_axisbelow(False)
	plt.grid(False)
	plt.grid(axis='y', linestyle='--', linewidth=0.5, color=(0.0, 0.0, 0.0))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300)
	plt.show()


# def plot_losses(filenames, title, losses, n_plots=None, ylimits=None, save_name=None):
# 	if n_plots is None:
# 		n_plots = len(filenames)
#
# 	top_space = 1
# 	plots_per_row = 4
# 	plots_per_col = math.ceil(n_plots / plots_per_row)
# 	figheight = 4 * plots_per_col + top_space
# 	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(4 * plots_per_row, figheight), squeeze=False)
# 	for i in range(n_plots):
# 		axes[i // plots_per_row, i % plots_per_row].grid(False)
# 		if ylimits is not None: axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits)
#
# 	for i, (losses_per_split_m1, losses_per_split_m2) in enumerate(zip(losses_m1[:n_plots], losses_m2[:n_plots])):
# 		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# 		for j, (losses_m1, losses_m2) in enumerate(zip(losses_per_split_m1, losses_per_split_m2)):
# 			axes[i // plots_per_row, i % plots_per_row].plot(losses_m1, color=colors[j])
# 			axes[i // plots_per_row, i % plots_per_row].plot(losses_m2, color=colors[j], alpha=0.5)
# 		axes[i // plots_per_row, i % plots_per_row].grid(True)
# 		axes[i // plots_per_row, i % plots_per_row].set_title(f'{filenames[i][:-14]}\n'
# 										  f'(model: {np.average(losses_per_split_m1[:, -1], axis=0) * 100:.2f}E-2, '
# 										  f'comp: {np.average(losses_per_split_m2[:, -1], axis=0) * 100:.2f}E-2)',
# 										  fontsize=10)
#
# 		yticks = axes[i // plots_per_row, i % plots_per_row].get_yticks()
# 		axes[i // plots_per_row, i % plots_per_row].set_yticks(yticks)
#
# 	top_space_rel = top_space / figheight
# 	fig.text(0.5, 1. - top_space_rel * 1 / 2, title, ha='center', fontsize=20)
# 	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))
# 	# fig.subplots_adjust(
# 	# 	hspace=0.5,
# 	# )
#
# 	if save_name is not None: plt.savefig(f'./out/plots/{save_name}')


def plot_losses_splits(title, filenames, losses_m1, losses_m2, n_plots=None, plots_per_row=5, top_space=1, ylimits=None, save_name=None):
	if n_plots is None:
		n_plots = len(filenames)

	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 4 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(4 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].grid(False)
		if ylimits is not None: axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits)

	for i, (losses_per_split_m1, losses_per_split_m2) in enumerate(zip(losses_m1[:n_plots], losses_m2[:n_plots])):
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		for j, (losses_m1, losses_m2) in enumerate(zip(losses_per_split_m1, losses_per_split_m2)):
			axes[i // plots_per_row, i % plots_per_row].plot(losses_m1, color=colors[j])
			axes[i // plots_per_row, i % plots_per_row].plot(losses_m2, color=colors[j], alpha=0.5)
		axes[i // plots_per_row, i % plots_per_row].grid(True)
		axes[i // plots_per_row, i % plots_per_row].set_title(f'{filenames[i][:-14]}\n'
										  f'(model: {np.average(losses_per_split_m1[:, -1], axis=0) * 100:.2f}E-2, '
										  f'comp: {np.average(losses_per_split_m2[:, -1], axis=0) * 100:.2f}E-2)',
										  fontsize=10)

		yticks = axes[i // plots_per_row, i % plots_per_row].get_yticks()
		axes[i // plots_per_row, i % plots_per_row].set_yticks(yticks)

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel * 1 / 2, title, ha='center', fontsize=20)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))
	# fig.subplots_adjust(
	# 	hspace=0.5,
	# )

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


def plot_traces(title, traces, states, keys=None, n_plots=None, n_columns=2, ylimits=None, save_name=None):
	if type(traces) in (pd.DataFrame, pd.Series):
		traces = [traces]

	columns = traces[0].columns if type(traces[0]) is pd.DataFrame else [str(idx) for idx in range(traces[0].shape[1])]
	traces = [pd.DataFrame(data=trace, columns=columns) for trace in traces]

	if keys is None: keys = traces[0].columns
	keys = [key for key in keys if key != '']
	if n_plots is None: n_plots = len(keys)
	if ylimits is None: ylimits = (np.array([trace.values for trace in traces]).min(), np.array([trace.values for trace in traces]).max())

	top_space = 2
	n_rows = math.ceil(n_plots / n_columns)
	figheight = 2 * n_rows + top_space
	fig, axes = plt.subplots(n_rows, n_columns, figsize=(10 * n_columns, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // n_columns, i % n_columns].grid(False)
		axes[i // n_columns, i % n_columns].set_ylim(ylimits[0], ylimits[1])

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', fontsize=20)
	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('rainbow_r')(1.0), alpha=0.3, label='reversal')],
		loc='lower center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
		ncol=1,
		fontsize='large'
	)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))
	fig.subplots_adjust(
		hspace=0.5,
	)

	heatmap = states.values.reshape(-1, 1) * 0.1

	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits
			axes[i // n_columns, i % n_columns].imshow(
				heatmap.T, aspect='auto', cmap='rainbow_r', interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			axes[i // n_columns, i % n_columns].plot(traces[0][key], color='black', linewidth=1.0)
			for trace in traces[1:]:
				axes[i // n_columns, i % n_columns].plot(trace[key], linewidth=1.0)
			axes[i // n_columns, i % n_columns].grid(False)
			axes[i // n_columns, i % n_columns].set_title(f'{key}', fontsize=12)
			i += 1

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


def plot_traces_splits(title, states, traces, traces_additional=None, plot_init=True, n_plots=None, n_columns=2, ylimits=None, save_name=None):
	keys = [key for key in traces.columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (traces[keys].values.min(), traces[keys].values.max())

	top_space = 2
	n_rows = math.ceil(n_plots / n_columns)
	figheight = 2 * n_rows + top_space
	fig, axes = plt.subplots(n_rows, n_columns, figsize=(10 * n_columns, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // n_columns, i % n_columns].grid(False)
		axes[i // n_columns, i % n_columns].set_ylim(ylimits[0], ylimits[1])

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', fontsize=20)
	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('rainbow_r')(1.0), alpha=0.3, label='reversal')],
		loc='lower center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
		ncol=1,
		fontsize='large'
	)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))
	fig.subplots_adjust(
		hspace=0.5,
	)

	heatmap = states.values.reshape(-1, 1) * 0.1

	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits
			axes[i // n_columns, i % n_columns].imshow(
				heatmap.T, aspect='auto', cmap='rainbow_r', interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			if traces_additional is not None:
				for traces_per_split in traces_additional:
					axes[i // n_columns, i % n_columns].plot(traces_per_split[key], linewidth=1.0)
			axes[i // n_columns, i % n_columns].plot(traces[key], color='black', alpha=1. if plot_init else 0., linewidth=1.0)
			axes[i // n_columns, i % n_columns].grid(False)
			axes[i // n_columns, i % n_columns].set_title(f'{key}', fontsize=12)
			i += 1

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)


def plot_performance_plot(neuron_scores, neuron_scores_control=None, limits=None, save_name=None):
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	plt.figure(figsize=(15, 15))
	plt.xlabel('GLM r2 score (median)', fontsize=20, labelpad=50)
	plt.ylabel('HGM r2 score (median)', fontsize=20)

	if limits is None:
		gcamp_x = np.array([score for scores in neuron_scores.values() for score in scores[0]])
		if len(neuron_scores_control) > 0:
			gfp_x = np.array([score for scores in neuron_scores_control.values() for score in scores[0]])
			full_x = np.array(np.concatenate([gcamp_x, gfp_x]))
		else:
			full_x = np.array(gcamp_x)

		gcamp_y = np.array([score for scores in neuron_scores.values() for score in scores[1]])
		if len(neuron_scores_control) > 0:
			gfp_y = np.array([score for scores in neuron_scores_control.values() for score in scores[1]])
			full_y = np.array(np.concatenate([gcamp_y, gfp_y]))
		else:
			full_y = np.array(gcamp_y)

		min_x, max_x = full_x.min(), full_x.max()
		min_y, max_y = full_y.min(), full_y.max()
	else:
		min_x, max_x, min_y, max_y = limits
	xstep = (max_x - min_x) / 10
	ystep = (max_y - min_y) / 10
	# min_x -= 2 * xstep
	# max_x += 2 * xstep
	# min_y -= 2 * ystep
	# max_y += 2 * ystep
	bp_width = 0.01

	plt.xlim(min_x, max_x)
	plt.ylim(min_y, max_y)

	plt.plot([-10, 10], [-10, 10], color='black', linestyle='--')

	# GFP =============================================================================================================
	gfp_boxplot_xticks, gfp_boxplot_xtick_labels = list(), list()

	keys = neuron_scores_control.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores_control[key]
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

	if len(neuron_scores_control) > 0:
		# gfp_high_score_hgm_impr = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_gfp.values()]), threshold)
		# gfp_high_score_glm = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_gfp.values()]), threshold)
		gfp_high_score_hgm = np.quantile(np.concatenate([pair[1] for pair in neuron_scores_control.values()]), threshold)
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

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


def plot_improvement_plot(neuron_scores, neuron_scores_control=None, limits=None, save_name=None):
	# TODO: automatic limit finding in this plot too
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	plt.figure(figsize=(15, 15))
	plt.xlabel('GLM r2 score (median)', fontsize=20, labelpad=25)
	plt.ylabel('HGM-GLM r2 score (median)', fontsize=20)

	if limits is None:
		gcamp_x = np.array([score[0] for scores in neuron_scores.values() for score in scores])
		if len(neuron_scores_control) > 0:
			gfp_x = np.array([score[0] for scores in neuron_scores_control.values() for score in scores])
			full_x = np.array(np.concatenate([gcamp_x, gfp_x]))
		else:
			full_x = np.array(gcamp_x)

		gcamp_y = np.array([score[1] for scores in neuron_scores.values() for score in scores])
		if len(neuron_scores_control) > 0:
			gfp_y = np.array([score[1] for scores in neuron_scores_control.values() for score in scores])
			full_y = np.array(np.concatenate([gcamp_y, gfp_y]))
		else:
			full_y = np.array(gcamp_y)

		min_x, max_x = full_x.min(), full_x.max()
		min_y, max_y = full_y.min(), full_y.max()
	else:
		min_x, max_x, min_y, max_y = limits
	xstep = (max_x - min_x) / 10
	ystep = (max_y - min_y) / 10
	# min_x -= 2 * xstep
	# max_x += 2 * xstep
	# min_y -= 2 * ystep
	# max_y += 2 * ystep
	bp_width = 0.01

	plt.xlim(min_x, max_x)
	plt.ylim(min_y, max_y)

	# plt.plot([min_x, max_x + 1], [min_y, max_y + 1], color='black', linestyle='--')

	# GFP =============================================================================================================
	gfp_boxplot_xticks, gfp_boxplot_xtick_labels = list(), list()

	keys = neuron_scores_control.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores_control[key]
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
		if x >= 0:
			plt.text(x, min_y - 0.01, label, color='red', ha='center', va='top', rotation=45, fontsize=5,
					 transform=plt.gca().transData)
	# =================================================================================================================

	if len(neuron_scores_control) > 0:
		gfp_high_score_hgm_impr = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_control.values()]),
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
		if x >= 0:
			plt.text(x, min_y - 0.01, label, color='blue', ha='center', va='top', rotation=45, fontsize=5,
					 transform=plt.gca().transData)
	# =================================================================================================================

	xtickspos = np.arange(min_x, max_x, xstep)
	plt.xticks(xtickspos, [f'{val:.2f}' for val in xtickspos])
	ytickspos = np.arange(min_y, max_y, ystep)
	plt.yticks(ytickspos, [f'{val:.2f}' for val in ytickspos])

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()
