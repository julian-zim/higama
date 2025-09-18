import math
import numpy as np
import pandas as pd
from matplotlib import patches, pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from matplotlib import cm


def subplot(title, datas, n_plots, plots_per_row=5, top_space=1, save_name=None):
	if type(datas) in (pd.DataFrame, pd.Series):
		datas = [datas]

	plots_per_row = min(n_plots, plots_per_row)
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

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel / 2, title, ha='center', fontsize=20)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300)
	plt.show()


def plot_manifolds(filenames, modes, states, vars_expl, vars_expl_rs=None, n_plots=None, plots_per_row=5, top_space=1, save_name=None):
	if n_plots is None:
		n_plots = len(filenames)

	key = 'CC' if 'CC1' in modes[0].columns else 'PC'

	plots_per_row = min(n_plots, plots_per_row)
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
		lc = LineCollection(segments, cmap='jet', norm=plt.Normalize(np.min(colors), np.max(colors)))  # ToDo: better colors
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
	fig.text(0.5, 1. - top_space_rel / 2, 'manifolds', ha='center', fontsize=20)
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


"""
Plots traces cycling color according to pyplots policy, with the first one black
"""
def plot_traces(title, states, traces,
				n_plots=None, plots_per_row=2, top_space=2, ylimits=None, save_name=None):
	#region setup
	if type(traces) is pd.DataFrame:
		traces = [traces]

	keys = [key for key in traces[0].columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (traces[0][keys].values.min(), traces[0][keys].values.max())
	#endregion

	#region plot setup
	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 3 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(15 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].grid(False)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel / 4, title, ha='center', fontsize=20)
	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='reversal')],
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
	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits
			axes[i // plots_per_row, i % plots_per_row].imshow(
				heatmap.T, aspect='auto',
				cmap=mcolors.LinearSegmentedColormap.from_list('middle_third',
															   plt.get_cmap('binary')(np.linspace(0, 1/2, 256))),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			axes[i // plots_per_row, i % plots_per_row].plot(traces[0][key], color='black', linewidth=1.0)
			for trace in traces[1:]:
				axes[i // plots_per_row, i % plots_per_row].plot(traces[0][key], linewidth=1.0)

			axes[i // plots_per_row, i % plots_per_row].grid(False)
			axes[i // plots_per_row, i % plots_per_row].set_title(f'{key}', fontsize=12)
			i += 1
	#endregion

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


"""
Plots single trace in black and optional splits in cycling colors, with the first being opaques and the second transparents
"""
def plot_traces_x_splits(title, states, traces, traces_splits=None, traces_splits_ref=None, plot_first=True,
						 n_plots=None, plots_per_row=2, top_space=2, ylimits=None, save_name=None):
	#region setup
	keys = [key for key in traces.columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (traces[keys].values.min(), traces[keys].values.max())
	#endregion

	#region plot setup
	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 3 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(15 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].grid(False)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces.shape), color='gray', linewidth=0.25)

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel / 4, title, ha='center', fontsize=20)
	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='reversal')],
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
	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits
			axes[i // plots_per_row, i % plots_per_row].imshow(
				heatmap.T, aspect='auto',
				cmap=mcolors.LinearSegmentedColormap.from_list('middle_third',
															   plt.get_cmap('binary')(np.linspace(0, 1 / 2, 256))),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			axes[i // plots_per_row, i % plots_per_row].plot(traces[key], color='black', alpha=1.0 if plot_first else 0.0, linewidth=1.0)

			if traces_splits is not None:
				colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
				for j in range(len(traces_splits)):
					color = next(colors)
					axes[i // plots_per_row, i % plots_per_row].plot(traces_splits[j][key], linewidth=1.0, color=color)
					if traces_splits_ref is not None:
						axes[i // plots_per_row, i % plots_per_row].plot(traces_splits_ref[j][key], linewidth=1.0, color=color, alpha=0.5)

			axes[i // plots_per_row, i % plots_per_row].grid(False)
			axes[i // plots_per_row, i % plots_per_row].set_title(f'{key}', fontsize=12)
			i += 1
	#endregion

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


"""
Plots two pairs of traces in red and green black, with the respective first one being opaque and the second transparent
"""
def plot_traces_comparisons(title, states, traces, traces_ref=None,
							n_plots=None, plots_per_row=2, top_space=2, ylimits=None, save_name=None):
	#region setup
	keys = [key for key in traces[0].columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (np.array([trace[keys].values for trace in traces]).min(), np.array([trace[keys].values for trace in traces]).max())
	#endregion

	#region plot setup
	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 3 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(15 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].grid(False)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel / 4, title, ha='center', fontsize=20)
	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='reversal')],
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
	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits
			axes[i // plots_per_row, i % plots_per_row].imshow(
				heatmap.T, aspect='auto',
				cmap=mcolors.LinearSegmentedColormap.from_list('middle_third',
															   plt.get_cmap('binary')(np.linspace(0, 1 / 2, 256))),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)

			axes[i // plots_per_row, i % plots_per_row].plot(traces[0][key], linewidth=1.0, color='C0')
			axes[i // plots_per_row, i % plots_per_row].plot(traces[1][key], linewidth=1.0, color='green')

			if traces_ref is not None:
				axes[i // plots_per_row, i % plots_per_row].plot(traces_ref[0][key], linewidth=1.0, color='C0', alpha=0.5)
				axes[i // plots_per_row, i % plots_per_row].plot(traces_ref[1][key], linewidth=1.0, color='green', alpha=0.5)

			axes[i // plots_per_row, i % plots_per_row].grid(False)
			axes[i // plots_per_row, i % plots_per_row].set_title(f'{key}', fontsize=12)
			i += 1
	#endregion

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


"""
Plots losses for splits, with additional losses in transparent, as references.
"""
def plot_losses_splits(title, filenames, losses, losses_reference=None,
					   n_plots=None, plots_per_row=5, top_space=1, ylimits=None, save_name=None):
	if n_plots is None:
		n_plots = len(filenames)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = 4 * plots_per_col + top_space
	fig, axes = plt.subplots(plots_per_col, plots_per_row, figsize=(4 * plots_per_row, figheight), squeeze=False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].grid(False)
		if ylimits is not None: axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits)

	loss_ref = True
	if losses_reference is None:
		loss_ref = False
		losses_reference = losses
	for i, (losses_per_split_m1, losses_per_split_m2) in enumerate(zip(losses[:n_plots], losses_reference[:n_plots])):
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		for j, (losses, losses_reference) in enumerate(zip(losses_per_split_m1, losses_per_split_m2)):
			axes[i // plots_per_row, i % plots_per_row].plot(losses, color=colors[j])
			if loss_ref:
				axes[i // plots_per_row, i % plots_per_row].plot(losses_reference, color=colors[j], alpha=0.5)
		axes[i // plots_per_row, i % plots_per_row].grid(True)
		axes[i // plots_per_row, i % plots_per_row].set_title(f'{filenames[i][:-14]}\n' +
															  f'(model: {np.average(losses_per_split_m1[:, -1], axis=0) * 100:.2f}E-2' +
															  (f', comp: {np.average(losses_per_split_m2[:, -1], axis=0) * 100:.2f}E-2)' if loss_ref else ''),
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


def plot_performance_plot(neuron_scores, neuron_scores_control=None, limits=None, content_size=20, save_name=None):
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	plt.figure(figsize=(10, 10))
	plt.xlabel('GLM r2 score (median)', fontsize=12.5, labelpad=50)
	plt.ylabel('HGM r2 score (median)', fontsize=12.5)

	if limits is None:
		x_all = np.array([score for scores_glm, scores_hgm in neuron_scores.values() for score in scores_glm])
		if len(neuron_scores_control) > 0:
			x_all = np.array(np.concatenate(
				[x_all,
				 np.array([score for scores_glm, scores_hgm in neuron_scores_control.values() for score in scores_glm])]
			))

		y_all = np.array([score for scores_glm, scores_hgm in neuron_scores.values() for score in scores_hgm])
		if len(neuron_scores_control) > 0:
			y_all = np.array(np.concatenate(
				[y_all,
				 np.array([score for scores_glm, scores_hgm in neuron_scores_control.values() for score in scores_hgm])]
			))

		x_min, x_max = x_all.min(), x_all.max()
		y_min, y_max = y_all.min(), y_all.max()
		x_step, y_step = (x_max - x_min) / 10, (y_max - y_min) / 10
		x_min -= x_step
		x_max += x_step
		y_min -= y_step
		y_max += y_step
	else:
		x_min, x_max, y_min, y_max = limits
		x_step, y_step = (x_max - x_min) / 10, (y_max - y_min) / 10

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)

	bp_pxs = (
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (content_size, 0)
		)[0]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[0],
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (0, content_size)
		)[1]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[1]
	)

	plt.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--')
	plt.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--')
	plt.plot([min(x_min, y_min), (max(x_max, y_max))], [min(x_min, y_min), max(x_max, y_max)], color='black', linestyle='--')

	# GFP =============================================================================================================
	boxplot_xticks_gfp, boxplot_xtick_labels_gfp = list(), list()

	for key in neuron_scores_control.keys():
		scores_glm, scores_hgm = neuron_scores_control[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm, s=content_size, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm, positions=[x], widths=bp_pxs[0], patch_artist=True,
								showfliers=False)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=bp_pxs[1], patch_artist=True,
								showfliers=False, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('tomato')
				patch.set_alpha(0.5)
			for median in box['medians']:
				median.set_color('lightsteelblue')
				median.set_linewidth(1)

	for x, label in zip(boxplot_xticks_gfp, boxplot_xtick_labels_gfp):
		if x_min <= x <= x_max:
			plt.text(x, -0.05, label, color='red', ha='center', va='top', rotation=45, fontsize=7.5,
					 transform=plt.gca().get_xaxis_transform())
	# =================================================================================================================

	if len(neuron_scores_control) > 0:
		# high_score_glm_gfp = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_control.values()]),
		# 								 threshold)
		high_score_hgm_gfp = np.quantile(np.concatenate([pair[1] for pair in neuron_scores_control.values()]),
										 threshold)
		# high_score_hgm_impr_gfp = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_control.values()]),
		# 									  threshold)
		plt.plot([x_min, x_max + x_step], [high_score_hgm_gfp, high_score_hgm_gfp], color='black', linestyle='--')
	else:
		high_score_hgm_gfp = -1E3
		# high_score_hgm_impr_gfp = -1E3

	# GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	for key in neuron_scores.keys():
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm, s=content_size, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		# if x < high_score_glm_gfp: continue
		y = np.median(scores_hgm)
		if y < high_score_hgm_gfp: continue

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm, positions=[x], widths=bp_pxs[0], patch_artist=True,
								showfliers=False)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=bp_pxs[1], patch_artist=True,
								showfliers=False, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('skyblue')
				patch.set_alpha(0.5)
			for median in box['medians']:
				median.set_color('darkorange')
				median.set_linewidth(1)

	for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
		if x_min <= x <= x_max:
			plt.text(x, -0.05, label, color='blue', ha='center', va='top', rotation=45, fontsize=7.5,
					 transform=plt.gca().get_xaxis_transform())
	# =================================================================================================================

	x_ticks_pos = np.arange(x_min, x_max, x_step)
	plt.xticks(x_ticks_pos, [f'{val:.3f}' for val in x_ticks_pos])
	y_ticks_pos = np.arange(y_min, y_max, y_step)
	plt.yticks(y_ticks_pos, [f'{val:.3f}' for val in y_ticks_pos])

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()


def plot_improvement_plot(neuron_scores, neuron_scores_control=None, limits=None, content_size=20, save_name=None):
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	plt.figure(figsize=(10, 10))
	plt.xlabel('GLM r2 score (median)', fontsize=12.5, labelpad=50)
	plt.ylabel('HGM-GLM r2 score (median)', fontsize=12.5)

	if limits is None:
		x_all = np.array([score for scores_glm, scores_hgm in neuron_scores.values() for score in scores_glm])
		if len(neuron_scores_control) > 0:
			x_all = np.array(np.concatenate(
				[x_all,
				 np.array([score for scores_glm, scores_hgm in neuron_scores_control.values() for score in scores_glm])]
			))

		y_all = np.array([score for scores_glm, scores_hgm in neuron_scores.values() for score in scores_hgm - scores_glm])
		if len(neuron_scores_control) > 0:
			y_all = np.array(np.concatenate(
				[y_all,
				 np.array([score for scores_glm, scores_hgm in neuron_scores_control.values() for score in scores_hgm - scores_glm])]
			))

		x_min, x_max = x_all.min(), x_all.max()
		y_min, y_max = y_all.min(), y_all.max()
		x_step, y_step = (x_max - x_min) / 10, (y_max - y_min) / 10
		x_min -= x_step
		x_max += x_step
		y_min -= y_step
		y_max += y_step
	else:
		x_min, x_max, y_min, y_max = limits
		x_step, y_step = (x_max - x_min) / 10, (y_max - y_min) / 10

	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)

	bp_pxs = (
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (content_size, 0)
		)[0]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[0],
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (0, content_size)
		)[1]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[1]
	)

	plt.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--')
	plt.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--')
	# plt.plot([x_min, x_max + x_step], [y_min, y_max + y_step], color='black', linestyle='--')

	# GFP =============================================================================================================
	boxplot_xticks_gfp, boxplot_xtick_labels_gfp = list(), list()

	for key in neuron_scores_control.keys():
		scores_glm, scores_hgm = neuron_scores_control[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=content_size, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm - scores_glm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm - scores_glm, positions=[x], widths=bp_pxs[0], patch_artist=True,
								showfliers=False)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=bp_pxs[1], patch_artist=True,
								showfliers=False, vert=False)
		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('tomato')
				patch.set_alpha(0.5)
			for median in box['medians']:
				median.set_color('lightsteelblue')
				median.set_linewidth(1)

	for x, label in zip(boxplot_xticks_gfp, boxplot_xtick_labels_gfp):
		if x_min <= x <= x_max:
			plt.text(x, -0.05, label, color='red', ha='center', va='top', rotation=45, fontsize=7.5,
					 transform=plt.gca().get_xaxis_transform())
	# =================================================================================================================

	if len(neuron_scores_control) > 0:
		# high_score_glm_gfp = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_control.values()]),
		# 								 threshold)
		# high_score_hgm_gfp = np.quantile(np.concatenate([pair[1] for pair in neuron_scores_control.values()]),
		# 								 threshold)
		high_score_hgm_impr_gfp = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_control.values()]),
											  threshold)
		plt.plot([x_min, x_max + x_step], [high_score_hgm_impr_gfp, high_score_hgm_impr_gfp], color='black', linestyle='--')
	else:
		high_score_hgm_impr_gfp = -1E3
		# high_score_hgm_gfp = -1E3

	# GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	keys = neuron_scores.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=content_size, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		# if x < high_score_glm_gfp: continue
		y = np.median(scores_hgm - scores_glm)
		if y < high_score_hgm_impr_gfp: continue

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		boxplot_v = plt.boxplot(scores_hgm - scores_glm, positions=[x], widths=bp_pxs[0], patch_artist=True,
								showfliers=False)
		boxplot_h = plt.boxplot(scores_glm, positions=[y], widths=bp_pxs[1], patch_artist=True,
								showfliers=False, vert=False)

		for box in [boxplot_v, boxplot_h]:
			for patch in box['boxes']:
				patch.set_facecolor('skyblue')
				patch.set_alpha(0.5)
			for median in box['medians']:
				median.set_color('darkorange')
				median.set_linewidth(1)

	for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
		if x >= x_min:
			plt.text(x, -0.05, label, color='blue', ha='center', va='top', rotation=45, fontsize=7.5,
					 transform=plt.gca().get_xaxis_transform())
	# =================================================================================================================

	x_ticks_pos = np.arange(x_min, x_max, x_step)
	plt.xticks(x_ticks_pos, [f'{val:.3f}' for val in x_ticks_pos])
	y_ticks_pos = np.arange(y_min, y_max, y_step)
	plt.yticks(y_ticks_pos, [f'{val:.3f}' for val in y_ticks_pos])

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}', dpi=300)
	plt.show()
