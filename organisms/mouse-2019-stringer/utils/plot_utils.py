import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors, patches, cm
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_manifolds(filenames, modes, vars_expl, vars_expl_rs=None,
				   title='manifolds', n_plots=None, plots_per_row=4, top_space=2, save_name=None):
	#region setup
	if n_plots is None:
		n_plots = len(filenames)

	key = modes[0].columns[0][:-1]
	#endregion

	#region plot setup
	fig_size = (4, 4)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		ax.set_xticks([])
		ax.set_xticklabels([])
		ax.set_yticks([])
		ax.set_yticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 6, title, ha='center', va='center', fontsize=20)

	#endregion

	#region plot
	scale = [] #[-1, 0, 1]
	for i, modes_ in enumerate(modes[:n_plots]):


		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}\n{sum(vars_expl[i]) * 100:.1f}%' +
										  (f' ({sum(vars_expl_rs[i]) * 100:.1f}%)' if vars_expl_rs is not None else ''),
										  fontsize=10.5)

		ax.plot(modes_[f'{key}1'], modes_[f'{key}2'])
		# ax.set_xlabel('test')
		ax.set_xticks(scale)
		ax.set_xticklabels(scale)
		# ax.set_ylabel('test')
		ax.set_yticks(scale)
		ax.set_yticklabels(scale)
		ax.grid(True)
		for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()

	if vars_expl_rs is not None:
		print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}% ({np.mean(np.array(vars_expl_rs)) * 100:.1f}%)')
	else:
		print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}%')



"""
Plots losses for splits, with additional losses in transparent, as references.
"""
def plot_losses_splits(title, filenames, losses, losses_reference=None,
					   n_plots=None, plots_per_row=4, top_space=1, ylimits=None, save_name=None):
	#region setup
	if n_plots is None:
		n_plots = len(filenames)
	#endregion

	#region plot setup
	fig_size = (4, 4)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		# ax.set_xticks([])
		# ax.set_xticklabels([])
		# ax.set_yticks([])
		# ax.set_yticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		if ylimits is not None: axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits)

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel / 2, title, ha='center', va='center', fontsize=20)
	#endregion

	#region plot
	loss_ref = True
	if losses_reference is None:
		loss_ref = False
		losses_reference = losses
	for i, (losses_per_split_m1, losses_per_split_m2) in enumerate(zip(losses[:n_plots], losses_reference[:n_plots])):
		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}\n' +
					 f'(model: {np.average(losses_per_split_m1[:, -1], axis=0) * 100:.2f}E-2' +
					 (f', comp: {np.average(losses_per_split_m2[:, -1], axis=0) * 100:.2f}E-2)' if loss_ref else ''),
					 fontsize=10)

		for j, (losses, losses_reference) in enumerate(zip(losses_per_split_m1, losses_per_split_m2)):
			ax.plot(losses, color=colors[j])
			if loss_ref:
				ax.plot(losses_reference, color=colors[j], alpha=0.5)

		ax.set_yticks(axes[i // plots_per_row, i % plots_per_row].get_yticks())
		ax.grid(True)
		for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()



#region traces
"""
Plots traces cycling color according to pyplots policy, with the first one black
"""
def plot_traces(title, traces,
				n_plots=None, plots_per_row=2, top_space=1, ylimits=None, save_name=None):
	#region setup
	if type(traces) is pd.DataFrame:
		traces = [traces]

	keys = [key for key in traces[0].columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (
		np.array(
			[trace[keys].values for trace in traces]
		).min(),
		np.array(
			[trace[keys].values for trace in traces]
		).max()
	)
	#endregion

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		# ax.set_xticklabels([])
		# ax.set_yticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits

			ax = axes[i // plots_per_row, i % plots_per_row]
			ax.set_title(f'{key}', fontsize=12)

			ax.plot(traces[0][key], color='black', linewidth=1.0)
			for trace in traces[1:]:
				ax.plot(trace[key], linewidth=1.0)

			for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
				spine.set_visible(True)
			i += 1
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()


"""
Plots traces cycling color according to pyplots policy, with the first one black, with states
"""
def plot_traces_states(title, states, traces,
				label='running speed', n_plots=None, plots_per_row=2, top_space=1, ylimits=None, save_name=None):
	#region setup
	if type(traces) is pd.DataFrame:
		traces = [traces]

	keys = [key for key in traces[0].columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (
		np.array(
			[trace[keys].values for trace in traces]
		).min(),
		np.array(
			[trace[keys].values for trace in traces]
		).max()
	)
	#endregion

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		# ax.set_xticklabels([])
		# ax.set_yticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label=label)],
		loc='center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
		ncol=1,
		fontsize='large'
	)

	heatmap = states.values.reshape(-1, 1) * 0.1
	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits

			ax = axes[i // plots_per_row, i % plots_per_row]
			ax.set_title(f'{key}', fontsize=12)

			ax.imshow(
				heatmap.T, aspect='auto',
				cmap=mcolors.LinearSegmentedColormap.from_list('middle_third',
															   plt.get_cmap('binary')(np.linspace(0, 1/2, 256))),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			ax.plot(traces[0][key], color='black', linewidth=1.0)
			for trace in traces[1:]:
				ax.plot(trace[key], linewidth=1.0)

			for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
				spine.set_visible(True)
			i += 1
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()


"""
Plots single trace in black and optional splits in cycling colors, with the first being opaques and the second transparents
"""
def plot_traces_x_splits(title, states, traces_first, traces_splits=None, traces_splits_ref=None, plot_first=True,
						 n_plots=None, plots_per_row=2, top_space=1, ylimits=None, save_name=None):
	#region setup
	keys = [key for key in traces_first.columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None:
		ylimits = (
			np.array(
				([traces_first[keys].values] if plot_first else []) +
				([trace_splits[keys].values for trace_splits in traces_splits] if traces_splits is not None else []) +
				([trace_splits_ref[keys].values for trace_splits_ref in traces_splits_ref] if traces_splits_ref is not None else [])
			).min(),
			np.array(
				([traces_first[keys].values] if plot_first else []) +
				([trace_splits[keys].values for trace_splits in traces_splits] if traces_splits is not None else []) +
				([trace_splits_ref[keys].values for trace_splits_ref in traces_splits_ref] if traces_splits_ref is not None else [])
			).max()
		)
	#endregion

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		# ax.set_xticklabels([])
		# ax.set_yticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces_first.shape), color='gray', linewidth=0.25)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='running_speed')],
		loc='center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
		ncol=1,
		fontsize='large'
	)

	heatmap = states.values.reshape(-1, 1) * 0.1
	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits

			ax = axes[i // plots_per_row, i % plots_per_row]
			ax.set_title(f'{key}', fontsize=12)

			ax.imshow(
				heatmap.T, aspect='auto',
				cmap=mcolors.LinearSegmentedColormap.from_list('middle_third',
															   plt.get_cmap('binary')(np.linspace(0, 1 / 2, 256))),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			ax.plot(traces_first[key], color='black', alpha=1.0 if plot_first else 0.0, linewidth=1.0)
			if traces_splits is not None:
				colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
				for j in range(len(traces_splits)):
					color = next(colors)
					ax.plot(traces_splits[j][key], color=color, linewidth=1.0)
					if traces_splits_ref is not None:
						ax.plot(traces_splits_ref[j][key], color=color, alpha=0.5, linewidth=1.0)

			for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
				spine.set_visible(True)

			i += 1
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()


"""
Plots two pairs of traces in red and green black, with the respective first one being opaque and the second transparent
"""
def plot_traces_comparisons(title, states, traces_first, traces=None, traces_ref=None, plot_first=True,
							n_plots=None, plots_per_row=2, top_space=1, ylimits=None, save_name=None):
	#region setup
	keys = [key for key in traces_first.columns if key != '']
	if n_plots is None:
		n_plots = len(keys)
	if ylimits is None: ylimits = (
		np.array(
			([traces_first[keys].values] if plot_first else []) +
			([trace[keys].values for trace in traces] if traces is not None else []) +
			([trace_ref[keys].values for trace_ref in traces_ref] if traces_ref is not None else [])
		).min(),
		np.array(
			([traces_first[keys].values] if plot_first else []) +
			([trace[keys].values for trace in traces] if traces is not None else []) +
			([trace_ref[keys].values for trace_ref in traces_ref] if traces_ref is not None else [])
		).max()
	)
	#endregion

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		# ax.set_xticklabels([])
		# ax.set_yticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		axes[i // plots_per_row, i % plots_per_row].plot(np.zeros(traces_first.shape), color='gray', linewidth=0.25)
		axes[i // plots_per_row, i % plots_per_row].set_ylim(ylimits[0], ylimits[1])

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='running_speed')],
		loc='center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
		ncol=1,
		fontsize='large'
	)

	heatmap = states.values.reshape(-1, 1) * 0.1
	#endregion

	#region plot
	i = 0
	for key in keys[:n_plots]:
		if key != '':
			min_val, max_val = ylimits

			ax = axes[i // plots_per_row, i % plots_per_row]
			ax.set_title(f'{key}', fontsize=12)

			ax.imshow(
				heatmap.T, aspect='auto',
				cmap=mcolors.LinearSegmentedColormap.from_list('middle_third',
															   plt.get_cmap('binary')(np.linspace(0, 1 / 2, 256))),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], min_val, max_val]
			)
			ax.plot(traces_first[key], color='black', alpha=1.0 if plot_first else 0.0, linewidth=1.0)
			if traces is not None:
				ax.plot(traces[0][key], color='C0', linewidth=1.0)
				ax.plot(traces[1][key], color='green', linewidth=1.0)
			if traces_ref is not None:
				ax.plot(traces_ref[0][key], color='C0', alpha=0.5, linewidth=1.0)
				ax.plot(traces_ref[1][key], color='green', alpha=0.5, linewidth=1.0)

			for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
				spine.set_visible(True)

			i += 1
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()



#region score
def plot_performance_plot(neuron_scores, neuron_scores_control=None,
						  title='scores', limits=None, plot_size=30, save_name=None):
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	plt.figure(figsize=(plot_size * 1/3, plot_size * 1/3))
	plt.grid(False)
	plt.xlabel('BOM r2 score (median)', fontsize=plot_size * 25/60, labelpad=50)
	plt.ylabel('HGM r2 score (median)', fontsize=plot_size * 25/60)
	plt.title(title)

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
			plt.gca().transData.transform((0, 0)) + (plot_size, 0)
		)[0]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[0],
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (0, plot_size)
		)[1]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[1]
	)

	plt.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--')
	plt.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--')
	plt.plot([min(x_min, y_min), (max(x_max, y_max))], [min(x_min, y_min), max(x_max, y_max)], color='black', linestyle='--')

	#region GFP =============================================================================================================
	boxplot_xticks_gfp, boxplot_xtick_labels_gfp = list(), list()

	for key in neuron_scores_control.keys():
		scores_glm, scores_hgm = neuron_scores_control[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=1.0, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		if len(scores_hgm) > 1:  # change this to 0 to enable median lines for one-neuron points
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
			plt.text(x, -0.035, label, color='red', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_xaxis_transform())
	#endregion =================================================================================================================

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

	#region GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	for key in neuron_scores.keys():
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_glm)
		# if x < high_score_glm_gfp: continue
		y = np.median(scores_hgm)
		if y <= high_score_hgm_gfp: continue

		plt.scatter(scores_glm, scores_hgm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=1.0, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		if len(scores_hgm) > 1:  # change this to 0 to enable median lines for one-neuron points
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
			plt.text(x, -0.035, label, color='blue', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_xaxis_transform())
	#endregion =================================================================================================================

	x_ticks_pos = np.arange(x_min, x_max, x_step)
	plt.xticks(x_ticks_pos, [f'{val:.2f}' for val in x_ticks_pos])
	y_ticks_pos = np.arange(y_min, y_max, y_step)
	plt.yticks(y_ticks_pos, [f'{val:.2f}' for val in y_ticks_pos])

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()


def plot_improvement_plot(neuron_scores, neuron_scores_control=None,
						  title='scores', limits=None, plot_size=30, save_name=None):
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	# plt.figure(figsize=(plot_size * 1/4, plot_size * 1/4))
	plt.figure(figsize=(plot_size * 1/3, plot_size * 1/3))
	plt.grid(False)
	plt.xlabel('BOM r2 score (median)', fontsize=plot_size * 25/60, labelpad=50)
	plt.ylabel('HGM-BOM r2 score improvement (median)', fontsize=plot_size * 25/60)
	plt.title(title)

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
			plt.gca().transData.transform((0, 0)) + (plot_size, 0)
		)[0]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[0],
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (0, plot_size)
		)[1]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[1]
	)

	plt.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--')
	plt.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--')
	# plt.plot([x_min, x_max + x_step], [y_min, y_max + y_step], color='black', linestyle='--')

	#region GFP =============================================================================================================
	boxplot_xticks_gfp, boxplot_xtick_labels_gfp = list(), list()

	for key in neuron_scores_control.keys():
		scores_glm, scores_hgm = neuron_scores_control[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=1.0, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm - scores_glm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		if len(scores_hgm) > 1:  # change this to 0 to enable median lines for one-neuron points
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
			plt.text(x, -0.05, label, color='red', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_xaxis_transform())
	#endregion =================================================================================================================

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

	#region GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	keys = neuron_scores.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_glm)
		# if x < high_score_glm_gfp: continue
		y = np.median(scores_hgm - scores_glm)
		if y <= high_score_hgm_impr_gfp: continue

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=1.0, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		if len(scores_hgm) > 1:  # change this to 0 to enable median lines for one-neuron points
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
			plt.text(x, -0.05, label, color='blue', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_xaxis_transform())
	#endregion =================================================================================================================

	x_ticks_pos = np.arange(x_min, x_max, x_step)
	plt.xticks(x_ticks_pos, [f'{val:.3f}' for val in x_ticks_pos])
	y_ticks_pos = np.arange(y_min, y_max, y_step)
	plt.yticks(y_ticks_pos, [f'{val:.3f}' for val in y_ticks_pos])

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()
#endregion



# temp
def plot_improvement_plot_y(neuron_scores, neuron_scores_control=None,
						  title='scores', limits=None, plot_size=30, save_name=None):
	if neuron_scores_control is None: neuron_scores_control = dict()

	threshold = 0.95

	# plt.figure(figsize=(plot_size * 1/4, plot_size * 1/4))
	plt.figure(figsize=(plot_size * 1/3, plot_size * 1/3))
	plt.grid(False)
	plt.xlabel('BOM r2 score (median)', fontsize=plot_size * 25/60, labelpad=50)
	plt.ylabel('HGM-BOM r2 score improvement (median)', fontsize=plot_size * 25/60)
	plt.title(title)

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
			plt.gca().transData.transform((0, 0)) + (plot_size, 0)
		)[0]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[0],
		plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0)) + (0, plot_size)
		)[1]
		- plt.gca().transData.inverted().transform(
			plt.gca().transData.transform((0, 0))
		)[1]
	)

	plt.plot([x_min, x_max], [0, 0], color='gray', linewidth=1, linestyle='--')
	plt.plot([0, 0], [y_min, y_max], color='gray', linewidth=1, linestyle='--')
	# plt.plot([x_min, x_max + x_step], [y_min, y_max + y_step], color='black', linestyle='--')

	#region GFP =============================================================================================================
	boxplot_xticks_gfp, boxplot_xtick_labels_gfp = list(), list()

	for key in neuron_scores_control.keys():
		scores_glm, scores_hgm = neuron_scores_control[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=1.0, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm - scores_glm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		if len(scores_hgm) > 1:  # change this to 0 to enable median lines for one-neuron points
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
			plt.text(x, -0.05, label, color='red', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_xaxis_transform())
	#endregion =================================================================================================================

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

	#region GCAMP ===========================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()
	boxplot_yticks, boxplot_ytick_labels = list(), list()

	keys = neuron_scores.keys()
	for key in keys:
		scores_glm, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_glm = scores_glm.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_glm)
		# if x < high_score_glm_gfp: continue
		y = np.median(scores_hgm - scores_glm)
		if y <= high_score_hgm_impr_gfp: continue

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=1.0, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')
		boxplot_yticks.append(y)
		boxplot_ytick_labels.append(f'{key}')

		if len(scores_hgm) > 1:  # change this to 0 to enable median lines for one-neuron points
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

	# for x, label in zip(boxplot_xticks, boxplot_xtick_labels):
	# 	if x >= x_min:
	# 		plt.text(x, -0.05, label, color='blue', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
	# 				 transform=plt.gca().get_xaxis_transform())
	for y, label in zip(boxplot_yticks, boxplot_ytick_labels):
		if y >= x_min:
			plt.text(1.01, y, label, color='blue', ha='left', va='center', rotation=0, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_yaxis_transform())


	ax2 = plt.gca().secondary_yaxis("right")
	ax2.set_yticks(boxplot_yticks)
	ax2.set_yticklabels([])
	plt.gca().grid(False)




	#endregion =================================================================================================================

	x_ticks_pos = np.arange(x_min, x_max, x_step)
	plt.xticks(x_ticks_pos, [f'{val:.2f}' for val in x_ticks_pos])
	y_ticks_pos = np.arange(y_min, y_max, y_step)
	plt.yticks(y_ticks_pos, [f'{val:.2f}' for val in y_ticks_pos])

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()
