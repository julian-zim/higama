import os
import math
import numpy as np
import pandas as pd
from matplotlib import patches, pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib import cm


def subplot(title, datas,
			n_plots=None, plots_per_row=4, top_space=1, save_name=None):
	#region setup
	if type(datas) in (pd.DataFrame, pd.Series):
		datas = [datas]

	if n_plots is None:
		n_plots = len(datas[0].columns)
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

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel / 2, title, ha='center', va='center', fontsize=20)
	#endregion

	#region plot
	for data in datas:
		if data is not None:
			for i, key in enumerate(data):
				y = data[key]
				ax = axes[i // plots_per_row, i % plots_per_row]
				ax.set_title(key, fontsize=8)
				ax.plot(y)
				ax.grid(True)
				for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
					spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()


#region manifolds
def plot_manifolds(filenames, modes, states, vars_expl, vars_expl_rs=None,
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

	colormap = ListedColormap(plt.get_cmap('hsv')(np.linspace(0, 1, plt.get_cmap('hsv').N))[:, :3] * 0.8, name='hsv_dark')
	fig.legend(
		handles=[(patches.Patch(color=colormap(i / len(states[0].columns)), alpha=0.3, label=states[0].columns[i])) for i in range(len(states[0].columns))],
		loc='center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 4 / 6),
		ncol=1,
		fontsize='large'
	)
	#endregion

	#region plot
	scale = [] #[-1, 0, 1]
	normalizer = plt.Normalize(0, len(states[0].columns))
	for i, (modes_, states_) in enumerate(zip(modes[:n_plots], states[:n_plots])):
		lc_points = np.array([modes_[f'{key}1'], modes_[f'{key}2']]).T.reshape(-1, 1, 2)
		lc_segments = np.concatenate([lc_points[:-1], lc_points[1:]], axis=1)
		lc_colors = np.argmax(states_.values, axis=1)[:-1]
		lc = LineCollection(lc_segments, cmap=colormap, norm=normalizer)
		lc.set_array(lc_colors)
		lc.set_linewidth(2)

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}\n{sum(vars_expl[i]) * 100:.1f}%' +
										  (f' ({sum(vars_expl_rs[i]) * 100:.1f}%)' if vars_expl_rs is not None else ''),
										  fontsize=10.5)

		ax.plot(modes_[f'{key}1'], modes_[f'{key}2'], alpha=0)
		ax.add_collection(lc)
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


def plot_manifolds_3d(filenames, modes, states, vars_expl, vars_expl_rs=None,
					  title='manifolds', n_plots=None, plots_per_row=4, top_space=3, animate=False, save_name=None):
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
							 figsize=(figwidth, figheight), squeeze=False,
							 subplot_kw={'projection': '3d'})
	for ax in axes.ravel():
		ax.set_xticks([])
		ax.set_xticklabels([])
		ax.set_yticks([])
		ax.set_yticklabels([])
		ax.set_zticks([])
		ax.set_zticklabels([])
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 6, title, ha='center', va='center', fontsize=20)

	colormap = ListedColormap(plt.get_cmap('hsv')(np.linspace(0, 1, plt.get_cmap('hsv').N))[:, :3] * 0.8, name='hsv_dark')
	fig.legend(
		handles=[(patches.Patch(color=colormap(i / len(states[0].columns)), alpha=0.3, label=states[0].columns[i])) for i in range(len(states[0].columns))],
		loc='center',
		bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 6),
		ncol=1,
		fontsize='large'
	)
	#endregion

	#region plot
	scale = [-1, 0, 1]
	normalizer = plt.Normalize(0, len(states[0].columns))
	for i, (modes_, states_) in enumerate(zip(modes[:n_plots], states[:n_plots])):
		lc_points = np.array([modes_[f'{key}1'], modes_[f'{key}2'], modes_[f'{key}3']]).T.reshape(-1, 1, 3)
		lc_segments = np.concatenate([lc_points[:-1], lc_points[1:]], axis=1)
		lc_colors = np.argmax(states_.values, axis=1)[:-1]
		lc = Line3DCollection(lc_segments, cmap=colormap, norm=normalizer)
		lc.set_array(lc_colors)
		lc.set_linewidth(1)

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}\n{sum(vars_expl[i]) * 100:.1f}%' +
					 (f' ({sum(vars_expl_rs[i]) * 100:.1f}%)' if vars_expl_rs is not None else ''),
					 fontsize=10.5)

		# ax.plot(modes_[f'{key}1'], modes_[f'{key}2'], modes_[f'{key}3'], alpha=0)
		ax.add_collection3d(lc)
		# ax.set_xlabel('test')
		ax.set_xticks(scale)
		ax.set_xticklabels(scale)
		# ax.set_ylabel('test')
		ax.set_yticks(scale)
		ax.set_yticklabels(scale)
		# ax.set_zlabel('test')
		ax.set_zticks(scale)
		ax.set_zticklabels(scale)
		ax.grid(True)
		for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel)
	# fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None:
		if animate:
			def update(frame):
				for i, ax in enumerate(axes.ravel()):
					ax.view_init(elev=30, azim=frame + i * 45)  # offset if you want
				return []  # axes.ravel()
			anim = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50, blit=False)
			anim.save(f'./out/plots/{save_name}.gif', writer='pillow', fps=30)
		else:
			plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()

	if vars_expl_rs is not None:
		print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}% ({np.mean(np.array(vars_expl_rs)) * 100:.1f}%)')
	else:
		print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}%')


def plot_manifolds_3d_sep(filenames, modes, states, vars_expl, vars_expl_rs=None,
						  titles=None, n_plots=None, top_space=3, animate=False, save_name=None):
	if n_plots is None:
		n_plots = len(filenames)
	if titles is None:
		titles = ['manifold'] * n_plots
	if save_name is not None:
		os.makedirs(f'./out/plots/{save_name}', exist_ok=True)

	for i in range(n_plots):
		plot_manifolds_3d([filenames[i]], [modes[i]], [states[i]], [vars_expl[i]], vars_expl_rs=[vars_expl_rs[i]] if vars_expl_rs is not None else None,
						  title=titles[i], n_plots=1, plots_per_row=1, top_space=top_space, animate=animate,
						  save_name=f'{save_name}/{filenames[i]}' if save_name is not None else None)
#endregion


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
def plot_traces(title, states, traces,
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

	fig.legend(
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='reversal')],
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
			ax.plot(traces[0][key], color='black', linewidth=0.5)
			for trace in traces[1:]:
				ax.plot(trace[key], linewidth=0.5)

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
	if ylimits is None: ylimits = (
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
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='reversal')],
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
			ax.plot(traces_first[key], color='black', alpha=1.0 if plot_first else 0.0, linewidth=0.5)
			if traces_splits is not None:
				colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
				for j in range(len(traces_splits)):
					color = next(colors)
					ax.plot(traces_splits[j][key], color=color, linewidth=0.5)
					if traces_splits_ref is not None:
						ax.plot(traces_splits_ref[j][key], color=color, alpha=0.5, linewidth=0.5)

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
		handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label='reversal')],
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
			ax.plot(traces_first[key], color='black', alpha=1.0 if plot_first else 0.0, linewidth=0.5)
			if traces is not None:
				ax.plot(traces[0][key], color='C0', linewidth=0.5)
				ax.plot(traces[1][key], color='green', linewidth=0.5)
			if traces_ref is not None:
				ax.plot(traces_ref[0][key], color='C0', alpha=0.5, linewidth=0.5)
				ax.plot(traces_ref[1][key], color='green', alpha=0.5, linewidth=0.5)

			for spine in axes[i // plots_per_row, i % plots_per_row].spines.values():
				spine.set_visible(True)

			i += 1
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()
#endregion



def plot_heatmap_stack(traces, states, label_bot, label_top, height_ratios=(1, 4),
					   save_name=None):
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
	if save_name is not None: plt.savefig(f'./out/plots/{save_name}.png', dpi=300, bbox_inches='tight')
	plt.show()


def plot_mcos(bars, bar_labels, labels, order,
			  normalized=True, log_scale=False, save_name=None):
	if log_scale:
		# if normalized:
		# 	print('Warning: Ignoring log scale as data is expected to be normalized. Set \'normalized=False\' in order to prevent this.')
		# else:
		# 	sums = bars.sum(axis=0)
		# 	# print('sums:\n', sums)
		# 	sums_log = sums.map(lambda x: math.log(x * 100 + 0.1) / math.log(10) + 1)
		# 	# print('sums_log:\n', sums_log)
		# 	for label in bars.columns:
		# 		# print(f'bars[{label}]\n', bars[label])
		# 		# print(f'sums[{label}]\n', sums[label])
		# 		# print(f'bars[{label}] / sums[{label}]\n', bars[label] / sums[label])
		# 		# print(f'sums_log[{label}]\n', sums_log[label])
		# 		# print(f'bars[{label}] / sums[{label}] * sums_log[{label}]\n', bars[label] / sums[label] * sums_log[label])
		# 		bars[label] = bars[label] / sums[label] * sums_log[label]
		raise NotImplementedError('ToDo: Implement Log scaling.')

	bar_width = 10
	max_y = 1 if normalized else bars.values.T.sum(axis=1).max()

	index_orders = [
		['m', 'c', 'o'].index(order[0]),
		['m', 'c', 'o'].index(order[1]),
		['m', 'c', 'o'].index(order[2])
	]
	color_pool = [cm.get_cmap('rocket')(0.4), cm.get_cmap('rocket')(0.9), 'gray']
	colors = [color_pool[idx] for idx in index_orders]
	bar_labels_color_pool = [cm.get_cmap('Greys')(v) for v in [0.1, 0.9, 0.1]]
	bar_labels_colors = [bar_labels_color_pool[idx] for idx in index_orders]
	# bg_color = (0.9, 0.9, 0.9)

	if normalized:
		bar_labels_positions = [
			[0.025 * max_y for _ in bars.loc[order[0]]],
			[y1 + y2 - 0.025 * max_y * y2 for y1, y2 in zip(bars.loc[order[0]], bars.loc[order[1]])],
			[0.975 * max_y for _ in bars.loc[order[2]]]
		]
	x_positions = list((idx + 0.5) * bar_width for idx in range(len(labels)))
	y_positions = [
		[0 for _ in bars.loc[order[0]]],
		[y1 for y1 in bars.loc[order[0]]],
		[(y1 + y2) for y1, y2 in zip(bars.loc[order[0]], bars.loc[order[1]])]
	]

	# plt.rcParams['figure.facecolor'] = bg_color
	# plt.rcParams['axes.facecolor'] = bg_color
	plt.figure(figsize=(39, 13))

	for idx in range(3):
		plt.bar(x_positions, bars.loc[order[idx]], width=bar_width, bottom=y_positions[idx], color=colors[idx], edgecolor='none')

	if normalized:
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

	plt.yticks(np.arange(0, 1.05, 0.05) * max_y, labels=[f'{val:.1f}%' for val in np.arange(0, 105, 5) * max_y])
	plt.xticks(x_positions, labels)
	plt.xlim(0, x_positions[-1] + bar_width / 2)
	plt.ylim(0, max_y)

	plt.gca().set_axisbelow(False)
	plt.grid(False)
	plt.grid(axis='y', linestyle='--', linewidth=0.5, color=(0.0, 0.0, 0.0))

	for key in bars.T.columns:
		print(f'Average {key}: {np.mean(bars.T[key].values) * 100:.1f}%')

	# plt.rcParams.update(plt.rcParamsDefault)

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

		plt.scatter(scores_glm, scores_hgm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		if len(scores_hgm) > 0:  # change this to 1 to disable median lines for one-neuron points
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
		if y < high_score_hgm_gfp: continue

		plt.scatter(scores_glm, scores_hgm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		if len(scores_hgm) > 0:  # change this to 1 to disable median lines for one-neuron points
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

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm - scores_glm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		if len(scores_hgm) > 0:  # change this to 1 to disable median lines for one-neuron points
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
		if y < high_score_hgm_impr_gfp: continue

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		if len(scores_hgm) > 0:  # change this to 1 to disable median lines for one-neuron points
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

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=0.5, alpha=0.5)

		x = np.median(scores_glm)
		y = np.median(scores_hgm - scores_glm)

		boxplot_xticks_gfp.append(x)
		boxplot_xtick_labels_gfp.append(f'{key}')

		if len(scores_hgm) > 0:  # change this to 1 to disable median lines for one-neuron points
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
		if y < high_score_hgm_impr_gfp: continue

		plt.scatter(scores_glm, scores_hgm - scores_glm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=0.5, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')
		boxplot_yticks.append(y)
		boxplot_ytick_labels.append(f'{key}')

		if len(scores_hgm) > 0:  # change this to 1 to disable median lines for one-neuron points
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
