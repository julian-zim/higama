import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mplcolors, patches, ticker, cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def subplot(datas,
			title, n_plots=None, plots_per_row=4, top_space=1,
			save_path=None):
	if type(datas) in (pd.DataFrame, pd.Series):
		datas = [datas]

	if n_plots is None:
		n_plots = len(datas[0].columns)

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
			for i, key in enumerate(data.columns[:n_plots]):
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

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()


#region manifolds

def plot_manifolds_gradient(filenames, modes, coloring, vars_expl=None, vars_expl_rs=None,
							title='manifolds', n_plots=None, plots_per_row=4, top_space=2,
							save_path=None):
	if n_plots is None:
		n_plots = len(filenames)

	key = modes[0].columns[0][:-1]

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

	colormap = mplcolors.ListedColormap(plt.get_cmap('plasma')(np.linspace(0, 1, plt.get_cmap('hsv').N))[:, :3] * 0.8, name='hsv_dark')
	# fig.legend(
	# 	handles=[(patches.Patch(color=colormap(i / len(coloring[0].columns)), alpha=0.3, label=coloring[0].columns[i])) for i in range(len(coloring[0].columns))],
	# 	loc='center',
	# 	bbox_to_anchor=(0.5, 1. - top_space_rel * 4 / 6),
	# 	ncol=1,
	# 	fontsize='large'
	# )
	#endregion

	#region plot
	scale = [] #[-1, 0, 1]
	normalizer = plt.Normalize(0, len(coloring[0].columns))
	for i, (modes_, colors_) in enumerate(zip(modes[:n_plots], coloring[:n_plots])):
		i0 = 1
		if f'{key}0' in modes_.columns:
			i0 = 0

		lc_points = np.array([modes_[f'{key}{i0}'], modes_[f'{key}{i0+1}']]).T.reshape(-1, 1, 2)
		lc_segments = np.concatenate([lc_points[:-1], lc_points[1:]], axis=1)
		lc_colors = colors_.values[:-1].reshape(-1)
		lc = LineCollection(lc_segments, cmap=colormap, norm=normalizer)
		lc.set_array(lc_colors)
		lc.set_linewidth(2)

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}' +
					 (f'\n{sum(vars_expl[i]) * 100:.1f}%' if vars_expl is not None else '') +
					 (f' ({sum(vars_expl_rs[i]) * 100:.1f}%)' if vars_expl is not None and vars_expl_rs is not None else ''),
					 fontsize=10.5)

		ax.plot(modes_[f'{key}{i0}'], modes_[f'{key}{i0+1}'], alpha=0)
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

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

	if vars_expl is not None:
		if vars_expl_rs is not None:
			print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}% ({np.array(vars_expl_rs).sum(axis=1).mean() * 100:.1f}%)')
		else:
			print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}%')


def plot_manifolds(filenames, modes, states=None, vars_expl=None, vars_expl_rs=None,
				   title='manifolds', n_plots=None, plots_per_row=4, top_space=2,
				   save_path=None):
	if n_plots is None:
		n_plots = len(filenames)

	key = modes[0].columns[0][:-1]

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

	if states is not None:
		colormap = mplcolors.ListedColormap(plt.get_cmap('hsv')(np.linspace(0, 1, plt.get_cmap('hsv').N))[:, :3] * 0.8, name='hsv_dark')
		fig.legend(
			handles=[(patches.Patch(color=colormap(i / len(states[0].columns)), alpha=0.3, label=states[0].columns[i])) for i in range(len(states[0].columns))],
			loc='center',
			bbox_to_anchor=(0.5, 1. - top_space_rel * 4 / 6),
			ncol=1,
			fontsize='large'
		)
	else:
		colormap = mplcolors.ListedColormap(plt.get_cmap('bwr')(np.linspace(0, 1, plt.get_cmap('bwr').N))[:, :3] * 1.0, name='bwr_const')
		states = [pd.DataFrame(data=np.zeros((mode.shape[0], 1), dtype=np.float16), columns=['-']) for mode in modes]
	#endregion

	#region plot
	scale = [] #[-1, 0, 1]
	normalizer = plt.Normalize(0, len(states[0].columns))
	for i, (modes_, states_) in enumerate(zip(modes[:n_plots], states[:n_plots])):
		i0 = 1
		if f'{key}0' in modes_.columns:
			i0 = 0

		lc_points = np.array([modes_[f'{key}{i0}'], modes_[f'{key}{i0+1}']]).T.reshape(-1, 1, 2)
		lc_segments = np.concatenate([lc_points[:-1], lc_points[1:]], axis=1)
		lc_colors = np.argmax(states_.values, axis=1)[:-1]
		lc = LineCollection(lc_segments, cmap=colormap, norm=normalizer)
		lc.set_array(lc_colors)
		lc.set_linewidth(2)

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}' +
					 (f'\n{sum(vars_expl[i]) * 100:.1f}%' if vars_expl is not None else '') +
					 (f' ({sum(vars_expl_rs[i]) * 100:.1f}%)' if vars_expl is not None and vars_expl_rs is not None else ''),
					 fontsize=10.5)

		ax.plot(modes_[f'{key}{i0}'], modes_[f'{key}{i0+1}'], alpha=0)
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

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

	if vars_expl is not None:
		if vars_expl_rs is not None:
			print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}% ({np.array(vars_expl_rs).sum(axis=1).mean() * 100:.1f}%)')
		else:
			print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}%')


def plot_manifolds_3d(filenames, modes, states=None, vars_expl=None, vars_expl_rs=None,
					  title='manifolds', n_plots=None, plots_per_row=4, top_space=3, animate=False,
					  save_path=None):
	if n_plots is None:
		n_plots = len(filenames)

	key = modes[0].columns[0][:-1]

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

	if states is not None:
		colormap = mplcolors.ListedColormap(plt.get_cmap('hsv')(np.linspace(0, 1, plt.get_cmap('hsv').N))[:, :3] * 0.8, name='hsv_dark')
		fig.legend(
			handles=[(patches.Patch(color=colormap(i / len(states[0].columns)), alpha=0.3, label=states[0].columns[i])) for i in range(len(states[0].columns))],
			loc='center',
			bbox_to_anchor=(0.5, 1. - top_space_rel * 4 / 6),
			ncol=1,
			fontsize='large'
		)
	else:
		colormap = mplcolors.ListedColormap(plt.get_cmap('bwr')(np.linspace(0, 1, plt.get_cmap('bwr').N))[:, :3] * 1.0, name='bwr_const')
		states = [pd.DataFrame(data=np.zeros((mode.shape[0], 1), dtype=np.float16), columns=['-']) for mode in modes]
	#endregion

	#region plot
	scale = [-1, 0, 1]
	normalizer = plt.Normalize(0, len(states[0].columns))
	for i, (modes_, states_) in enumerate(zip(modes[:n_plots], states[:n_plots])):
		i0 = 1
		if f'{key}0' in modes_.columns:
			i0 = 0

		lc_points = np.array([modes_[f'{key}{i0}'], modes_[f'{key}{i0+1}'], modes_[f'{key}{i0+2}']]).T.reshape(-1, 1, 3)
		lc_segments = np.concatenate([lc_points[:-1], lc_points[1:]], axis=1)
		lc_colors = np.argmax(states_.values, axis=1)[:-1]
		lc = Line3DCollection(lc_segments, cmap=colormap, norm=normalizer)
		lc.set_array(lc_colors)
		lc.set_linewidth(1)

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}' +
					 (f'\n{sum(vars_expl[i]) * 100:.1f}%' if vars_expl is not None else '') +
					 (f' ({sum(vars_expl_rs[i]) * 100:.1f}%)' if vars_expl is not None and vars_expl_rs is not None else ''),
					 fontsize=10.5)

		# ax.plot(modes_[f'{key}{i0}'], modes_[f'{key}{i0 + 1}'], modes_[f'{key}{i0 + 2}'], alpha=0)
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

	if save_path is not None:
		if animate:
			def update(frame):
				for i, ax in enumerate(axes.ravel()):
					ax.view_init(elev=30, azim=frame + i * 45)  # offset if you want
				return []  # axes.ravel()
			anim = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50, blit=False)
			anim.save(f'{save_path}.gif', writer='pillow', fps=30)
		else:
			plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

	if vars_expl is not None:
		if vars_expl_rs is not None:
			print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}% ({np.array(vars_expl_rs).sum(axis=1).mean():.1f}%)')
		else:
			print(f'{np.array(vars_expl).sum(axis=1).mean() * 100:.1f}%')


def plot_manifolds_3d_sep(filenames, modes, states=None, vars_expl=None, vars_expl_rs=None,
						  titles=None, n_plots=None, top_space=3, animate=False,
						  save_path=None):
	if n_plots is None:
		n_plots = len(filenames)
	if titles is None:
		titles = ['manifold'] * n_plots
	if save_path is not None:
		os.makedirs(save_path, exist_ok=True)

	for i in range(n_plots):
		plot_manifolds_3d([filenames[i]], [modes[i]], [states[i]] if states is not None else None,
						  [vars_expl[i]] if vars_expl is not None else None, [vars_expl_rs[i]] if vars_expl_rs is not None else None,
						  title=titles[i], n_plots=1, plots_per_row=1, top_space=top_space, animate=animate,
						  save_path=f'{save_path}/{filenames[i]}' if save_path is not None else None)

#endregion


#region traces

"""
Plots two pairs of traces in red and green black, with the respective first one being opaque and the second transparent
"""
def plot_traces_splits_pairs(traces, traces_reference=None, traces_initial=None, states=None, label='state',
							title='trace pairs per split', ylimits=None, n_plots=None, plots_per_row=1, top_space=1,
							save_path=None):
	columns = [column for column in traces[0].columns if column != '']
	if n_plots is None:
		n_plots = len(columns)
	if ylimits is None:
		traces_all = np.array(
			[trace[columns].values for trace in traces] +
			([trace_ref[columns].values for trace_ref in traces_reference] if traces_reference is not None else []) +
			([traces_initial[columns].values] if traces_initial is not None else [])
		)
		ylimits = (traces_all.min(), traces_all.max())

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_ylim(ylimits)
		ax.plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	if states is not None:
		heatmap = states.values.reshape(-1, 1) * 0.1
		fig.legend(
			handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label=label)],
			loc='center',
			bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
			ncol=1,
			fontsize='large'
		)
	#endregion

	#region plot
	for i in range(len(columns[:n_plots])):
		column = columns[i]

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{column}', fontsize=12)

		if traces_initial is not None:
			ax.plot(traces_initial[column], color='black', linewidth=1.0)
		if traces_reference is not None:
			ax.plot(traces_reference[0][column], color='C0', alpha=0.5, linewidth=1.0)
			ax.plot(traces_reference[1][column], color='green', alpha=0.5, linewidth=1.0)
		ax.plot(traces[0][column], color='C0', linewidth=1.0)
		ax.plot(traces[1][column], color='green', linewidth=1.0)

		if states is not None:
			ax.imshow(
				heatmap.T, aspect='auto',
				cmap=mplcolors.LinearSegmentedColormap.from_list(
					'middle_third',
					plt.get_cmap('binary')(np.linspace(0, 1 / 2, 256))
				),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], *ylimits]
			)

		for spine in ax.spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

"""
Plots single trace in black and optional splits in cycling colors, with the first being opaques and the second transparents
"""
def plot_traces_splits(traces, traces_reference=None, traces_initial=None, states=None, label='state',
					   title='traces per split', ylimits=None, n_plots=None, plots_per_row=1, top_space=1,
					   save_path=None):
	columns = [column for column in traces[0].columns if column != '']
	if n_plots is None:
		n_plots = len(columns)
	if ylimits is None:
		traces_all = np.array(
			[trace[columns].values for trace in traces] +
			([trace_reference[columns].values for trace_reference in traces_reference] if traces_reference is not None else []) +
			([traces_initial[columns].values] if traces_initial is not None else [])
		)
		ylimits = (traces_all.min(), traces_all.max())

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(n_plots):
		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_ylim(ylimits)
		ax.plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	if states is not None:
		heatmap = states.values.reshape(-1, 1) * 0.1
		fig.legend(
			handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label=label)],
			loc='center',
			bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
			ncol=1,
			fontsize='large'
		)
	#endregion

	#region plot
	for i in range(len(columns[:n_plots])):
		column = columns[i]

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{column}', fontsize=12)

		if traces_initial is not None:
			ax.plot(traces_initial[column], color='black', linewidth=1.0)
		colors = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
		for j in range(len(traces)):
			color = next(colors)
			if traces_reference is not None:
				ax.plot(traces_reference[j][column], color=color, alpha=0.5, linewidth=1.0)
			ax.plot(traces[j][column], color=color, linewidth=1.0)

		if states is not None:
			ax.imshow(
				heatmap.T, aspect='auto',
				cmap=mplcolors.LinearSegmentedColormap.from_list(
					'middle_third',
					plt.get_cmap('binary')(np.linspace(0, 1 / 2, 256))
				),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], *ylimits]
			)

		for spine in ax.spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

"""
Plots traces cycling color according to pyplots policy, with the first one black
"""
def plot_traces(traces, states=None, label='state',
				title='traces', ylimits=None, n_plots=None, plots_per_row=1, top_space=1,
				save_path=None):
	if type(traces) is not list:
		traces = [traces]
	columns = [column for column in traces[0].columns if column != '']
	if n_plots is None:
		n_plots = len(columns)
	if ylimits is None:
		traces_all = np.array([trace[columns].values for trace in traces])
		ylimits = (traces_all.min(), traces_all.max())

	#region plot setup
	fig_size = (15, 3)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	for i in range(min(len(columns), n_plots)):
		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_ylim(ylimits)
		ax.plot(np.zeros(traces[0].shape), color='gray', linewidth=0.25)

	top_space_rel = top_space / figheight
	#fig.text(0.5, 1., 'TOP', c='white', ha='center', va='top')
	fig.text(0.5, 1. - top_space_rel * 1 / 4, title, ha='center', va='center', fontsize=20)

	if states is not None:
		heatmap = states.values.reshape(-1, 1) * 0.1
		fig.legend(
			handles=[patches.Patch(color=plt.get_cmap('binary')(0.5), alpha=0.3, label=label)],
			loc='center',
			bbox_to_anchor=(0.5, 1. - top_space_rel * 3 / 4),
			ncol=1,
			fontsize='large'
		)
	#endregion

	#region plot
	for i in range(len(columns[:n_plots])):
		column = columns[i]

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{column}', fontsize=12)

		ax.plot(traces[0][column].values, color='black', linewidth=1.0)
		for trace in traces[1:]:
			ax.plot(trace[column].values, linewidth=1.0)

		if states is not None:
			ax.imshow(
				heatmap.T, aspect='auto',
				cmap=mplcolors.LinearSegmentedColormap.from_list(
					'middle_third',
					plt.get_cmap('binary')(np.linspace(0, 1/2, 256))
				),
				interpolation='nearest', alpha=0.3, extent=[0, heatmap.shape[0], *ylimits]
			)

		for spine in ax.spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel, hspace=0.5)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

"""
plot_traces wrapper for numpy arrays
"""
def plot_traces_np(traces, *args, **kwargs):
	if type(traces) is not list:
		traces = [traces]
	traces = [trace if type(trace) is pd.DataFrame else pd.DataFrame(trace) for trace in traces]
	plot_traces(traces, *args, **kwargs)

#endregion


#region losses

def plot_losses_splits(filenames, losses, losses_reference=None,
					   title='losses', ylimits=None, n_plots=None, plots_per_row=4, top_space=1,
					   save_path=None):
	if n_plots is None:
		n_plots = len(filenames)

	#region plot setup
	fig_size = (4, 4)

	plots_per_row = min(n_plots, plots_per_row)
	plots_per_col = math.ceil(n_plots / plots_per_row)
	figheight = fig_size[1] * plots_per_col + top_space
	figwidth = fig_size[0] * plots_per_row
	fig, axes = plt.subplots(plots_per_col, plots_per_row,
							 figsize=(figwidth, figheight), squeeze=False)
	for ax in axes.ravel():
		ax.grid(False)
		for spine in ax.spines.values():
			spine.set_visible(False)
	if ylimits is not None:
		for i in range(min(len(losses), n_plots)):
			ax = axes[i // plots_per_row, i % plots_per_row]
			if type(ylimits) is tuple:
				ax.set_ylim(ylimits)
			else:
				losses_min, losses_max = losses[i].min(), losses[i].max()
				shift = ylimits * 0.1
				# ylimits_i = (0, losses_min + (losses_max - losses_min) * ylimits)
				ylimits_i = (losses_min - shift, losses_min + (losses_max - losses_min) * ylimits + shift)
				ax.set_ylim(ylimits_i)

	top_space_rel = top_space / figheight
	fig.text(0.5, 1. - top_space_rel / 2, title, ha='center', va='center', fontsize=20)
	#endregion

	#region plot
	for i in range(len(losses[:n_plots])):
		losses_per_split_m1 = losses[i]
		if losses_reference is not None:
			losses_per_split_m2 = losses_reference[i]

		ax = axes[i // plots_per_row, i % plots_per_row]
		ax.set_title(f'{filenames[i]}\n' +
					 f'(model: {np.mean(losses_per_split_m1[:, -1], axis=0) * 100:.2f}E-2' +
					 (f', comp: {np.mean(losses_per_split_m2[:, -1], axis=0) * 100:.2f}E-2)' if losses_reference is not None else ''),
					 fontsize=10)

		colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
		for j in range(losses_per_split_m1.shape[0]):
			ax.plot(losses_per_split_m1[j, :], color=colors[j])
			if losses_reference is not None:
				ax.plot(losses_per_split_m2[j, :], color=colors[j], alpha=0.5)

		ax.grid(True)
		for spine in ax.spines.values():
			spine.set_visible(True)
	#endregion

	fig.subplots_adjust(top=1. - top_space_rel)
	fig.tight_layout(rect=(0., 0., 1., 1. - top_space_rel))

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

def plot_losses(filename, losses, *args, **kwargs):
	plot_losses_splits(np.array([filename]), [losses], *args, **kwargs)

def plot_loss(filename, loss, *args, **kwargs):
	plot_losses(filename, np.array([loss]), *args, **kwargs)

#endregion


#region score

def plot_performance_plot(neuron_scores, neuron_scores_control=None, control_cutoff=True,
						  title='scores', limits=None, force_medians=False, plot_size=30,
						  save_path=None):
	#region setup
	threshold = 0.95

	plt.figure(figsize=(plot_size * 1/3, plot_size * 1/3))
	plt.title(title)
	plt.xlabel('BOM r2 score (median)', fontsize=plot_size * 25/60, labelpad=50)
	plt.ylabel('HGM r2 score (median)', fontsize=plot_size * 25/60)
	plt.grid(False)

	if limits is None:
		x_all = np.array([score for scores_bom, scores_hgm in neuron_scores.values() for score in scores_bom])
		y_all = np.array([score for scores_bom, scores_hgm in neuron_scores.values() for score in scores_hgm])

		if neuron_scores_control is not None:
			x_all = np.concatenate(
				[x_all,
				 np.array([score for scores_bom, scores_hgm in neuron_scores_control.values() for score in scores_bom])]
			)
			y_all = np.concatenate(
				[y_all,
				 np.array([score for scores_bom, scores_hgm in neuron_scores_control.values() for score in scores_hgm])]
			)

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

	#endregion

	#region CONTROL ===========================================================================================================
	if neuron_scores_control is not None:
		boxplot_xticks_control, boxplot_xtick_labels_control = list(), list()

		for key in neuron_scores_control.keys():
			scores_bom, scores_hgm = neuron_scores_control[key]
			if len(scores_hgm.shape) == 0:
				scores_bom = scores_bom.reshape(1, 1)
				scores_hgm = scores_hgm.reshape(1, 1)

			x = np.median(scores_bom)
			y = np.median(scores_hgm)

			plt.scatter(scores_bom, scores_hgm, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=1.0, alpha=0.5)

			boxplot_xticks_control.append(x)
			boxplot_xtick_labels_control.append(f'{key}')

			if len(scores_hgm) > 1 or force_medians:
				boxplot_v = plt.boxplot(scores_hgm, positions=[x], widths=bp_pxs[0], patch_artist=True,
										showfliers=False)
				boxplot_h = plt.boxplot(scores_bom, positions=[y], widths=bp_pxs[1], patch_artist=True,
										showfliers=False, vert=False)
				for box in [boxplot_v, boxplot_h]:
					for patch in box['boxes']:
						patch.set_facecolor('tomato')
						patch.set_alpha(0.5)
					for median in box['medians']:
						median.set_color('lightsteelblue')
						median.set_linewidth(1)

		for x, label in zip(boxplot_xticks_control, boxplot_xtick_labels_control):
			if x_min <= x <= x_max:
				plt.text(x, -0.035, label, color='red', ha='center', va='top', rotation=45, fontsize=plot_size * 15/40,
						 transform=plt.gca().get_xaxis_transform())
	#endregion ================================================================================================================

	if neuron_scores_control is not None:
		# high_score_bom_control = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_control.values()]), threshold)
		high_score_hgm_control = np.quantile(np.concatenate([pair[1] for pair in neuron_scores_control.values()]), threshold)
		plt.plot([x_min, x_max + x_step], [high_score_hgm_control, high_score_hgm_control], color='black', linestyle='--')

	#region GCAMP =============================================================================================================
	boxplot_xticks, boxplot_xtick_labels = list(), list()

	for key in neuron_scores.keys():
		scores_bom, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_bom = scores_bom.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_bom)
		y = np.median(scores_hgm)
		# if neuron_scores_control is not None and control_cutoff and x <= high_score_bom_control:
		# 	continue
		if neuron_scores_control is not None and control_cutoff and y <= high_score_hgm_control:
			continue

		plt.scatter(scores_bom, scores_hgm, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=1.0, alpha=0.5)

		boxplot_xticks.append(x)
		boxplot_xtick_labels.append(f'{key}')

		if len(scores_hgm) > 1 or force_medians:
			boxplot_v = plt.boxplot(scores_hgm, positions=[x], widths=bp_pxs[0], patch_artist=True,
									showfliers=False)
			boxplot_h = plt.boxplot(scores_bom, positions=[y], widths=bp_pxs[1], patch_artist=True,
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

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

def plot_improvement_plot(neuron_scores, neuron_scores_control=None, control_cutoff=True,
						  title='scores', limits=None, force_medians=False, plot_size=30,
						  save_path=None):
	#region setup
	threshold = 0.95

	plt.figure(figsize=(plot_size * 1/3, plot_size * 1/3))
	plt.title(title)
	plt.xlabel('BOM r2 score (median)', fontsize=plot_size * 25/60, labelpad=50)
	plt.ylabel('HGM-BOM r2 score improvement (median)', fontsize=plot_size * 25/60)
	plt.grid(False)
	plt.gca().grid(False)

	if limits is None:
		x_all = np.array([score for scores_bom, scores_hgm in neuron_scores.values() for score in scores_bom])
		y_all = np.array([score for scores_bom, scores_hgm in neuron_scores.values() for score in scores_hgm - scores_bom])

		if neuron_scores_control is not None:
			x_all = np.concatenate(
				[x_all,
				 np.array([score for scores_bom, scores_hgm in neuron_scores_control.values() for score in scores_bom])]
			)
			y_all = np.concatenate(
				[y_all,
				 np.array([score for scores_bom, scores_hgm in neuron_scores_control.values() for score in scores_hgm - scores_bom])]
			)

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

	#endregion

	#region CONTROL =========================================================================================================
	if neuron_scores_control is not None:
		boxplot_yticks_control, boxplot_ytick_labels_control = list(), list()

		for key in neuron_scores_control.keys():
			scores_bom, scores_hgm = neuron_scores_control[key]
			if len(scores_hgm.shape) == 0:
				scores_bom = scores_bom.reshape(1, 1)
				scores_hgm = scores_hgm.reshape(1, 1)

			x = np.median(scores_bom)
			y = np.median(scores_hgm - scores_bom)

			plt.scatter(scores_bom, scores_hgm - scores_bom, s=plot_size * 2, c='tomato', edgecolor='black', linewidth=1.0, alpha=0.5)

			boxplot_yticks_control.append(y)
			boxplot_ytick_labels_control.append(f'{key}')

			if len(scores_hgm) > 1 or force_medians:
				boxplot_v = plt.boxplot(scores_hgm - scores_bom, positions=[x], widths=bp_pxs[0], patch_artist=True,
										showfliers=False)
				boxplot_h = plt.boxplot(scores_bom, positions=[y], widths=bp_pxs[1], patch_artist=True,
										showfliers=False, vert=False)
				for box in [boxplot_v, boxplot_h]:
					for patch in box['boxes']:
						patch.set_facecolor('tomato')
						patch.set_alpha(0.5)
					for median in box['medians']:
						median.set_color('lightsteelblue')
						median.set_linewidth(1)

		for y, label in zip(boxplot_yticks_control, boxplot_ytick_labels_control):
			if y >= y_min:
				plt.text(1.01, y, label, color='red', ha='left', va='center', rotation=0, fontsize=plot_size * 15 / 40,
						 transform=plt.gca().get_yaxis_transform())
		ax2 = plt.gca().secondary_yaxis('right')
		ax2.set_yticks(boxplot_yticks_control)
		ax2.set_yticklabels([])
	#endregion =================================================================================================================

	if neuron_scores_control is not None:
		# high_score_bom_control = np.quantile(np.concatenate([pair[0] for pair in neuron_scores_control.values()]), threshold)
		high_score_hgm_impr_control = np.quantile(np.concatenate([pair[1] - pair[0] for pair in neuron_scores_control.values()]), threshold)
		plt.plot([x_min, x_max + x_step], [high_score_hgm_impr_control, high_score_hgm_impr_control], color='black', linestyle='--')

	#region GCAMP ===========================================================================================================
	boxplot_yticks, boxplot_ytick_labels = list(), list()

	for key in neuron_scores.keys():
		scores_bom, scores_hgm = neuron_scores[key]
		if len(scores_hgm.shape) == 0:
			scores_bom = scores_bom.reshape(1, 1)
			scores_hgm = scores_hgm.reshape(1, 1)

		x = np.median(scores_bom)
		y = np.median(scores_hgm - scores_bom)
		# if neuron_scores_control is not None and control_cutoff and x <= high_score_bom_control:
		# 	continue
		if neuron_scores_control is not None and control_cutoff and y <= high_score_hgm_impr_control:
			continue

		plt.scatter(scores_bom, scores_hgm - scores_bom, s=plot_size * 2, c='skyblue', edgecolor='black', linewidth=1.0, alpha=0.5)

		boxplot_yticks.append(y)
		boxplot_ytick_labels.append(f'{key}')

		if len(scores_hgm) > 1 or force_medians:
			boxplot_v = plt.boxplot(scores_hgm - scores_bom, positions=[x], widths=bp_pxs[0], patch_artist=True,
									showfliers=False)
			boxplot_h = plt.boxplot(scores_bom, positions=[y], widths=bp_pxs[1], patch_artist=True,
									showfliers=False, vert=False)
			for box in [boxplot_v, boxplot_h]:
				for patch in box['boxes']:
					patch.set_facecolor('skyblue')
					patch.set_alpha(0.5)
				for median in box['medians']:
					median.set_color('darkorange')
					median.set_linewidth(1)

	for y, label in zip(boxplot_yticks, boxplot_ytick_labels):
		if y >= y_min:
			plt.text(1.01, y, label, color='blue', ha='left', va='center', rotation=0, fontsize=plot_size * 15/40,
					 transform=plt.gca().get_yaxis_transform())
	ax2 = plt.gca().secondary_yaxis('right')
	ax2.set_yticks(boxplot_yticks)
	ax2.set_yticklabels([])
	#endregion =================================================================================================================

	x_ticks_pos = np.arange(x_min, x_max, x_step)
	plt.xticks(x_ticks_pos, [f'{val:.2f}' for val in x_ticks_pos])
	y_ticks_pos = np.arange(y_min, y_max, y_step)
	plt.yticks(y_ticks_pos, [f'{val:.2f}' for val in y_ticks_pos])

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()

#endregion


def plot_heatmap_stack(traces, behavior=None, label_traces='neuronal activity', label_behavior='behavior', height_ratios=(1, 4),
					   timestamps=None, zlimits_activity=None, zlimits_behavior=None, cmap='rainbow', neuron_ids=False,
					   save_path=None):
	if behavior is not None:
		fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
												gridspec_kw={'height_ratios': height_ratios})
	else:
		fig, (ax_bottom) = plt.subplots(1, 1, figsize=(12, 6 * height_ratios[1] / sum(height_ratios)))

	if behavior is not None:
		if zlimits_behavior is not None:
			heatmap = np.clip(behavior.values.T, *zlimits_behavior)
		else:
			heatmap = behavior.values.T
		im_top = ax_top.imshow(heatmap,
							   aspect='auto', interpolation='nearest',
							   cmap=mplcolors.ListedColormap(plt.cm.get_cmap('Grays', 256)(np.linspace(0.2, 0.8, 256))))
		ax_top.set_ylabel(label_behavior)
		ax_top.set_yticks(np.arange(behavior.values.shape[1]))
		ax_top.set_yticklabels(behavior.columns, fontsize=8)
		ax_top.set_xticks(np.arange(0, behavior.values.shape[0], behavior.values.shape[0] // 10))
		if timestamps is not None:
			ax_top.set_xticklabels([f'{ts:.0f}s' for ts in timestamps[::len(timestamps) // 10]])
		fig.colorbar(im_top, ax=ax_top, orientation='vertical', fraction=0.05)
		ax_top.grid(False)

	if zlimits_activity is not None:
		heatmap = np.clip(traces.values.T, *zlimits_activity)
	else:
		heatmap = traces.values.T
	im_bottom = ax_bottom.imshow(heatmap,
								 aspect='auto', interpolation='nearest',
								 cmap=cmap)
	ax_bottom.set_ylabel(label_traces)
	if not neuron_ids:
		ax_bottom.set_yticks(np.arange(0, traces.values.shape[1], 100))
		ax_bottom.set_yticklabels([traces.columns[i] for i in np.arange(0, traces.values.shape[1], 100)], fontsize=4)
	else:
		ax_bottom.set_yticks(np.arange(traces.values.shape[1]))
		ax_bottom.set_yticklabels([column if not (column == '' or column[0].isdigit()) else '' for column in traces.columns], fontsize=4)
		ax_bottom.set_xticks(np.arange(0, traces.values.shape[0], traces.values.shape[0] // 10))
		if timestamps is not None:
			ax_bottom.set_xticklabels([f'{ts:.0f}s' for ts in timestamps[::len(timestamps) // 10]])
	fig.colorbar(im_bottom, ax=ax_bottom, orientation='vertical', fraction=0.05)
	ax_bottom.grid(False)

	plt.tight_layout()
	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()


def plot_mcos(bars, labels, bar_labels=None, order=('m', 'c', 'o'),
			  normalized=True, log_scale=False,
			  save_path=None):
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

	#region setup
	bars = bars[labels]

	bar_width = 10
	max_y = 1 if normalized else bars.values.sum(axis=0).max()

	index_orders = [
		['m', 'c', 'o'].index(order[0]),
		['m', 'c', 'o'].index(order[1]),
		['m', 'c', 'o'].index(order[2])
	]

	x_positions = list((idx + 0.5) * bar_width for idx in range(len(labels)))
	y_positions = [
		[0 for _ in bars.loc[order[0]]],
		[y1 for y1 in bars.loc[order[0]]],
		[(y1 + y2) for y1, y2 in zip(bars.loc[order[0]], bars.loc[order[1]])]
	]
	#endregion

	# region plot
	plt.figure(figsize=(39, 13))

	color_pool = [cm.get_cmap('rocket')(0.4), cm.get_cmap('rocket')(0.9), 'gray']
	colors = [color_pool[idx] for idx in index_orders]

	for idx in range(3):
		plt.bar(x_positions, bars.loc[order[idx]], width=bar_width, bottom=y_positions[idx], color=colors[idx], edgecolor='none')

	if normalized and bar_labels is not None:
		bar_labels_color_pool = [cm.get_cmap('Greys')(v) for v in [0.1, 0.9, 0.1]]
		bar_labels_colors = [bar_labels_color_pool[idx] for idx in index_orders]

		bar_labels_positions = [
			[0.025 * max_y for _ in bars.loc[order[0]]],
			[y1 + y2 - 0.025 * max_y * y2 for y1, y2 in zip(bars.loc[order[0]], bars.loc[order[1]])],
			[0.975 * max_y for _ in bars.loc[order[2]]]
		]

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
	plt.grid(axis='y', linestyle='--', linewidth=1.0, color=(0.0, 0.0, 0.0))
	#endregion

	for key in bars.T.columns:
		print(f'Average {key}: {np.mean(bars.T[key].values) * 100:.1f}%')

	# plt.rcParams.update(plt.rcParamsDefault)

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()


def plot_markers(markers, markers_reconstructed=None, join_topbot=False,
				 save_path=None):
	fig, axes = plt.subplots(1 if join_topbot else 2, 4, figsize=(16, 6 if join_topbot else 8), squeeze=False)
	if join_topbot:
		max_x, max_y = (np.max(markers[[marker for marker in markers.columns if marker[-1] == 'x']].values),
						np.max(markers[[marker for marker in markers.columns if marker[-1] == 'y']].values))

	if markers_reconstructed is not None:
		for i in range(4):
			name = markers_reconstructed.columns[i * 4].split('-')[0]

			ax = axes[0, i]
			ax.scatter(markers_reconstructed[f'{name}-top_x'].values, markers_reconstructed[f'{name}-top_y'].values, color='red', alpha=0.1, s=5)

			if join_topbot:
				ax.scatter(markers_reconstructed[f'{name}-bot_x'].values, markers_reconstructed[f'{name}-bot_y'].values, color='red', alpha=0.1, s=5)
			else:
				ax = axes[1, i]
				ax.scatter(markers_reconstructed[f'{name}-bot_x'].values, markers_reconstructed[f'{name}-bot_y'].values, color='red', alpha=0.1, s=5)

	for i in range(4):
		name = markers.columns[i * 4].split('-')[0]

		ax = axes[0, i]
		ax.scatter(markers[f'{name}-top_x'].values, markers[f'{name}-top_y'].values, color='blue', alpha=0.1, s=5)

		if join_topbot:
			# ax.set_xlim(0, max_x)
			ax.set_ylim(0, max_y)
			ax.invert_yaxis()
			ax.set_title(name)
			ax.scatter(markers[f'{name}-bot_x'].values, markers[f'{name}-bot_y'].values, color='green', alpha=0.1, s=5)
		else:
			ax.invert_yaxis()
			ax.set_title(name + '_top')
			ax = axes[1, i]
			ax.scatter(markers[f'{name}-bot_x'].values, markers[f'{name}-bot_y'].values, color='green', alpha=0.1, s=5)
			ax.invert_yaxis()
			ax.set_title(name + '_bot')

	if save_path is not None: plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
	plt.show()


def plot_hyperparams(hyperparams, datasets=None, ll='ll00', histograms=False, space='logarithmic', total_mean=True, save_dirpath=None):
	if save_dirpath is not None:
		os.makedirs(save_dirpath, exist_ok=True)

	if datasets is None:
		datasets = sorted(list(set([dataset for model in hyperparams[ll].keys() for dataset in hyperparams[ll][model]['datasets']])))

	for dataset in datasets:
		identifier = f'{ll} {dataset} ({'log' if space == 'logarithmic' else 'lin'} space)'

		fig, axes = plt.subplots(len(hyperparams[ll].keys()), 2,
								 figsize=(2 * 5 + 4, len(hyperparams[ll].keys()) * 3 + 4), squeeze=False)
		fig.suptitle(
			identifier,
			y=0.93, fontsize=25,
		)

		for row_id, model in enumerate(hyperparams[ll].keys()):
			# setup
			row = axes[row_id]

			bbox = row[0].get_position()
			y_center = bbox.y0 + bbox.height / 2
			fig.text(0.02, y_center, model,
					 va='center', ha='center', fontsize=20,
					 )

			for ax in row:
				ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
				for spine in ax.spines.values():
					spine.set_visible(False)

			if dataset not in hyperparams[ll][model]['datasets'].keys(): continue

			wb_plot, wm_plot, vb_plot = None, None, None
			if model.endswith('ss'):
				vb_plot = row[0]
			else:
				wb_plot = row[0]
			if model.startswith('hgm'):
				wm_plot = row[1]

			# plots
			for hyperparam in ['wb_reg_lambda', 'vb_reg_lambda', 'wm_reg_lambda']:
				plot = {
					'wb_reg_lambda': wb_plot,
					'vb_reg_lambda': vb_plot,
					'wm_reg_lambda': wm_plot,
				}[hyperparam]

				if plot is not None:
					plot.tick_params(axis='both', which='both', bottom=True, left=True, labelbottom=True,
									 labelleft=True)
					for spine in plot.spines.values():
						spine.set_visible(True)
					plot.invert_yaxis()

					x_values = hyperparams[ll][model]['datasets'][dataset][hyperparam]['all'][::-1]
					y_values = hyperparams[ll][model]['datasets'][dataset]['mpe_loss']['all'][::-1]
					means = hyperparams[ll][model]['datasets'][dataset][hyperparam]['mean']
					if total_mean:
						total_means = hyperparams[ll][model]['means'][hyperparam]

					plot.set_title(f'{hyperparam} ({x_values.shape[0]} runs)', fontsize=10)

					if space == 'logarithmic':
						x_values = np.log10(x_values)
						means = {mean_type: np.log10(means[mean_type]) for mean_type in means.keys()}
						if total_mean:
							total_means = {mean_type: np.log10(total_means[mean_type]) for mean_type in total_means.keys()}

					if histograms:
						plot.hist(y_values, bins=25)
					else:
						plot.plot(x_values, y_values, marker='o', linestyle='None', alpha=0.5)
						# plot.plot([x_values[0], x_values[-1]], [means['linear'], means['linear']], label='local mean', color='C2', alpha=0.33)
						plot.plot([means['geometric']] * 2, [y_values[0], y_values[-1]], label='local mean', color='C2')
						if total_mean:
							# plot.plot([x_values[0], x_values[-1]], [total_means['linear'], total_means['linear']], label='global mean', color='C1', alpha=0.33)
							plot.plot([total_means['geometric']] * 2, [y_values[0], y_values[-1]], label='global mean', color='C1')

						plot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
						plot.legend()

		if save_dirpath is not None:
			plt.savefig(os.path.join(save_dirpath, f'{identifier}.png'), dpi=300, bbox_inches='tight')
