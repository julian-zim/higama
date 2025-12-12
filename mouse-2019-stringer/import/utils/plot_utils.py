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
Plots traces cycling color according to pyplots policy, with the first one black
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

