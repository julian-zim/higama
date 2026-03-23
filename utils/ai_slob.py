from typing import Dict, Tuple, Optional, Iterable, List
import os
import numpy as np
import matplotlib.pyplot as plt
import math


def compute_neuron_percentiles_f(neuron_scores, neuron_scores_shuffle):
	percentiles = dict()
	for label in neuron_scores.keys():
		score = neuron_scores[label]
		control_scores = neuron_scores_shuffle[label]
		better_bom = score[0] > control_scores[0]
		better_hgm = score[1] > control_scores[1]
		# better_bom_ratio = sum(better_bom) / len(control_scores[0])
		better_bom_ratio = sum(better_bom)
		# better_hgm_ratio = sum(better_hgm) / len(control_scores[1])
		better_hgm_ratio = sum(better_hgm)
		percentiles[label] = (better_bom_ratio, better_hgm_ratio)

	return percentiles


def compute_neuron_percentiles(
	neuron_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
	neuron_scores_shuffle: Dict[str, Tuple[np.ndarray, np.ndarray]],
	model: str = 'hgm',
) -> Dict[str, float]:
	"""Compute mid-rank percentiles for each neuron.

	For each neuron label:
	  - `neuron_scores[label][idx]` must be a single scalar (or length-1
		array) containing the observed score for the chosen model.
	  - `neuron_scores_shuffle[label][idx]` must be a 1-D array of control
		samples (e.g. length 100). If not 1-D it will be flattened.

	The percentile is computed as:
	  pct = (count(ctrl < obs) + 0.5 * count(ctrl == obs)) / N * 100

	Returns a dict mapping neuron label -> percentile (float in [0,100]) or
	np.nan for neurons without controls or empty control arrays.
	"""

	def _model_index(m: str) -> int:
		mm = m.lower()
		if mm == 'hgm':
			return 1
		if mm == 'bom':
			return 0
		raise ValueError("model must be 'hgm' or 'bom'")

	idx = _model_index(model)
	percentiles: Dict[str, float] = {}

	for label in neuron_scores.keys():
		# get observed scalar
		try:
			observed_arr = neuron_scores[label][idx]
		except Exception:
			# mark missing/invalid observed as NaN
			percentiles[label] = float('nan')
			continue

		obs_a = np.asarray(observed_arr)
		if obs_a.ndim == 0:
			obs = float(obs_a)
		elif obs_a.size == 1:
			obs = float(obs_a.ravel()[0])
		else:
			# non-scalar observed -> mark NaN
			percentiles[label] = float('nan')
			continue

		# get control array for same neuron
		if label not in neuron_scores_shuffle:
			percentiles[label] = float('nan')
			continue

		ctrl_arr = neuron_scores_shuffle[label][idx]
		if ctrl_arr is None:
			percentiles[label] = float('nan')
			continue

		ctrl = np.asarray(ctrl_arr)
		if ctrl.size == 0:
			percentiles[label] = float('nan')
			continue

		if ctrl.ndim != 1:
			# flatten any higher-dim control array to 1-D
			ctrl = ctrl.ravel()

		less = np.sum(ctrl < obs)
		equal = np.sum(ctrl == obs)
		n = float(ctrl.size)
		pct = (less + 0.5 * equal) / n * 100.0
		percentiles[label] = float(pct)

	return percentiles



def plot_control_histograms(
	neuron_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
	neuron_scores_shuffle: Dict[str, Tuple[np.ndarray, np.ndarray]],
	neuron_labels: Optional[Iterable[str]] = None,
	model: str = 'hgm',
	bins: int = 40,
	ncols: int = 3,
	figsize_per_plot: Tuple[int, int] = (6, 3),
	save_path: Optional[str] = None,
) -> List[str]:
	"""Plot histogram of control distribution for each requested neuron.

	If `save_path` is provided, this function will save each neuron's histogram
	as a separate PNG file in that directory under the `neurons/` subfolder
	named {label}.png (dpi=300). It will always display a combined
	multi-panel figure and also save that combined figure as `all.png` inside
	`save_path` when provided.

	Returns the list of labels that were plotted.
	"""

	def _model_index(m: str) -> int:
		mm = m.lower()
		if mm == 'hgm':
			return 1
		if mm == 'bom':
			return 0
		raise ValueError("model must be 'hgm' or 'bom'")

	idx = _model_index(model)

	# choose labels
	if neuron_labels is None:
		labels = list(neuron_scores.keys())
	else:
		labels = list(neuron_labels)
		if len(labels) == 0:
			labels = list(neuron_scores.keys())

	if len(labels) == 0:
		return []

	# prepare save dirs if needed
	neurons_dir = None
	if save_path is not None:
		os.makedirs(save_path, exist_ok=True)
		neurons_dir = os.path.join(save_path, 'individual')
		os.makedirs(neurons_dir, exist_ok=True)

	# create individual plots if saving requested
	plotted: List[str] = []
	for label in labels:
		# observed
		try:
			obs_arr = neuron_scores[label][idx]
		except Exception:
			continue
		obs_a = np.asarray(obs_arr)
		if obs_a.ndim == 0:
			obs = float(obs_a)
		elif obs_a.size == 1:
			obs = float(obs_a.ravel()[0])
		else:
			continue

		# controls
		if label not in neuron_scores_shuffle:
			continue
		ctrl_arr = neuron_scores_shuffle[label][idx]
		if ctrl_arr is None:
			continue
		ctrl = np.asarray(ctrl_arr)
		if ctrl.size == 0:
			continue
		if ctrl.ndim != 1:
			ctrl = ctrl.ravel()

		# always create figure for display, and save individual if requested
		fig, ax = plt.subplots(figsize=figsize_per_plot)
		try:
			plt.style.use('seaborn-whitegrid')
		except Exception:
			pass


		# Control histogram: tomato
		counts, edges, patches = ax.hist(ctrl, bins=bins, color='tomato', alpha=0.85, edgecolor='black', linewidth=0.4)
		for p in patches:
			p.set_linewidth(0.4)

		median = float(np.nanmedian(ctrl))
		mean = float(np.nanmean(ctrl))
		# Observation line: blue
		l_obs = ax.axvline(obs, color='blue', linewidth=2.0, label='observed')
		# Median/mean lines for control: red
		l_med = ax.axvline(median, color='red', linestyle='--', linewidth=1.4, label='median')
		l_mean = ax.axvline(mean, color='red', linestyle=':', linewidth=1.4, label='mean')

		less = np.sum(ctrl < obs)
		equal = np.sum(ctrl == obs)
		n_ctrl = float(ctrl.size)
		pct = (less + 0.5 * equal) / n_ctrl * 100.0

		ax_text = f"pct={pct:.1f}\nmedian={median:.3g}\nmean={mean:.3g}"
		# place the small info box near top-right of axes
		ax.text(0.98, 0.95, ax_text, transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

		ax.set_title(f"{label}")
		ax.set_ylabel('Count')
		ax.set_xlabel('Control value')
		ax.legend(handles=[l_obs, l_med, l_mean], fontsize=8)

		if neurons_dir is not None:
			save_fname = os.path.join(neurons_dir, f"r2-{np.median(neuron_scores[label][1]).item():.2f}_r2i-{np.median(neuron_scores[label][1]).item()-np.median(neuron_scores[label][0]).item():.2f}_n-{label}.png")
			fig.tight_layout()
			fig.savefig(save_fname, dpi=300)

		plt.close(fig)
		plotted.append(label)

	# combined multi-panel figure (always displayed)
	n = len(labels)
	ncols = max(1, min(ncols, n))
	nrows = math.ceil(n / ncols)
	fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

	try:
		plt.style.use('seaborn-whitegrid')
	except Exception:
		pass

	displayed: List[str] = []
	for i, label in enumerate(labels):
		ax = fig.add_subplot(nrows, ncols, i + 1)

		# observed
		try:
			obs_arr = neuron_scores[label][idx]
		except Exception:
			ax.text(0.5, 0.5, f"Missing observed for {label}", ha='center', va='center')
			ax.set_title(label)
			continue

		obs_a = np.asarray(obs_arr)
		if obs_a.ndim == 0:
			obs = float(obs_a)
		elif obs_a.size == 1:
			obs = float(obs_a.ravel()[0])
		else:
			ax.text(0.5, 0.5, f"Observed for {label} not scalar", ha='center', va='center')
			ax.set_title(label)
			continue

		# controls
		if label not in neuron_scores_shuffle:
			ax.text(0.5, 0.5, f"No control for {label}", ha='center', va='center')
			ax.set_title(label)
			continue

		ctrl_arr = neuron_scores_shuffle[label][idx]
		if ctrl_arr is None:
			ax.text(0.5, 0.5, f"No control for {label}", ha='center', va='center')
			ax.set_title(label)
			continue

		ctrl = np.asarray(ctrl_arr)
		if ctrl.size == 0:
			ax.text(0.5, 0.5, f"Empty control for {label}", ha='center', va='center')
			ax.set_title(label)
			continue

		if ctrl.ndim != 1:
			ctrl = ctrl.ravel()


		# Control histogram: tomato
		counts, edges, patches = ax.hist(ctrl, bins=bins, color='tomato', alpha=0.85, edgecolor='black', linewidth=0.4)
		for p in patches:
			p.set_linewidth(0.4)

		median = float(np.nanmedian(ctrl))
		mean = float(np.nanmean(ctrl))

		# Observation line: blue
		l_obs = ax.axvline(obs, color='blue', linewidth=2.0, label='observed')
		# Median/mean lines for control: red
		l_med = ax.axvline(median, color='red', linestyle='--', linewidth=1.4, label='median')
		l_mean = ax.axvline(mean, color='red', linestyle=':', linewidth=1.4, label='mean')

		less = np.sum(ctrl < obs)
		equal = np.sum(ctrl == obs)
		n_ctrl = float(ctrl.size)
		pct = (less + 0.5 * equal) / n_ctrl * 100.0

		ax_text = f"pct={pct:.1f}\nmedian={median:.3g}\nmean={mean:.3g}"
		ax.text(0.98, 0.95, ax_text, transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

		ax.set_title(f"{label}")
		ax.set_ylabel('Count')
		ax.set_xlabel('Control value')
		ax.legend(handles=[l_obs, l_med, l_mean], fontsize=8)
		displayed.append(label)

	# hide any empty subplots
	for j in range(n, nrows * ncols):
		fig.add_subplot(nrows, ncols, j + 1).axis('off')

	plt.tight_layout()
	# if save_path provided, save combined figure as all.png
	if save_path is not None:
		try:
			combined_path = os.path.join(save_path, 'all.png')
			fig.tight_layout()
			fig.savefig(combined_path, dpi=300)
		except Exception:
			pass

	plt.show()
	return displayed



def plot_percentile_hist(percentiles: Dict[str, float], bins: int = 50, title: Optional[str] = None, threshold: Optional[float] = None, figsize: Tuple[int, int] = (6, 4), save_path: Optional[str] = None):
	"""Plot histogram of percentile values across neurons.

	Args:
		percentiles: mapping label -> percentile (0-100). NaNs are ignored.
		bins: number of histogram bins.
		title: optional title for the plot.
		threshold: optional vertical line at this percentile value.
		figsize: figure size in inches.

	Returns:
		(fig, ax) tuple with the matplotlib Figure and Axes.
	"""
	# collect valid percentile values
	vals = np.array([v for v in percentiles.values() if v is not None and not (isinstance(v, float) and np.isnan(v))])

	fig, ax = plt.subplots(figsize=figsize)
	if vals.size == 0:
		ax.text(0.5, 0.5, 'No valid percentile values to plot', ha='center', va='center')
		if title:
			ax.set_title(title)
		return fig, ax


	# Histogram: skyblue for control/percentiles
	counts, edges, patches = ax.hist(vals, bins=bins, color='skyblue', edgecolor='black', linewidth=0.5, alpha=0.9)

	mean = float(np.nanmean(vals))
	median = float(np.nanmedian(vals))

	# Mean/median lines: blue
	l_mean = ax.axvline(mean, color='blue', linestyle=':', linewidth=1.8, label=f'mean={mean:.1f}')
	l_median = ax.axvline(median, color='blue', linestyle='--', linewidth=1.6, label=f'median={median:.1f}')

	if threshold is not None:
		ax.axvline(threshold, color='blue', linestyle='-', linewidth=1.6, label=f'threshold={threshold}')
		n_above = int(np.sum(vals >= threshold))
		pct_above = n_above / vals.size * 100.0
		thr_text = f'{n_above}/{vals.size} ({pct_above:.1f}%) ≥ {threshold:.0f}% thr'

	ax.set_xlabel('Percentile')
	ax.set_ylabel('Count')
	if title:
		ax.set_title(title)

	legend = ax.legend(fontsize=9)
	# position threshold text left and below the legend, slightly to the right
	if threshold is not None:
		try:
			fig.canvas.draw()
			# get_window_extent may require a renderer; use get_tightbbox fallback
			renderer = fig.canvas.get_renderer()
			bbox = legend.get_window_extent(renderer=renderer)
			bbox_axes = ax.transAxes.inverted().transform([[bbox.x0, bbox.y0], [bbox.x1, bbox.y1]])
			x0_axes, y0_axes = bbox_axes[0]
			# place text slightly to the right of legend's left edge and slightly below its bottom
			text_x = min(0.95, x0_axes + 0.02)
			text_y = max(0.01, y0_axes - 0.03)
		except Exception:
			text_x, text_y = 0.04, 0.88

		ax.text(text_x, text_y, thr_text, transform=ax.transAxes,
				ha='left', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

	# save if requested
	if save_path is not None:
		d = os.path.dirname(save_path)
		if d:
			os.makedirs(d, exist_ok=True)
		fig.tight_layout()
		fig.savefig(f"{save_path}.png", dpi=300)

	plt.tight_layout()
	plt.show()
	return fig, ax



def plot_performance_histogram(
	neuron_scores: Dict[str, Tuple[np.ndarray, np.ndarray]],
	neuron_scores_shuffle: Dict[str, Tuple[np.ndarray, np.ndarray]],
	model: str = 'hgm',
	bins: int = 50,
	figsize: Tuple[int, int] = (7, 4),
	title: Optional[str] = None,
	save_path: Optional[str] = None,
):
	"""Plot a histogram of per-neuron median performance values (second tuple index by default).

	For each neuron in `neuron_scores` this function will:
	  - extract the observed value at the chosen model index (must be scalar)
	  - extract the control samples at the same index (if available) and compute the median

	The primary histogram shows the distribution of control medians across neurons. The
	observed values are overlaid (as a semi-transparent histogram) and the mean/median for
	the observed distribution are drawn with vertical lines (same color, different linestyles).

	Returns (fig, ax).
	"""

	def _model_index(m: str) -> int:
		mm = m.lower()
		if mm == 'hgm':
			return 1
		if mm == 'bom':
			return 0
		raise ValueError("model must be 'hgm' or 'bom'")

	idx = _model_index(model)

	labels = list(neuron_scores.keys())
	if len(labels) == 0:
		raise ValueError('neuron_scores is empty')

	# Collect per-neuron medians for observed tuples and control tuples.
	obs_vals = []
	ctrl_medians = []
	for label in labels:
		# observed tuple -> convert each element to a scalar by taking median if needed
		try:
			obs_tuple = neuron_scores[label]
		except Exception:
			# missing observed entry
			continue
		if not isinstance(obs_tuple, (list, tuple)):
			# unexpected format; try to coerce
			obs_tuple = (obs_tuple,)

		# try to extract the model index element and reduce it to a scalar median
		try:
			obs_elem = obs_tuple[idx]
		except Exception:
			# tuple too short
			continue

		obs_arr = np.asarray(obs_elem)
		if obs_arr.size == 0:
			# skip empty
			continue
		# if array-like, take median; if scalar, take as-is
		obs_scalar = float(np.nanmedian(obs_arr)) if obs_arr.size > 1 else float(obs_arr.ravel()[0])
		obs_vals.append(obs_scalar)

		# control tuple -> similarly reduce to scalar median for the same index
		ctrl_scalar = None
		if label in neuron_scores_shuffle:
			ctrl_tuple = neuron_scores_shuffle[label]
			if isinstance(ctrl_tuple, (list, tuple)) and len(ctrl_tuple) > idx:
				ctrl_elem = ctrl_tuple[idx]
				if ctrl_elem is not None:
					ctrl_arr = np.asarray(ctrl_elem)
					if ctrl_arr.size > 0:
						ctrl_scalar = float(np.nanmedian(ctrl_arr))

		if ctrl_scalar is not None:
			ctrl_medians.append(ctrl_scalar)

	ctrl_medians = np.asarray(ctrl_medians)
	obs_vals = np.asarray(obs_vals)

	fig, ax = plt.subplots(figsize=figsize)
	try:
		plt.style.use('seaborn-whitegrid')
	except Exception:
		pass

	if ctrl_medians.size == 0 and obs_vals.size == 0:
		ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center')
		if title:
			ax.set_title(title)
		return fig, ax

	# Use common bin edges and normalize histograms to fractions so that
	# differing sample sizes (e.g. few controls vs many neurons) are comparable.
	combined = None
	if ctrl_medians.size > 0 and obs_vals.size > 0:
		combined = np.concatenate([ctrl_medians, obs_vals])
	elif ctrl_medians.size > 0:
		combined = ctrl_medians
	elif obs_vals.size > 0:
		combined = obs_vals

	bin_edges = None
	if combined is not None:
		try:
			bin_edges = np.histogram_bin_edges(combined, bins=bins)
		except Exception:
			bin_edges = bins

	# plot control medians normalized to fraction of controls
	if ctrl_medians.size > 0:
		if bin_edges is not None:
			bins_use = bin_edges
		else:
			bins_use = bins
		weights_ctrl = np.ones_like(ctrl_medians, dtype=float) / float(ctrl_medians.size)

		counts_ctrl, edges_ctrl, patches_ctrl = ax.hist(
			ctrl_medians,
			bins=bins_use,
			weights=weights_ctrl,
			color='tomato',
			alpha=0.85,
			edgecolor='black',
			linewidth=0.4,
			label='control medians',
		)
		for p in patches_ctrl:
			p.set_linewidth(0.4)

	# plot observed medians normalized to fraction of neurons
	if obs_vals.size > 0:
		if bin_edges is not None:
			bins_use = bin_edges
		else:
			bins_use = bins
		weights_obs = np.ones_like(obs_vals, dtype=float) / float(obs_vals.size)

		counts_obs, edges_obs, patches_obs = ax.hist(
			obs_vals,
			bins=bins_use,
			weights=weights_obs,
			color='skyblue',
			alpha=0.6,
			edgecolor='black',
			linewidth=0.4,
			label='observed medians',
		)
		for p in patches_obs:
			p.set_linewidth(0.4)

	# draw mean/median lines for observed (same color, different linestyles)
	if obs_vals.size > 0:
		obs_mean = float(np.nanmean(obs_vals))
		obs_med = float(np.nanmedian(obs_vals))

		# Mean/median lines for observed: blue
		ax.axvline(obs_mean, color='blue', linestyle=':', linewidth=1.8, label=f'obs mean={obs_mean:.3g}')
		ax.axvline(obs_med, color='blue', linestyle='--', linewidth=1.6, label=f'obs median={obs_med:.3g}')

	# draw mean/median for control medians
	if ctrl_medians.size > 0:
		ctrl_mean = float(np.nanmean(ctrl_medians))
		ctrl_med = float(np.nanmedian(ctrl_medians))

		# Mean/median lines for control: red
		ax.axvline(ctrl_mean, color='red', linestyle=':', linewidth=1.6, label=f'ctrl mean={ctrl_mean:.3g}')
		ax.axvline(ctrl_med, color='red', linestyle='--', linewidth=1.4, label=f'ctrl median={ctrl_med:.3g}')

	ax.set_xlabel('Value (model index {})'.format(idx))
	ax.set_ylabel('Count')
	if title:
		ax.set_title(title)
	else:
		ax.set_title('Performance histogram (median of controls)')

	ax.legend(fontsize=9)

	# save if requested
	if save_path is not None:
		d = os.path.dirname(save_path)
		if d:
			os.makedirs(d, exist_ok=True)
		fig.tight_layout()
		fig.savefig(f"{save_path}.png", dpi=300)

	plt.tight_layout()
	plt.show()
	return fig, ax



