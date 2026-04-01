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

	def _model_index_or_diff(m: str):
		mm = m.lower()
		if mm in ('hgm',):
			return 1
		if mm in ('bom',):
			return 0
		if mm in ('improvement', 'diff', 'difference'):
			return 'diff'
		raise ValueError("model must be 'hgm', 'bom', or 'improvement' (or 'diff', 'difference')")

	mode = _model_index_or_diff(model)
	percentiles: Dict[str, float] = {}

	for label in neuron_scores.keys():
		try:
			if mode == 'diff':
				observed_arr = np.asarray(neuron_scores[label][1]) - np.asarray(neuron_scores[label][0])
			else:
				observed_arr = np.asarray(neuron_scores[label][mode]) if isinstance(mode, int) else None
		except Exception:
			percentiles[label] = float('nan')
			continue

		if observed_arr is None:
			percentiles[label] = float('nan')
			continue

		obs_a = np.asarray(observed_arr)
		if obs_a.ndim == 0:
			obs = float(obs_a)
		elif obs_a.size == 1:
			obs = float(obs_a.ravel()[0])
		else:
			percentiles[label] = float('nan')
			continue

		if label not in neuron_scores_shuffle:
			percentiles[label] = float('nan')
			continue

		try:
			if mode == 'diff':
				ctrl_arr = np.asarray(neuron_scores_shuffle[label][1]) - np.asarray(neuron_scores_shuffle[label][0])
			else:
				ctrl_arr = np.asarray(neuron_scores_shuffle[label][mode]) if isinstance(mode, int) else None
		except Exception:
			percentiles[label] = float('nan')
			continue

		if ctrl_arr is None:
			percentiles[label] = float('nan')
			continue

		ctrl = np.asarray(ctrl_arr)
		if ctrl.size == 0:
			percentiles[label] = float('nan')
			continue

		if ctrl.ndim != 1:
			ctrl = ctrl.ravel()

		less = np.sum(ctrl < obs)
		equal = np.sum(ctrl == obs)
		n = float(ctrl.size)
		pct = (less + 0.5 * equal) / n * 100.0
		percentiles[label] = float(pct)

	return percentiles


def percentile_to_p_value_one_sided(
	percentiles: Dict[str, float]
) -> Dict[str, float]:
	"""
	Convert percentiles to right-sided (greater than or equal) one-sided p-values.

	For each neuron label:
	  - p-value = 1 - (percentile / 100)
	  - If percentile is NaN or None, p-value is set to np.nan

	Args:
		percentiles: dict mapping neuron label to percentile (float in [0, 100])
	Returns:
		dict mapping neuron label to one-sided p-value (float in [0, 1])
	"""
	pvals: Dict[str, float] = {}
	for label, pct in percentiles.items():
		if pct is None or (isinstance(pct, float) and np.isnan(pct)):
			pvals[label] = np.nan
		else:
			pvals[label] = 1.0 - (pct / 100.0)
	return pvals


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

	def _model_index_or_diff(m: str):
		mm = m.lower()
		if mm == 'hgm':
			return 1
		if mm == 'bom':
			return 0
		if mm in ('improvement', 'diff', 'difference'):
			return 'diff'
		raise ValueError("model must be 'hgm', 'bom', or 'improvement'/'diff'/'difference'")

	mode = _model_index_or_diff(model)

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
			if mode == 'diff':
				obs_arr = np.asarray(neuron_scores[label][1]) - np.asarray(neuron_scores[label][0])
			elif isinstance(mode, int):
				obs_arr = neuron_scores[label][mode]
			else:
				obs_arr = None
		except Exception:
			continue
		if obs_arr is None:
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
		try:
			if mode == 'diff':
				ctrl_arr = np.asarray(neuron_scores_shuffle[label][1]) - np.asarray(neuron_scores_shuffle[label][0])
			elif isinstance(mode, int):
				ctrl_arr = neuron_scores_shuffle[label][mode]
			else:
				ctrl_arr = None
		except Exception:
			continue
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
		ax.text(0.98, 0.95, ax_text, transform=ax.transAxes, ha='right', va='top', fontsize=9,
				bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

		if mode == 'diff':
			ax.set_title(f"{label} (improvement)")
			ax.set_xlabel('Improvement (HGM - BOM)')
		else:
			ax.set_title(f"{label}")
			ax.set_xlabel('Control value')
		ax.set_ylabel('Count')
		ax.legend(handles=[l_obs, l_med, l_mean], fontsize=8)

		if neurons_dir is not None:
			if mode == 'diff':
				save_fname = os.path.join(neurons_dir,
					f"impr-{np.median(obs):.2f}_n-{label}.png")
			else:
				save_fname = os.path.join(neurons_dir,
					f"r2-{np.median(neuron_scores[label][1]).item():.2f}_r2i-{np.median(neuron_scores[label][1]).item() - np.median(neuron_scores[label][0]).item():.2f}_n-{label}.png")
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
			if mode == 'diff':
				obs_arr = np.asarray(neuron_scores[label][1]) - np.asarray(neuron_scores[label][0])
			elif isinstance(mode, int):
				obs_arr = neuron_scores[label][mode]
			else:
				obs_arr = None
		except Exception:
			ax.text(0.5, 0.5, f"Missing observed for {label}", ha='center', va='center')
			ax.set_title(label)
			continue
		if obs_arr is None:
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
		try:
			if mode == 'diff':
				ctrl_arr = np.asarray(neuron_scores_shuffle[label][1]) - np.asarray(neuron_scores_shuffle[label][0])
			elif isinstance(mode, int):
				ctrl_arr = neuron_scores_shuffle[label][mode]
			else:
				ctrl_arr = None
		except Exception:
			ax.text(0.5, 0.5, f"No control for {label}", ha='center', va='center')
			ax.set_title(label)
			continue
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
		ax.text(0.98, 0.95, ax_text, transform=ax.transAxes, ha='right', va='top', fontsize=9,
				bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

		if mode == 'diff':
			ax.set_title(f"{label} (improvement)")
			ax.set_xlabel('Improvement (HGM - BOM)')
		else:
			ax.set_title(f"{label}")
			ax.set_xlabel('Control value')
		ax.set_ylabel('Count')
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


def plot_percentile_hist(percentiles: Dict[str, float], bins: int = 50, title: Optional[str] = None,
						 threshold: Optional[float] = None, figsize: Tuple[int, int] = (6, 4),
						 save_path: Optional[str] = None):
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
	offset = - (edges[1] - edges[0])

	l_mean = ax.axvline(mean + offset, color='blue', linestyle=':', linewidth=1.8, label=f'mean={mean:.1f}')
	l_median = ax.axvline(median + offset, color='blue', linestyle='--', linewidth=1.6, label=f'median={median:.1f}')
	l_threshold = None
	if threshold is not None:
		l_threshold = ax.axvline(threshold + offset, color='blue', linestyle='-', linewidth=1.6, label=f'threshold={threshold}')
		n_above = int(np.sum(vals >= threshold))
		pct_above = n_above / vals.size * 100.0
		thr_text = f'{n_above}/{vals.size} ({pct_above:.1f}%) ≥ {threshold:.0f}% thr'

	ax.set_xlabel('Percentile')
	ax.set_ylabel('Count')
	if title:
		ax.set_title(title)

	handles = [l_mean, l_median]
	if l_threshold is not None:
		handles.append(l_threshold)
	legend = ax.legend(handles=handles, fontsize=9)

	# position threshold text left and below the legend, slightly to the right
	if threshold is not None:
		n_above = int(np.sum(vals >= threshold))
		pct_above = n_above / vals.size * 100.0
		thr_text = f'{n_above}/{vals.size} ({pct_above:.1f}%) ≥ {threshold:.0f}% thr'
		# Avoid using get_renderer, fallback to default position
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
		threshold: Optional[float] = None,
		id_only: bool = True,
):
	"""Plot a histogram of per-neuron median performance values (second tuple index by default).

	For each neuron in `neuron_scores` this function will:
	  - extract the observed value at the chosen model index (must be scalar)
	  - extract the control samples at the same index (if available) and compute the median

	The primary histogram shows the distribution of control medians across neurons. The
	observed values are overlaid (as a semi-transparent histogram) and the mean/median for
	the observed distribution are drawn with vertical lines (same color, different linestyles).

	If 'threshold' is set, a vertical black line is drawn at the threshold value, and all neurons
	whose observed value passes the threshold are labeled on the plot.

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

	# Only keep neurons with string IDs (first char not digit) for all calculations and plotting
	labels = [k for k in neuron_scores.keys() if not k[0].isdigit()]
	if len(labels) == 0:
		raise ValueError('No string-ID neurons in neuron_scores')

	# Filter neuron_scores and neuron_scores_shuffle to only string-ID keys
	neuron_scores_id = {k: v for k, v in neuron_scores.items() if k in labels}
	neuron_scores_shuffle_id = {k: v for k, v in neuron_scores_shuffle.items() if k in labels}

	calc_labels = labels
	plot_labels = labels

	# Collect per-neuron medians for observed tuples and control tuples.
	obs_vals = []
	ctrl_medians = []
	obs_label_pairs = []  # (label, obs_scalar)
	for label in calc_labels:
		# observed tuple -> convert each element to a scalar by taking median if needed
		try:
			obs_tuple = neuron_scores_id[label]
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
		obs_label_pairs.append((label, obs_scalar))

		# control tuple -> similarly reduce to scalar median for the same index
		ctrl_scalar = None
		if label in neuron_scores_shuffle_id:
			ctrl_tuple = neuron_scores_shuffle_id[label]
			if isinstance(ctrl_tuple, (list, tuple)) and len(ctrl_tuple) > idx:
				ctrl_elem = ctrl_tuple[idx]
				if ctrl_elem is not None:
					ctrl_arr = np.asarray(ctrl_elem)
					if ctrl_arr.size > 0:
						ctrl_scalar = float(np.nanmedian(ctrl_arr))

		if ctrl_scalar is not None:
			ctrl_medians.append(ctrl_scalar)

	# No need to filter for plotting anymore, as all keys are string IDs
	obs_label_pairs_plot = obs_label_pairs
	obs_vals_plot = np.array([val for (label, val) in obs_label_pairs_plot])

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
	all_ctrl_values = []
	all_ctrl_values_flat = np.array([])  # Ensure always defined
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
			color='lightcoral',
			alpha=0.6,
			edgecolor='black',
			linewidth=0.4,
			label='control',
		)
		for p in patches_ctrl:
			p.set_linewidth(0.4)

		# Collect all control values for all neurons at the selected model index
		for label in labels:
			if label in neuron_scores_shuffle:
				ctrl_tuple = neuron_scores_shuffle[label]
				if isinstance(ctrl_tuple, (list, tuple)) and len(ctrl_tuple) > idx:
					ctrl_elem = ctrl_tuple[idx]
					if ctrl_elem is not None:
						ctrl_arr = np.asarray(ctrl_elem)
						if ctrl_arr.size > 0:
							all_ctrl_values.append(ctrl_arr.ravel())
		if all_ctrl_values:
			all_ctrl_values_flat = np.concatenate(all_ctrl_values)
		else:
			all_ctrl_values_flat = np.array([])
	# plot observed medians normalized to fraction of neurons (only id_keys if id_only)
	if obs_vals_plot.size > 0:
		if bin_edges is not None:
			bins_use = bin_edges
		else:
			bins_use = bins
		weights_obs = np.ones_like(obs_vals_plot, dtype=float) / float(obs_vals_plot.size)

		counts_obs, edges_obs, patches_obs = ax.hist(
			obs_vals_plot,
			bins=bins_use,
			weights=weights_obs,
			color='skyblue',
			alpha=0.6,
			edgecolor='black',
			linewidth=0.4,
			label='observed',
		)
		for p in patches_obs:
			p.set_linewidth(0.4)

		# If threshold is set, interpret as percentile and plot vertical line at that value
		if threshold is not None and ctrl_medians.size > 0:
			# Compute threshold value using ALL neuron scores (not just string-ID)
			all_ctrl_vals_for_thresh = []
			for ctrl_tuple in neuron_scores_shuffle.values():
				if isinstance(ctrl_tuple, (list, tuple)) and len(ctrl_tuple) > idx:
					ctrl_elem = ctrl_tuple[idx]
					if ctrl_elem is not None:
						ctrl_arr = np.asarray(ctrl_elem)
						if ctrl_arr.size > 0:
							all_ctrl_vals_for_thresh.append(ctrl_arr.ravel())
			if all_ctrl_vals_for_thresh:
				all_ctrl_vals_for_thresh = np.concatenate(all_ctrl_vals_for_thresh)
				perc_val = np.quantile(all_ctrl_vals_for_thresh, threshold)
			else:
				perc_val = np.nan
			# Calculate bin width (assume uniform bins)
			if bin_edges is not None and len(bin_edges) > 1:
				bin_width = bin_edges[1] - bin_edges[0]
			else:
				bin_width = 0
			line_position = perc_val #+ bin_width
			ax.axvline(line_position, color='black', linestyle='-', linewidth=1.5,
					   label=f'Threshold ({threshold * 100:.1f}%) = {perc_val:.2f}')
			# Label logic: for each observed bin with left edge >= perc_val, plot neuron labels below the x-axis
			if bin_edges is not None and len(bin_edges) > 1:
				obs_vals_arr = np.array([val for _, val in obs_label_pairs_plot])
				bin_idxs = np.digitize(obs_vals_arr, bin_edges) - 1
				# Fix: assign values equal to the rightmost edge to the last bin
				last_bin = len(bin_edges) - 2
				bin_idxs[obs_vals_arr == bin_edges[-1]] = last_bin
				for i in range(len(bin_edges) - 1):
					left_edge = bin_edges[i]
					right_edge = bin_edges[i+1]
					if left_edge >= perc_val:
						in_bin = [j for j, idx in enumerate(bin_idxs) if idx == i]
						if not in_bin:
							continue
						labels_in_bin = [obs_label_pairs_plot[j][0] for j in in_bin]
						label_text = ',\n'.join(labels_in_bin)
						x_pos = (left_edge + right_edge) / 2
						# Lower the label and annotation line further below the x-axis
						y_offset = -0.18  # Lower than previous -0.12
						ax.text(x_pos, y_offset, label_text, ha='right', va='top', fontsize=8, color='black', rotation=45, transform=ax.get_xaxis_transform())
						ax.annotate(
							'',
							xy=(x_pos, 0), xycoords='data',
							xytext=(x_pos, y_offset), textcoords=('data', 'axes fraction'),
							arrowprops=dict(arrowstyle='-', color='gray', linewidth=0.8),
							annotation_clip=False
						)

	# Draw mean/median lines for observed (blue) and control (red) using only letter-ID neurons
	if obs_vals.size > 0:
		obs_mean = float(np.nanmean(obs_vals))
		obs_med = float(np.nanmedian(obs_vals))
		ax.axvline(obs_mean, color='blue', linestyle=':', linewidth=1.8, label=f'obs mean={obs_mean:.3g}')
		ax.axvline(obs_med, color='blue', linestyle='--', linewidth=1.6, label=f'obs median={obs_med:.3g}')

	if ctrl_medians.size > 0:
		ctrl_mean = float(np.nanmean(ctrl_medians))
		ctrl_med = float(np.nanmedian(ctrl_medians))
		ax.axvline(ctrl_mean, color='red', linestyle=':', linewidth=1.6, label=f'ctrl mean={ctrl_mean:.3g}')
		ax.axvline(ctrl_med, color='red', linestyle='--', linewidth=1.4, label=f'ctrl median={ctrl_med:.3g}')

	ax.set_xlabel(f'{model} r2 score')
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



def volcano_plot_proper(neuron_scores, neuron_p_values,
						mode='improvement', n_label=None):
	x_vals = None
	match mode:
		case 'improvement':
			x_vals = np.array([(neuron_scores[key][1] - neuron_scores[key][0]).squeeze() for key in neuron_scores.keys()])
			print(x_vals < 0)

	y_vals = -np.log10(np.array([neuron_p_values[key] for key in neuron_scores.keys()]))

	plt.scatter(x_vals, y_vals,
				s=5)


def volcano_plot(
		neuron_scores: dict,
		p_values: dict,
		n_label: int | None = 0,
		model: str = 'hgm',
		p_thresh: float = 0.05,
		title: str = 'Volcano Plot',
		figsize: tuple = (8, 6),
		s: int = 18,
		alpha: float = 0.7,
		show: bool = True,
		save_path: str = None,
		xlim=None,
		ylim=None,
):
	"""
	Create a volcano plot: x = neuron score, y = -log10(p-value).
	Blue: x < 0 & p < p_thresh; Red: x > 0 & p < p_thresh; Gray: p >= p_thresh.
	Only the n_label neurons with the best (lowest) p-value and highest x score are labeled.
	Args:
		neuron_scores: dict of neuron scores.
		p_values: dict of p-values.
		n_label: number of neurons to label (with best p-value, highest x).
		...other args as before...
	"""
	model_lower = model.lower()
	use_diff = model_lower in ('improvement', 'diff', 'difference')
	idx = 1 if model_lower == 'hgm' else 0
	keys = list(p_values.keys())
	x_vals, y_vals, pvals_valid, valid_keys = [], [], [], []
	for k in keys:
		try:
			arr = neuron_scores[k]
			arr0 = np.asarray(arr[0])
			arr1 = np.asarray(arr[1])
			if use_diff:
				# Use difference (index 1 - index 0)
				if arr0.size == 1 and arr1.size == 1:
					xval = float(arr1.ravel()[0]) - float(arr0.ravel()[0])
				elif arr0.ndim == 0 and arr1.ndim == 0:
					xval = float(arr1) - float(arr0)
				else:
					xval = np.nan
			else:
				arr_idx = np.asarray(arr[idx])
				if arr_idx.ndim == 0:
					xval = float(arr_idx)
				elif arr_idx.size == 1:
					xval = float(arr_idx.ravel()[0])
				else:
					xval = np.nan
		except Exception:
			xval = np.nan
		pval = p_values[k]
		yval = -math.log10(pval) if (pval is not None and pval > 0) else np.nan
		if not (np.isnan(xval) or np.isnan(yval)):
			x_vals.append(xval)
			y_vals.append(yval)
			pvals_valid.append(pval)
			valid_keys.append(k)
	x = np.array(x_vals)
	y = np.array(y_vals)
	pvals = np.array(pvals_valid)
	sig = pvals < p_thresh
	left = x < 0
	right = x > 0
	fig, ax = plt.subplots(figsize=figsize)
	ax.scatter(x[~sig], y[~sig], s=s, alpha=alpha, c='gray', label=f'p ≥ {p_thresh}')
	ax.scatter(x[sig & left], y[sig & left], s=s, alpha=alpha, c='royalblue', label=f'x < 0 & p < {p_thresh}')
	ax.scatter(x[sig & right], y[sig & right], s=s, alpha=alpha, c='crimson', label=f'x > 0 & p < {p_thresh}')
	ax.axhline(-math.log10(p_thresh), color='gray', linestyle='--', lw=1, label=f'p = {p_thresh}')
	ax.axvline(0, color='black', linestyle=':', lw=1)
	if use_diff:
		xlab = f"HGM r2 score improvement"
	else:
		xlab = f"{model.upper()} neuron score"
	ax.set_xlabel(xlab)
	ax.set_ylabel('-log10(p-value)')
	ax.set_title(title)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.grid(True, linestyle=':', alpha=0.5)
	# ax.legend()

	# Label only the neurons with the best (lowest) p-value, sorted by x descending, and only if their ID does not start with a digit
	if (n_label is None or n_label > 0) and len(valid_keys) > 0:
		# Filter valid_keys and their indices to only those whose first char is not a digit
		non_digit_idxs = [i for i, k in enumerate(valid_keys) if not str(k)[0].isdigit()]
		if len(non_digit_idxs) == 0:
			pass  # nothing to label
		else:
			min_p = np.nanmin([p_values[valid_keys[i]] for i in non_digit_idxs])
			idxs = [i for i in non_digit_idxs if p_values[valid_keys[i]] == min_p]
			idxs_sorted = sorted(idxs, key=lambda i: x[i], reverse=True)
			if n_label is None:
				idxs_top = idxs_sorted  # label all with best p-value
			else:
				idxs_top = idxs_sorted[:n_label]
			n_labels = len(idxs_top)
			if n_labels > 0:
				xlim_plot = ax.get_xlim()
				ylim_plot = ax.get_ylim()
				# Move label further to the right (e.g., 2% beyond right border)
				x_label = xlim_plot[1] + 0.02 * (xlim_plot[1] - xlim_plot[0])
				# Highest x gets highest y label position
				y_positions = np.linspace(ylim_plot[1], ylim_plot[0], n_labels)
				for i, idx in enumerate(idx_top := idxs_top):
					k = valid_keys[idx]
					ax.annotate(
						k,
						xy=(x[idx], y[idx]),
						xytext=(x_label, y_positions[i]),
						textcoords='data',
						fontsize=10,
						color='black',
						weight='bold',
						ha='left',
						va='center',
						bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
						arrowprops=dict(arrowstyle='-', lw=1, color='gray', alpha=0.7, shrinkA=0, shrinkB=5)
					)

	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=300)
	if show:
		plt.show()
	plt.close()
