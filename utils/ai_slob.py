from collections import defaultdict
from typing import Callable, Dict, Iterable, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import json

try:
	import seaborn as sns

	_HAS_SNS = True
except Exception:
	_HAS_SNS = False
	from scipy.stats import gaussian_kde


def merge_neuron_scores_per_shuffle(
		neuron_scores_per_shuffle: Iterable[Dict[str, Tuple[np.ndarray, np.ndarray]]],
		stack: bool = False,
		ignore_missing: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
	"""
	Merge a list of per-shuffle dicts into a single dict where each value is a tuple
	of numpy arrays (bom_values, hgm_values).

	Behavior:
	  - For each neuron label, collect the per-shuffle entries into lists.
	  - Convert those lists to numpy arrays before returning:
		  * If all entries for a neuron have identical numeric shapes and stacking succeeds,
			results will be a numeric ndarray with shape (n_shuffles_present, ...).
		  * Otherwise, results will be an object-dtype ndarray: np.array(list, dtype=object).
	  - If a neuron is missing from some shuffles and ignore_missing=False, None placeholders
		are inserted. The function will attempt to replace None with NaNs matching the shape
		of the first non-None entry (useful for numeric arrays), otherwise None will remain
		and the returned array will be object-dtype.

	Returns:
	  merged: dict mapping label -> (bom_values_array, hgm_values_array)
	"""
	bom_collect = defaultdict(list)
	hgm_collect = defaultdict(list)

	shuffle_list = list(neuron_scores_per_shuffle)
	# gather the set of all labels that appear anywhere
	all_labels = set()
	for d in shuffle_list:
		all_labels.update(d.keys())

	for d in shuffle_list:
		for label in all_labels:
			if label in d:
				pair = d[label]
				bom_collect[label].append(pair[0])
				hgm_collect[label].append(pair[1])
			else:
				if not ignore_missing:
					bom_collect[label].append(None)
					hgm_collect[label].append(None)
				# if ignore_missing, we simply skip this shuffle for this neuron

	def to_numpy_array(lst):
		"""Convert a python list of entries into a numpy array.

		- If all entries are None -> return empty object array of length 0.
		- Try to replace None with NaN-filled arrays/scalars when possible (if first non-None
		  entry is a numeric ndarray or numeric scalar).
		- Try np.stack for homogeneous numeric arrays (stack==True or not; we always attempt stacking
		  to produce numeric arrays when possible). If stacking fails, return object-dtype array.
		"""
		if len(lst) == 0:
			return np.array([], dtype=object)

		# detect first non-None
		first_non_none = None
		for x in lst:
			if x is not None:
				first_non_none = x
				break

		if first_non_none is None:
			# all None -> return object-dtype array of None
			return np.array(lst, dtype=object)

		# If there are None placeholders, try to replace them with NaN equivalents
		if any(x is None for x in lst):
			filled = []
			for x in lst:
				if x is not None:
					filled.append(x)
				else:
					# create NaN placeholder matching shape/type of first_non_none
					if isinstance(first_non_none, np.ndarray):
						try:
							nan_fill = np.full(first_non_none.shape, np.nan, dtype=first_non_none.dtype)
						except Exception:
							# fallback to float NaNs
							nan_fill = np.full(first_non_none.shape, np.nan, dtype=float)
						filled.append(nan_fill)
					else:
						# scalar or other: use np.nan
						filled.append(np.nan)
			lst = filled

		# Now try to stack into a numeric ndarray
		try:
			stacked = np.stack(lst, axis=0)
			# Normalize stacked shapes:
			# - If stacking produced a 1D array of scalars, make it (n, 1)
			# - Otherwise collapse all trailing dimensions into a single feature dimension
			if stacked.ndim == 1:
				stacked = stacked.reshape(stacked.shape[0], 1)
			else:
				# collapse dims after the first into one feature dimension
				stacked = stacked.reshape(stacked.shape[0], -1)
			return stacked
		except Exception:
			# stacking failed (incompatible shapes or types) -> return object array
			return np.array(lst, dtype=object)

	merged = {}
	for label in sorted(all_labels):
		bom_list = bom_collect[label]
		hgm_list = hgm_collect[label]

		bom_out = to_numpy_array(bom_list)
		hgm_out = to_numpy_array(hgm_list)

		merged[label] = (bom_out, hgm_out)

	return merged


def compute_neuron_percentiles(
		neuron_scores_actual: Dict[str, Tuple[np.ndarray, np.ndarray]],
		neuron_scores_per_shuffle: Iterable[Dict[str, Tuple[np.ndarray, np.ndarray]]],
		model: Union[str, int] = 'hgm',
		agg_func: Callable[[np.ndarray], float] = None,
) -> Dict[str, float]:
	"""
	For each neuron key in `neuron_scores_actual`, compute the percentile of the actual score
	relative to the distribution built from `neuron_scores_per_shuffle`.

	Args:
	  neuron_scores_actual: dict mapping neuron_label -> (bom_array, hgm_array)
		(matches the structure produced by your `assemble` function).
	  neuron_scores_per_shuffle: iterable (list) of shuffle dicts,
		each mapping neuron_label -> (bom_array, hgm_array).
	  model: 'hgm' or 'bom' or 0/1 integer selecting the tuple element.
	  agg_func: function to reduce an array to a scalar (default: np.nanmean of flattened array).

	Returns:
	  dict mapping neuron_label -> percentile (0..100) where percentile p means
	  actual > p% of shuffle values; ties are handled using mid-rank:
	  (count_less + 0.5 * count_equal) / N * 100.
	"""
	# choose index 0=bom, 1=hgm
	if isinstance(model, str):
		model = model.lower()
		if model == 'bom':
			model_index = 0
		elif model == 'hgm':
			model_index = 1
		else:
			raise ValueError("model must be 'bom' or 'hgm' (or 0/1).")
	else:
		model_index = int(model)

	if agg_func is None:
		def agg_func_default(x: np.ndarray) -> float:
			# flatten and take nanmean by default
			if x is None:
				return np.nan
			x = np.asarray(x)
			if x.size == 0:
				return np.nan
			return float(np.nanmean(x))

		agg_func = agg_func_default

	percentiles = {}
	# Convert shuffles to list to iterate multiple times if needed
	shuffles_list = list(neuron_scores_per_shuffle)

	for label, actual_pair in neuron_scores_actual.items():
		try:
			actual_arr = actual_pair[model_index]
		except Exception:
			percentiles[label] = np.nan
			continue

		actual_value = agg_func(actual_arr)

		# Build distribution across shuffles for this neuron label
		dist_values = []
		for shuffle_dict in shuffles_list:
			if label not in shuffle_dict:
				# some shuffles might not contain this neuron -> skip
				continue
			shuffle_arr = shuffle_dict[label][model_index]
			val = agg_func(shuffle_arr)
			# If agg returns nan (e.g. empty), skip
			if np.isnan(val):
				continue
			dist_values.append(val)

		if len(dist_values) == 0 or np.isnan(actual_value):
			percentiles[label] = np.nan
			continue

		dist = np.asarray(dist_values)
		count_less = np.sum(dist < actual_value)
		count_equal = np.sum(dist == actual_value)
		N = len(dist)

		# mid-rank percentile
		pct = (count_less + 0.5 * count_equal) / N * 100.0
		percentiles[label] = float(pct)

	return percentiles


# Plot distributions for neurons: histogram of shuffle values + actual value marker


def agg_default(x):
	if x is None:
		return np.nan
	a = np.asarray(x)
	if a.size == 0:
		return np.nan
	return float(np.nanmean(a))


def compute_dist_and_percentile(label, neuron_scores_actual, neuron_scores_per_shuffle,
								model_index=1, agg_func=agg_default):
	# get actual scalar
	if label not in neuron_scores_actual:
		return None, None, "missing_actual"
	actual_arr = neuron_scores_actual[label][model_index]
	actual_val = agg_func(actual_arr)
	# gather shuffle values
	dist = []
	for shuffle_dict in neuron_scores_per_shuffle:
		if label not in shuffle_dict:
			continue
		val = agg_func(shuffle_dict[label][model_index])
		if np.isnan(val):
			continue
		dist.append(val)
	if len(dist) == 0 or np.isnan(actual_val):
		return np.array(dist), actual_val, "insufficient_shuffles"
	dist = np.asarray(dist)
	# mid-rank percentile
	count_less = np.sum(dist < actual_val)
	count_equal = np.sum(dist == actual_val)
	N = len(dist)
	pct = (count_less + 0.5 * count_equal) / N * 100.0
	return dist, actual_val, pct


def plot_first_n_neurons(neuron_scores_actual, neuron_scores_per_shuffle,
						 model='hgm', agg_func=agg_default, n=None,
						 neuron_labels=None, bins=30, figsize=(18, 7), threshold=95):
	"""
	Plot neurons according to selection logic:
	  - If `neuron_labels` is provided, plot those (up to n).
	  - Else if `threshold` is not None, compute percentiles and select neurons with
		percentile >= threshold. If that selection is non-empty, plot those neurons.
		If the caller provided `n` (an integer), the results are capped to `n`; if
		`n` is None (the default), all selected neurons are plotted. If the selection
		is empty, plot ALL neurons (no cap).
	  - Else (threshold is None), fall back to original behavior: choose top-n by actual value.

	Args:
	  threshold: percentile cutoff (0-100). Use None to disable threshold selection.
	"""
	# keep a copy of the original model argument for percentile computation
	original_model_arg = model

	# determine model index (and normalize `model` for later use)
	if isinstance(model, str):
		model = model.lower()
		if model == 'bom':
			model_index = 0
		elif model == 'hgm':
			model_index = 1
		else:
			raise ValueError("model must be 'bom' or 'hgm'")
	else:
		model_index = int(model)

	# Determine which neuron labels to plot
	if neuron_labels is None:
		if threshold is not None:
			# compute percentiles for all neurons
			percentiles = compute_neuron_percentiles(
				neuron_scores_actual, neuron_scores_per_shuffle,
				model=original_model_arg, agg_func=agg_func
			)
			# select labels meeting the threshold
			selected = [label for label, pct in percentiles.items() if not np.isnan(pct) and pct >= threshold]
			# sort selected by percentile descending (so most significant first)
			selected.sort(key=lambda L: percentiles[L], reverse=True)

			if len(selected) > 0:
				# If caller provided n (int), cap to n; otherwise (n is None) plot all selected
				if n is None:
					chosen = selected
				else:
					chosen = selected[:n]
			else:
				# none meet the threshold -> plot ALL neurons (no cap)
				chosen = list(neuron_scores_actual.keys())
		else:
			# original behavior: choose top-n by actual aggregated value
			entries = []
			for label, pair in neuron_scores_actual.items():
				val = agg_func(pair[model_index])
				entries.append((label, val))
			entries = [e for e in entries if not np.isnan(e[1])]
			entries.sort(key=lambda kv: kv[1], reverse=True)
			chosen = [label for label, _ in entries][:n]
			if len(chosen) < n:
				for label in neuron_scores_actual.keys():
					if label not in chosen:
						chosen.append(label)
					if len(chosen) >= n:
						break
	else:
		# explicit list provided by user: respect it but cap at n
		chosen = list(neuron_labels)[:n]

	# Remove neurons that are missing actual values entirely (user requested to skip them).
	# A neuron is considered missing if it's not in neuron_scores_actual or its aggregated
	# actual value is NaN.
	filtered_chosen = []
	for label in chosen:
		if label not in neuron_scores_actual:
			# skip missing actual
			continue
		try:
			actual_val_check = agg_func(neuron_scores_actual[label][model_index])
		except Exception:
			# treat as missing if aggregation fails
			continue
		if np.isnan(actual_val_check):
			continue
		filtered_chosen.append(label)

	# use filtered list for plotting
	chosen = filtered_chosen

	# plotting grid: maximum 5 plots per row, create as many rows as needed
	total_to_plot = len(chosen)
	# determine number of columns (at most 5)
	cols = min(5, total_to_plot) if total_to_plot > 0 else 1
	rows = int(np.ceil(total_to_plot / cols)) if total_to_plot > 0 else 1

	# Compute automatic figure size so each subplot has enough space.
	# Derive a per-subplot base from the provided `figsize` defaults:
	base_fig_w, base_fig_h = figsize
	# historically figsize=(18,7) was used for up to 5 cols and 2 rows, so
	# use that as baseline per-axis sizes
	per_col_w = max(2.5, base_fig_w / 5.0)
	per_row_h = max(2.5, base_fig_h / 2.0)

	fig_w = per_col_w * cols
	fig_h = per_row_h * rows
	# clamp to avoid absurdly large figures
	fig_w = min(fig_w, 40)
	fig_h = min(fig_h, 25)

	fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)
	axes = np.array(axes).flatten()

	# if nothing to plot, show message and return
	if total_to_plot == 0:
		fig.text(0.5, 0.5, "No neurons selected to plot.", ha='center', va='center', fontsize=12)
		thresh_msg = f" (percentile>= {threshold})" if threshold is not None else ""
		plt.suptitle(
			f"Shuffle distributions and actual scores (model={original_model_arg}){thresh_msg} — plotted 0 neurons",
			y=1.02)
		plt.show()
		return []

	# safe default for loop index
	i = -1

	for i, label in enumerate(chosen):
		ax = axes[i]
		dist, actual_val, pct_or_status = compute_dist_and_percentile(label,
																	  neuron_scores_actual,
																	  neuron_scores_per_shuffle,
																	  model_index=model_index,
																	  agg_func=agg_func)
		if isinstance(pct_or_status, str):
			# status message: missing or insufficient
			ax.text(0.5, 0.5, f"{label}\nStatus: {pct_or_status}",
					ha='center', va='center', transform=ax.transAxes)
			ax.set_title(label)
			ax.set_ylabel("")
			ax.set_xlabel("")
			ax.set_yticks([])
			continue

		pct = pct_or_status
		# histogram
		if len(dist) > 0:
			ax.hist(dist, bins=bins, density=True, alpha=0.6, color='C0', edgecolor='k')
			if _HAS_SNS:
				try:
					sns.kdeplot(dist, ax=ax, color='C1')
				except Exception:
					pass
			else:
				# fallback KDE using scipy
				try:
					kde = gaussian_kde(dist)
					xs = np.linspace(np.min(dist), np.max(dist), 200)
					ax.plot(xs, kde(xs), color='C1', lw=1)
				except Exception:
					pass
		else:
			ax.text(0.5, 0.5, "no shuffle values", ha='center', va='center', transform=ax.transAxes)

		# mark actual value
		ax.axvline(actual_val, color='red', linestyle='--', lw=2, label=f'actual ({actual_val:.3g})')
		title_extra = f" — percentile {pct:.1f}" if not np.isnan(pct) else ""
		ax.set_title(f"{label}{title_extra}")
		ax.legend(loc='upper right', fontsize='small')
		ax.set_xlabel("aggregated score (mean)")
		ax.set_ylabel("density")

	# hide unused axes if any
	if i >= 0:
		for j in range(i + 1, len(axes)):
			axes[j].axis('off')

	thresh_msg = f" (percentile>= {threshold})" if threshold is not None else ""
	plt.suptitle(
		f"Shuffle distributions and actual scores (model={original_model_arg}){thresh_msg} — plotted {total_to_plot} neurons",
		y=1.02)
	plt.show()


def plot_percentile_hist(percentiles, bins=50, title=None, threshold=None,
						 figsize=(8, 4), save_path=None, show=True):
	"""
	Plot a histogram of percentiles.

	Args:
	  percentiles: dict mapping label->percentile (0..100) or an iterable/array of percentile values.
	  bins: histogram bins (int or sequence)
	  title: optional title string
	  threshold: optional percentile threshold to mark with a vertical line (e.g., 95)
	  figsize: figure size tuple
	  save_path: if provided, save the figure to this path
	  show: whether to call plt.show() (set False for headless runs)

	Returns:
	  (counts, bin_edges) as returned by plt.hist
	"""
	# Accept dict or iterable
	if isinstance(percentiles, dict):
		vals = [v for v in percentiles.values() if not np.isnan(v)]
	else:
		vals = [float(v) for v in percentiles if not np.isnan(v)]

	fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
	if len(vals) == 0:
		ax.text(0.5, 0.5, 'No percentile values to plot', ha='center', va='center')
		if title is None:
			ax.set_title('Percentile histogram')
		else:
			ax.set_title(title)
		if save_path:
			fig.savefig(save_path)
		if show:
			plt.show()
		return np.array([]), np.array([])

	counts, bin_edges, patches = ax.hist(vals, bins=bins, color='C0', edgecolor='k', alpha=0.7)
	ax.set_xlabel('percentile')
	ax.set_ylabel('count')
	if title is None:
		ax.set_title('Percentile distribution')
	else:
		ax.set_title(title)

	# annotate summary statistics
	mean = float(np.mean(vals))
	med = float(np.median(vals))
	ax.axvline(mean, color='C2', linestyle='-', linewidth=1, label=f'mean {mean:.1f}')
	ax.axvline(med, color='C3', linestyle='--', linewidth=1, label=f'median {med:.1f}')

	# prepare threshold annotation count default
	above = 0
	# draw threshold line and include count in its legend label so it is clearly visible
	above = 0
	if threshold is not None:
		above = sum(1 for v in vals if v >= threshold)
		ax.axvline(threshold, color='red', linestyle='-.', linewidth=1.5,
				   label=f'threshold {threshold} (n={above})')

	# create legend (legend now contains the threshold + count)
	legend = ax.legend(loc='upper left', fontsize='small')

	# Additionally, indicate how many neurons are at 100% (or effectively 100).
	# Use a small epsilon to account for floating rounding.
	try:
		count_100 = int(sum(1 for v in vals if v >= 100.0 - 1e-8))
	except Exception:
		count_100 = 0

	if count_100 > 0:
		# Try to place the annotation just below the legend bbox in axes coords.
		try:
			fig.canvas.draw()
			try:
				renderer = fig.canvas.get_renderer()
			except Exception:
				renderer = getattr(fig.canvas, 'renderer', None)

			if renderer is not None:
				bbox_disp = legend.get_window_extent(renderer)
				bbox_axes = ax.transAxes.inverted().transform_bbox(bbox_disp)
				ann_x = bbox_axes.x0
				ann_y = bbox_axes.y0 - 0.03
				# clamp to inside axes
				ann_x = min(max(ann_x, 0.01), 0.95)
				ann_y = max(ann_y, 0.01)
			else:
				ann_x, ann_y = 0.05, 0.80
		except Exception:
			ann_x, ann_y = 0.05, 0.80

		ax.text(ann_x, ann_y, f'100%: {count_100}', ha='left', va='top', transform=ax.transAxes,
				bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize='small')

	if save_path:
		fig.savefig(save_path)
	if show:
		plt.show()

	return counts, bin_edges


def build_sorted_percentiles(percentiles, neuron_scores,
							 model_index=1, agg_func=agg_default):
	entries = []
	for label, pct in percentiles.items():
		# skip missing/NaN percentiles
		if pct is None or np.isnan(pct):
			continue
		# skip if neuron not present
		if label not in neuron_scores:
			continue
		# get aggregated actual HGM value for tie-breaking
		try:
			actual_val = agg_func(neuron_scores[label][model_index])
		except Exception:
			continue
		if actual_val is None or np.isnan(actual_val):
			continue
		entries.append((label, float(pct), float(actual_val)))

	# Sort by percentile desc, then actual_val desc
	entries.sort(key=lambda t: (t[1], t[2]), reverse=True)

	# Create an OrderedDict mapping label -> (percentile, actual_val)
	ordered = OrderedDict((label, (pct, actual)) for label, pct, actual in entries)

	return entries, ordered

