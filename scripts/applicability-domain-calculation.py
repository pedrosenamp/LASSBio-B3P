"""Paste the WHOLE file into a modern KNIME Python Script node with TWO input
tables and ONE output table. No Python Object ports are needed.

INPUT 0: training set ONLY, with the numeric attributes selected for AD.
INPUT 1: test/VS set, with the SAME attributes and preprocessing.
Keep 'Old RowID' in Input 1 if the final Joiner uses that column.
The original KNIME RowIDs are also retained in the output.

OUTPUT: RowIDs + optional Old RowID + AD result columns. Join this compact
output to the prediction table. Large auxiliary molecular/fingerprint object
columns are not copied. Expanded numeric fingerprint bits ARE supported.

Method: Euclidean distance to the closest training compound. Threshold:
  1. Compute the mean of all distinct unordered training-pair distances.
  2. Retain distances STRICTLY below that mean.
  3. threshold = mean(retained) + Z * std(retained).
This follows the DOCUMENTED Enalos Domain - Similarity approach, not a
claim of bit-for-bit equivalence to the Java node. Conventions are explicit:
each unordered pair counted once, no self distances, population std by
default (STD_DDOF=0), and boundary distance <= threshold is Inside.
Compare a small run against the current node before replacing a published
analysis. A known existing threshold can be supplied as FIXED_THRESHOLD.

References: https://doi.org/10.1039/C4RA07756C (section 4.1.5).
Do not change feature selection, scaling or Z based on test-set performance.
This is a geometric AD flag, not a probability or guarantee of accuracy.

Memory: training attributes + one native KNIME input batch + fixed-size
query/output blocks + one distance block. NO full VS x training distance
matrix, NO VS x VS matrix, and NO full training x training matrix are stored.
Exact distance work still scales with reference rows x query rows x features.
Dependencies: numpy, pandas, scipy, pyarrow, and KNIME Python Integration.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.spatial.distance import cdist, pdist
from itertools import chain
import knime.scripting.io as knio

# ============================ CONFIGURATION ============================

# None: infer numeric columns from the FILTERED training table. Prefer the
# existing Column Filter Configuration to select exactly your model attributes.
# Alternatively supply explicit names: FEATURE_COLUMNS = ["MolWt", "TPSA"]
FEATURE_COLUMNS = None
EXCLUDE_COLUMNS = {
    "Activity", "activity", "Class", "class", "Label", "label", "Target",
    "Old RowID", "RowID", "ID", "SMILES", "Prediction (GNN)",
    "P (active)", "P (inactive)",
}
ID_COLUMN = "Old RowID"  # Optional join key; excluded from distance features.
Z = 0.5                 # Set to the SAME Z configured in the existing node.
STD_DDOF = 0            # 0: population std; 1: sample std.
FIXED_THRESHOLD = None  # Optional pre-established threshold in the SAME space.
STANDARDIZE = False    # False preserves distances in the input representation.
# If True, means/std are fitted ONLY on Input 0, then reused for Input 1.
# Do not enable it if your upstream data are already scaled as intended.

QUERY_BLOCK_ROWS = 1000  # ALSO the output batch size, except the final batch.
REFERENCE_BLOCK_ROWS = 1024
CALIBRATION_BLOCK_ROWS = 512
# Default maximum raw distance buffer: 1000*1024*8 = 7.8 MiB for screening.
# Calibration uses at most 512*512*8 = 2 MiB. Other arrays add to total RAM.
INVALID_QUERY_POLICY = "mark"  # 'mark' keeps invalid rows as Unassessable;
# 'error' stops with affected RowIDs. Invalid training rows ALWAYS stop.

# ============================== HELPERS ================================

class RunningStats:
    """Merge per-block moments without storing all distances."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).ravel()
        count = values.size
        if count == 0:
            return
        if not np.isfinite(values).all():
            raise ValueError("Non-finite distance: check feature magnitudes/scaling.")
        block_mean = float(values.mean())
        delta_values = values - block_mean
        block_m2 = float(np.dot(delta_values, delta_values))
        total = self.n + count
        delta = block_mean - self.mean
        self.m2 += block_m2 + delta * delta * self.n * count / total
        self.mean += delta * count / total
        self.n = total


def training_pair_blocks(x, block_rows):
    """Distinct unordered pairs, including duplicates but excluding i == j."""
    for start in range(0, len(x), block_rows):
        a = x[start:start + block_rows]
        if len(a) > 1:
            yield pdist(a, metric="euclidean")
        for other in range(start + block_rows, len(x), block_rows):
            yield cdist(a, x[other:other + block_rows], metric="euclidean").ravel()


def calibrate_threshold(x, z, ddof, block_rows):
    if len(x) < 2:
        raise ValueError("At least two training rows are needed for calibration.")
    all_pairs = RunningStats()
    for distances in training_pair_blocks(x, block_rows):
        all_pairs.add(distances)
    lower_pairs = RunningStats()
    for distances in training_pair_blocks(x, block_rows):
        lower_pairs.add(distances[distances < all_pairs.mean])
    if lower_pairs.n == 0:
        raise ValueError(
            "No training distances are strictly below the mean: the documented "
            "threshold formula is undefined for this reference set. Check the "
            "training attributes or supply a justified FIXED_THRESHOLD."
        )
    if lower_pairs.n <= ddof:
        raise ValueError("Too few retained training distances for STD_DDOF.")
    std = float(np.sqrt(max(0.0, lower_pairs.m2 / (lower_pairs.n - ddof))))
    threshold = lower_pairs.mean + z * std
    if not np.isfinite(threshold):
        raise ValueError("The calibrated AD threshold is not finite.")
    return float(threshold), {
        "all_pair_count": all_pairs.n,
        "all_pair_mean": all_pairs.mean,
        "retained_pair_count": lower_pairs.n,
        "retained_pair_mean": lower_pairs.mean,
        "retained_pair_std": std,
    }


def closest_reference(query, reference, block_rows):
    """Exact nearest reference, with bounded temporary distance storage."""
    best = np.full(len(query), np.inf, dtype=np.float64)
    nearest = np.full(len(query), -1, dtype=np.int64)
    for start in range(0, len(reference), block_rows):
        distances = cdist(query, reference[start:start + block_rows], metric="euclidean")
        if not np.isfinite(distances).all():
            raise ValueError("Non-finite distance: check feature magnitudes/scaling.")
        local = np.argmin(distances, axis=1)
        minimum = distances[np.arange(len(query)), local]
        improved = minimum < best  # Ties keep the first training row.
        best[improved] = minimum[improved]
        nearest[improved] = start + local[improved]
    return best, nearest


def numeric_features(frame, columns):
    # Text/non-finite query features become explicitly unassessable, never zero.
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def fixed_query_chunks(batches, columns, block_rows):
    """Only selected numeric attributes and the optional join key enter pandas."""
    remainder = None
    for batch in batches:
        frame = batch[columns].to_pandas()
        if remainder is not None:
            frame = pd.concat([remainder, frame], axis=0)
            remainder = None
        offset = 0
        while len(frame) - offset >= block_rows:
            yield frame.iloc[offset:offset + block_rows].copy()
            offset += block_rows
        if offset < len(frame):
            remainder = frame.iloc[offset:].copy()
    if remainder is not None and len(remainder):
        yield remainder


def output_schema(key_type):
    fields = [pa.field("<RowID>", pa.string())]
    if key_type is not None:
        fields.append(pa.field(ID_COLUMN, key_type))
    fields.extend([
        pa.field("AD distance", pa.float64()),
        pa.field("AD threshold", pa.float64()),
        pa.field("AD status", pa.string()),
        pa.field("AD nearest training RowID", pa.string()),
        pa.field("AD nearest training ID", pa.string()),
    ])
    return pa.schema(fields)


def result_batch(frame, distances, threshold, status, nearest_rowids, nearest_ids, schema):
    # Reconstruct primitive arrays explicitly: one physical Arrow RecordBatch
    # per output block, even if pandas preserved internal Arrow fragments.
    arrays = [pa.array(frame.index.astype(str).tolist(), type=pa.string())]
    if ID_COLUMN in schema.names:
        arrays.append(pa.array(frame[ID_COLUMN].tolist(), type=schema.field(ID_COLUMN).type,
                               from_pandas=True))
    arrays.extend([
        pa.array(distances, type=pa.float64(), from_pandas=True),
        pa.array(np.full(len(frame), threshold), type=pa.float64()),
        pa.array(status, type=pa.string()),
        pa.array(nearest_rowids, type=pa.string()),
        pa.array(nearest_ids, type=pa.string()),
    ])
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def main():
    for name, value in [("QUERY_BLOCK_ROWS", QUERY_BLOCK_ROWS),
                        ("REFERENCE_BLOCK_ROWS", REFERENCE_BLOCK_ROWS),
                        ("CALIBRATION_BLOCK_ROWS", CALIBRATION_BLOCK_ROWS)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if STD_DDOF not in (0, 1) or not np.isfinite(Z) or Z < 0:
        raise ValueError("Use STD_DDOF 0 or 1, and a finite nonnegative Z.")
    if INVALID_QUERY_POLICY not in ("mark", "error"):
        raise ValueError("INVALID_QUERY_POLICY must be 'mark' or 'error'.")

    # Only the training table is fully loaded. The test/VS table is streamed.
    train = knio.input_tables[0].to_pandas()
    query_table = knio.input_tables[1]
    if len(train) == 0:
        raise ValueError("The reference training table is empty.")
    if FEATURE_COLUMNS is None:
        features = [name for name in train.columns
                    if name not in EXCLUDE_COLUMNS and name != ID_COLUMN
                    and not name.startswith(("AD ", "Prediction", "P ("))
                    and pd.api.types.is_numeric_dtype(train[name].dtype)]
    else:
        features = list(FEATURE_COLUMNS)
    if not features or len(features) != len(set(features)):
        raise ValueError("Select at least one feature; feature names must be unique.")
    if any(name in EXCLUDE_COLUMNS or name == ID_COLUMN for name in features):
        raise ValueError("Do not include Activity, join keys or metadata as AD features.")
    missing_train = [name for name in features if name not in train.columns]
    missing_query = [name for name in features if name not in query_table.column_names]
    if missing_train or missing_query:
        raise ValueError(f"Missing training attributes: {missing_train}; "
                         f"missing query attributes: {missing_query}.")
    print(f"Reference: TRAINING ONLY, {len(train):,} rows; {len(features)} attributes.", flush=True)
    print("AD attributes: " + ", ".join(features), flush=True)

    x_train = numeric_features(train, features)
    valid_train = np.isfinite(x_train).all(axis=1)
    if not valid_train.all():
        bad = train.index[~valid_train].astype(str).tolist()[:10]
        raise ValueError(f"Missing/non-finite/non-numeric TRAINING attributes; RowIDs: {bad}.")
    train_rowids = train.index.astype(str).to_numpy()
    train_ids = (train[ID_COLUMN].astype(str).to_numpy()
                 if ID_COLUMN in train.columns else train_rowids)
    center = np.zeros(x_train.shape[1], dtype=np.float64)
    scale = np.ones(x_train.shape[1], dtype=np.float64)
    if STANDARDIZE:
        center = x_train.mean(axis=0)
        scale = x_train.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("Non-finite training scaling parameters.")
        x_train = (x_train - center) / scale
    x_train = np.ascontiguousarray(x_train)
    print(f"Standardize: {STANDARDIZE}; Z: {Z}; std ddof: {STD_DDOF}.", flush=True)
    if FIXED_THRESHOLD is None:
        threshold, statistics = calibrate_threshold(
            x_train, Z, STD_DDOF, CALIBRATION_BLOCK_ROWS
        )
        print(f"Training-only calibration: {statistics}", flush=True)
    else:
        threshold = float(FIXED_THRESHOLD)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("FIXED_THRESHOLD must be finite and nonnegative.")
        print("Using the supplied, pre-established FIXED_THRESHOLD.", flush=True)
    print(f"AD threshold: {threshold:.12g}.", flush=True)

    has_key = ID_COLUMN in query_table.column_names
    selected = features + ([ID_COLUMN] if has_key else [])
    # Inspect ONLY the first native batch. A zero-row view of the full
    # source can still materialize the entire table in some KNIME versions.
    native_batches = iter(query_table.batches())
    first_batch = next(native_batches, None)
    if first_batch is not None:
        key_type = first_batch.to_pyarrow().column(ID_COLUMN).type if has_key else None
        batches = chain([first_batch], native_batches)
    else:
        # No native batches: the query is empty; only schema is needed.
        key_type = query_table.to_pyarrow().column(ID_COLUMN).type if has_key else None
        batches = iter(())
    if has_key and not (pa.types.is_string(key_type) or pa.types.is_integer(key_type)):
        raise TypeError(f"{ID_COLUMN} must be a string or integer join key, got {key_type}.")
    schema = output_schema(key_type)
    output = knio.BatchOutputTable.create(row_ids="keep")
    total = inside_total = outside_total = invalid_total = 0
    for number, frame in enumerate(
        fixed_query_chunks(batches, selected, QUERY_BLOCK_ROWS), start=1
    ):
        if has_key and frame[ID_COLUMN].isna().any():
            raise ValueError(f"Missing values in join key '{ID_COLUMN}'; "
                             "fix the identifiers before joining AD results.")
        values = numeric_features(frame, features)
        valid = np.isfinite(values).all(axis=1)
        if INVALID_QUERY_POLICY == "error" and not valid.all():
            raise ValueError("Invalid QUERY attributes; RowIDs: " +
                             ", ".join(frame.index[~valid].astype(str)[:10]))
        distance = np.full(len(frame), np.nan)
        nearest_rowids = [None] * len(frame)
        nearest_ids = [None] * len(frame)
        status = np.full(len(frame), "Unassessable", dtype=object)
        if valid.any():
            query = np.ascontiguousarray((values[valid] - center) / scale)
            minimum, indices = closest_reference(query, x_train, REFERENCE_BLOCK_ROWS)
            distance[valid] = minimum
            status[valid] = np.where(minimum <= threshold, "Inside", "Outside")
            for position, reference_index in zip(np.flatnonzero(valid), indices):
                nearest_rowids[position] = train_rowids[reference_index]
                nearest_ids[position] = train_ids[reference_index]
        record = result_batch(frame, distance, threshold, status,
                              nearest_rowids, nearest_ids, schema)
        output.append(record)
        total += len(frame)
        inside_total += int(np.sum(status == "Inside"))
        outside_total += int(np.sum(status == "Outside"))
        invalid_total += int(np.sum(~valid))
        print(f"AD block {number}: {len(frame):,} rows; {total:,} total.", flush=True)
    if total == 0:
        empty_record = pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in schema], schema=schema
        )
        empty = pa.Table.from_batches([empty_record])
        knio.output_tables[0] = knio.Table.from_pyarrow(empty, row_ids="keep")
    else:
        knio.output_tables[0] = output
    print(f"AD completed: {total:,} rows; Inside={inside_total:,}; "
          f"Outside={outside_total:,}; Unassessable={invalid_total:,}.", flush=True)
    print(f"Threshold={threshold:.12g}; Euclidean; no full screening distance matrix.", flush=True)
    if not has_key:
        print(f"No '{ID_COLUMN}' column was present: join the output by KNIME RowID.", flush=True)


main()

