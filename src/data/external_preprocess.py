"""
Reusable helpers for adapting external RUL datasets to this project's format.

These functions are intentionally not imported by main.py. Use them when you want
to validate the model on another dataset and need to convert tabular run-to-failure
data into the existing battery_data format:

    {
        "unit_001": [cycle_sequence, health_indicator_sequence],
        "unit_002": [cycle_sequence, health_indicator_sequence],
    }

The health indicator can be capacity, SOH, normalized degradation score, or any
monotonic proxy you want the model to predict.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_external_rul_csv(
    csv_path,
    unit_col="unit_id",
    cycle_col="cycle",
    target_col="capacity",
    sort=True,
    dropna=True,
):
    """Load a tabular RUL dataset and convert it to the project data format.

    Expected input shape:
        unit_id, cycle, capacity
        A001,    1,     1.98
        A001,    2,     1.97
        ...

    Parameters
    ----------
    csv_path : str or Path
        Source CSV path.
    unit_col : str
        Column identifying each battery/engine/gearbox/unit.
    cycle_col : str
        Time or cycle index column.
    target_col : str
        Health indicator column to predict.
    sort : bool
        Sort each unit by cycle before building sequences.
    dropna : bool
        Drop rows missing unit/cycle/target values.

    Returns
    -------
    dict
        Project-compatible data dictionary: {unit_id: [cycles, targets]}.
    """
    df = pd.read_csv(csv_path)
    required = [unit_col, cycle_col, target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if dropna:
        df = df.dropna(subset=required)
    if sort:
        df = df.sort_values([unit_col, cycle_col])

    data = {}
    for unit_id, unit_df in df.groupby(unit_col, sort=False):
        cycles = unit_df[cycle_col].to_numpy(dtype=float).tolist()
        targets = unit_df[target_col].to_numpy(dtype=float).tolist()
        if len(targets) >= 2:
            data[str(unit_id)] = [cycles, targets]
    return data


def capacity_to_soh(data_dict, rated_capacity=None, per_unit_initial=True):
    """Convert capacity-like targets to SOH-style normalized targets.

    If rated_capacity is provided, SOH = capacity / rated_capacity.
    Otherwise, each unit can be normalized by its first observed target value.
    """
    converted = {}
    for unit_id, (cycles, targets) in data_dict.items():
        targets = np.asarray(targets, dtype=float)
        if rated_capacity is not None:
            denom = float(rated_capacity)
        elif per_unit_initial:
            denom = float(targets[0])
        else:
            denom = float(np.nanmax(targets))
        denom = max(denom, 1e-8)
        converted[unit_id] = [list(cycles), (targets / denom).tolist()]
    return converted


def rul_to_health_index(data_dict, failure_threshold=None):
    """Convert cycle sequences to a normalized remaining-life health index.

    This is useful for datasets that do not provide capacity/SOH but have full
    run-to-failure trajectories. The generated target is:

        health = remaining_cycles / max_remaining_cycles

    If failure_threshold is provided, targets below that threshold are clipped to 0.
    """
    converted = {}
    for unit_id, (cycles, targets) in data_dict.items():
        n = len(cycles)
        remaining = np.arange(n - 1, -1, -1, dtype=float)
        denom = max(float(remaining[0]), 1.0)
        health = remaining / denom
        if failure_threshold is not None:
            original_targets = np.asarray(targets, dtype=float)
            health = np.where(original_targets <= failure_threshold, 0.0, health)
        converted[unit_id] = [list(cycles), health.tolist()]
    return converted


def filter_short_units(data_dict, min_length):
    """Remove units that are too short for a sliding-window experiment."""
    return {
        unit_id: series
        for unit_id, series in data_dict.items()
        if len(series[1]) > min_length
    }


def save_project_format(data_dict, out_path):
    """Save converted external data as .npy for later use with np.load(...).item()."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data_dict)
    return out_path


def prepare_external_rul_dataset(
    csv_path,
    out_path=None,
    unit_col="unit_id",
    cycle_col="cycle",
    target_col="capacity",
    target_mode="raw",
    rated_capacity=None,
    window_size=16,
):
    """One-stop adapter for external RUL CSV datasets.

    target_mode options:
        "raw"          keep target_col as-is
        "soh"          normalize target_col into SOH
        "rul_health"   ignore target magnitude and create normalized RUL health

    Example
    -------
    data = prepare_external_rul_dataset(
        "gearbox.csv",
        unit_col="bearing_id",
        cycle_col="time_step",
        target_col="health_indicator",
        target_mode="raw",
        out_path="data/raw/gearbox_project_format.npy",
    )
    """
    data = load_external_rul_csv(
        csv_path,
        unit_col=unit_col,
        cycle_col=cycle_col,
        target_col=target_col,
    )

    if target_mode == "soh":
        data = capacity_to_soh(data, rated_capacity=rated_capacity)
    elif target_mode == "rul_health":
        data = rul_to_health_index(data)
    elif target_mode != "raw":
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    data = filter_short_units(data, min_length=window_size + 1)
    if out_path is not None:
        save_project_format(data, out_path)
    return data
