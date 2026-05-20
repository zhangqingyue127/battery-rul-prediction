import os
import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


CALCE_BASE_URL = "https://web.calce.umd.edu/batteries/data"
DEFAULT_CALCE_BATTERIES = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
DEFAULT_CLEANING_CONFIG = {
    "hampel_window": 15,
    "hampel_sigma": 3.0,
    "smooth_window": 31,
    "smooth_polyorder": 2,
    "min_capacity": 0.05,
    "max_capacity": 1.25,
}


def _file_date_key(path):
    match = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xlsx$", Path(path).name)
    if not match:
        return (9999, 12, 31, Path(path).name)
    month, day, year = map(int, match.groups())
    return (2000 + year, month, day, Path(path).name)


def download_calce_data(data_dir, battery_list=None, base_url=CALCE_BASE_URL):
    """Download and extract CALCE CS2 battery zip files if they are missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    battery_list = battery_list or DEFAULT_CALCE_BATTERIES

    for battery_name in battery_list:
        zip_path = data_dir / f"{battery_name}.zip"
        extract_dir = data_dir / battery_name

        if not zip_path.exists():
            url = f"{base_url}/{battery_name}.zip"
            print(f"Downloading {url}")
            urlretrieve(url, zip_path)

        if not any(extract_dir.rglob("*.xlsx")):
            print(f"Extracting {zip_path}")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                zip_file.extractall(extract_dir)


def _read_calce_file_capacities(xlsx_path):
    excel_file = pd.ExcelFile(xlsx_path)
    channel_sheets = [
        sheet for sheet in excel_file.sheet_names
        if sheet.lower().startswith("channel")
    ]
    if not channel_sheets:
        return []

    df = pd.read_excel(
        xlsx_path,
        sheet_name=channel_sheets[0],
        usecols=["Cycle_Index", "Current(A)", "Discharge_Capacity(Ah)"],
    )
    df = df.dropna(subset=["Cycle_Index", "Current(A)", "Discharge_Capacity(Ah)"])
    if df.empty:
        return []

    capacities = []
    for _, cycle_df in df.groupby("Cycle_Index", sort=True):
        discharge_df = cycle_df[cycle_df["Current(A)"] < -1e-3]
        if discharge_df.empty:
            continue
        discharge_capacity = discharge_df["Discharge_Capacity(Ah)"]
        capacity = float(discharge_capacity.max() - discharge_capacity.min())
        if capacity > 0.05:
            capacities.append(capacity)
    return capacities


def _load_calce_battery(data_dir, battery_name):
    battery_dir = Path(data_dir) / battery_name
    xlsx_files = sorted(battery_dir.rglob("*.xlsx"), key=_file_date_key)
    capacities = []
    for xlsx_path in xlsx_files:
        capacities.extend(_read_calce_file_capacities(xlsx_path))

    cycles = list(range(1, len(capacities) + 1))
    return [cycles, capacities]


def _odd_window(window_size, series_length):
    window_size = int(window_size)
    if series_length < 3:
        return 0
    window_size = min(window_size, series_length)
    if window_size % 2 == 0:
        window_size -= 1
    return max(window_size, 3)


def _hampel_filter(values, window_size=15, n_sigmas=3.0):
    series = pd.Series(values, dtype=float)
    rolling_median = series.rolling(
        window=window_size,
        center=True,
        min_periods=max(3, window_size // 2),
    ).median()
    deviation = (series - rolling_median).abs()
    rolling_mad = deviation.rolling(
        window=window_size,
        center=True,
        min_periods=max(3, window_size // 2),
    ).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    outlier_mask = deviation > threshold

    cleaned = series.mask(outlier_mask, rolling_median)
    cleaned = cleaned.interpolate(limit_direction="both")
    return cleaned.to_numpy(dtype=float), int(outlier_mask.fillna(False).sum())


def clean_capacity_sequence(capacities, cleaning_config=None):
    """Repair local CALCE capacity spikes and smooth the degradation trend."""
    config = DEFAULT_CLEANING_CONFIG.copy()
    if cleaning_config:
        config.update(cleaning_config)

    values = np.asarray(capacities, dtype=float)
    if values.size == 0:
        return []

    values = np.clip(values, config["min_capacity"], config["max_capacity"])
    hampel_window = _odd_window(config["hampel_window"], len(values))
    if hampel_window:
        values, _ = _hampel_filter(
            values,
            window_size=hampel_window,
            n_sigmas=config["hampel_sigma"],
        )

    smooth_window = _odd_window(config["smooth_window"], len(values))
    polyorder = int(config["smooth_polyorder"])
    if smooth_window > polyorder + 1:
        values = savgol_filter(values, smooth_window, polyorder, mode="interp")

    values = np.clip(values, config["min_capacity"], config["max_capacity"])
    return values.astype(float).tolist()


def clean_calce_battery_data(battery_data, cleaning_config=None):
    cleaned = {}
    for battery_name, (cycles, capacities) in battery_data.items():
        cleaned[battery_name] = [
            list(cycles),
            clean_capacity_sequence(capacities, cleaning_config),
        ]
    return cleaned


def load_calce_battery_data(
    data_dir,
    battery_list=None,
    auto_download=False,
    clean=True,
    cleaning_config=None,
):
    """Load CALCE CS2 data into the same format used by the NASA workflow."""
    data_dir = Path(data_dir)
    battery_list = battery_list or DEFAULT_CALCE_BATTERIES
    cache_name = "CALCE_Battery_Data_clean.npy" if clean else "CALCE_Battery_Data.npy"
    npy_path = data_dir / cache_name

    if npy_path.exists():
        print(f"Loading cached CALCE data from {npy_path}")
        cached = np.load(npy_path, allow_pickle=True).item()
        return {name: cached[name] for name in battery_list if name in cached}

    raw_cache_path = data_dir / "CALCE_Battery_Data.npy"
    if clean and raw_cache_path.exists():
        raw_data = np.load(raw_cache_path, allow_pickle=True).item()
        battery_data = {
            name: raw_data[name] for name in battery_list if name in raw_data
        }
        cleaned_data = clean_calce_battery_data(battery_data, cleaning_config)
        np.save(npy_path, cleaned_data)
        print(f"Cached cleaned CALCE data saved to {npy_path}")
        return cleaned_data

    if auto_download:
        download_calce_data(data_dir, battery_list)

    battery_data = {}
    missing = []
    for battery_name in battery_list:
        battery_dir = data_dir / battery_name
        if not any(battery_dir.rglob("*.xlsx")):
            missing.append(str(battery_dir))
            continue
        battery_data[battery_name] = _load_calce_battery(data_dir, battery_name)

    if missing:
        raise FileNotFoundError(
            "Missing extracted CALCE folders with xlsx files: " + ", ".join(missing)
        )

    raw_cache_path = data_dir / "CALCE_Battery_Data.npy"
    np.save(raw_cache_path, battery_data)
    print(f"Cached raw CALCE data saved to {raw_cache_path}")
    if clean:
        battery_data = clean_calce_battery_data(battery_data, cleaning_config)

    np.save(npy_path, battery_data)
    print(f"Cached CALCE data saved to {npy_path}")
    return battery_data
