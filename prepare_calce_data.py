from src.data.calce_loader import DEFAULT_CALCE_BATTERIES, load_calce_battery_data


def main():
    battery_data = load_calce_battery_data(
        "data/raw/CALCE",
        DEFAULT_CALCE_BATTERIES,
        auto_download=True,
        clean=True,
    )
    for battery_name, (cycles, capacities) in battery_data.items():
        print(
            f"{battery_name}: {len(cycles)} cycles, "
            f"capacity {capacities[0]:.4f} -> {capacities[-1]:.4f} Ah"
        )


if __name__ == "__main__":
    main()
