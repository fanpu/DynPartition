from pathlib import Path


def get_folder_path(folder_name):
    base_path = Path(__file__).parent.joinpath(folder_name)

    if not base_path.exists():
        base_path.mkdir()

    return base_path


def get_saved_data_path():
    return get_folder_path("_saved_data")


def get_data_path():
    return get_folder_path("_data")


def get_log_path():
    return get_folder_path("_logs")


def get_plot_path():
    return get_folder_path("_plots")


def save_log_json(log_dict, name):
    import json
    with open(get_log_path().joinpath(f"{name}.json"), "w") as f:
        json.dump(log_dict, f)
