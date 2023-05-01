import timeit

import torch

from dynpartition.get_dir import save_log_json
from dynpartition.partitioner.utils import ALL_DEVICES


def test_data_transfer_speeds():
    devices = ALL_DEVICES
    logs = {
        "devices": {},
        "data_transfer_speeds": {},
    }
    speeds = logs["data_transfer_speeds"]

    for i_device in devices:
        for j_device in devices:
            if i_device == j_device:
                continue

            data = torch.randn(1000, 1000, device=i_device, requires_grad=False)

            def transfer():
                new_data = data.to(j_device)
                # print(f"Transfer from {data.device} to {new_data.device}")

            # transfer()
            # transfer()
            run_times = timeit.repeat(
                transfer,
                repeat=1000,
                number=1,
            )
            speeds[f"{i_device}_{j_device}"] = run_times

    for i_device in devices:
        if i_device == "cpu":
            logs["devices"][i_device] = "cpu"
            continue

        logs["devices"][i_device] = torch.cuda.get_device_name(i_device)

    save_log_json(logs, name=f"data_transfer_speeds_{len(devices)}")


if __name__ == '__main__':
    test_data_transfer_speeds()
