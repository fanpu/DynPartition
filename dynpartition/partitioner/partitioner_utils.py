from typing import Union, Tuple, Sequence, List

import torch


_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
_devices.append('cpu')
ALL_DEVICES: List[str] = _devices


def device_id_to_device_string(device_id):
    assert device_id <= torch.cuda.device_count()
    if device_id == torch.cuda.device_count():
        return 'cpu'
    else:
        return f'cuda:{device_id}'


def device_id_to_device(device_id):
    return torch.device(device_id_to_device_string(device_id))


def device_ordinal_to_device_id(device_ordinal):
    if device_ordinal == -1:
        return torch.cuda.device_count()
    return device_ordinal


def tensors_to_device(
        device: Union[str, torch.device],
        tensors: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(tensors, Sequence):
        # noinspection PyTypeChecker
        return tuple(t.to(device=device) for t in tensors)
    else:
        return tensors.to(device=device)


def allocation_summary(device_allocations):
    inv_list = {v: k for k, v in device_allocations.items()}.items()
    print("Device allocations", inv_list)
