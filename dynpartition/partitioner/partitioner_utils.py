from typing import Union, Tuple, Sequence

import torch


def device_id_to_device_string(device_id):
    assert device_id <= torch.cuda.device_count()
    if device_id == torch.cuda.device_count():
        return 'cpu'
    else:
        return f'cuda:{device_id}'


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
        return tuple(
            t.to(device=device) if t.device != device else t
            for t in tensors
        )
    else:
        if tensors.device != device:
            return tensors.to(device=device)
        else:
            return tensors

