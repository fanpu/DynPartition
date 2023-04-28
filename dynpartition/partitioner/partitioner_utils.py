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


def move_tensor_unless_same_device(tensor, dest_device_id):
    if device_ordinal_to_device_id(tensor.get_device()) == dest_device_id:
        return tensor
    else:
        return tensor.to(device_id_to_device_string(dest_device_id))
