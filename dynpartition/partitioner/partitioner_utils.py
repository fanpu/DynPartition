import torch


def device_id_to_device_string(device_id):
    assert device_id <= torch.cuda.device_count()
    if device_id == torch.cuda.device_count():
        return 'cpu'
    else:
        return f'cuda:{device_id}'
