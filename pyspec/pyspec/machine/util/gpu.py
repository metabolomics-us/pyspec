from tensorflow.python.client import device_lib


def get_available_gpus():
    """
    return all gpus we know about
    :return:
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_gpu_count() -> int:
    """
    reports how many gpus the system has
    :return:
    """
    gpus = get_available_gpus()
    return len(gpus)
