def get_gpu_count() -> int:
    """
    reports how many gpus the system has
    :return:
    """
    import subprocess

    return str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
