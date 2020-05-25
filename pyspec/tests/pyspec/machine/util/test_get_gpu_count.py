from pyspec.machine.util.gpu import get_gpu_count


def test_get_gpu_count():
    count = get_gpu_count()
    assert count > 0
