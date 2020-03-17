from pytest import fixture


@fixture
def require_gpu():
    import os

    from pyspec.machine.util.gpu import get_gpu_count

    # tests should always run on the last GPU!
    if get_gpu_count() > 1:
        print("configure tests to run on last GPU, since several are available: {}".format(get_gpu_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(get_gpu_count() - 1)
    else:
        print("Default gpu placement")
