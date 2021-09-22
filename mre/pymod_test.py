import build.pymod as pymod

def test_runtime_error_exception():
    try:
        pymod.test_runtime_error()
    except RuntimeError as e:
        print(e)
        print('successfully caught error')
