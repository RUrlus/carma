from build.pymod import test_runtime_error

try:
    test_runtime_error()
except RuntimeError as e:
    print(e)
    print('successfully caught error')
