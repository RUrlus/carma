cmake -S . -G Ninja -B build \
    -DCARMA_ENABLE_DEBUG=true \
    -DCARMA_ENABLE_EXTRA_DEBUG=true \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)') \
    -DCARMA_ENV_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    -Dpybind11_ROOT=$(pybind11-config --cmakedir) \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build --target install --parallel 2
