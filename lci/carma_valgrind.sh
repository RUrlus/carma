docker run -it \
    --rm \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e PYTHON_VERSION=3.7 \
    -e CPP=14 \
    -e COMPILER=gcc-7 \
    -e DEBUG=true \
    -e VALGRIND=true \
    -e PYTHON_PREFIX_PATH=/usr/ \
    carma_valgrind:0.1
