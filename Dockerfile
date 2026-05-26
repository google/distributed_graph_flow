FROM quay.io/pypa/manylinux_2_28:latest

RUN yum install -y java-11-openjdk-devel zip unzip gcc-c++ graphviz gdb

RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

WORKDIR /work

ENTRYPOINT ["/bin/bash", "-c"]
