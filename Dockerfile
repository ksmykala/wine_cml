FROM dvcorg/cml-py3

# FROM ubuntu:20.04 AS development

RUN apt-get update;\
    apt-get install -y python3.8 python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

# docker build -t gpurunner .
# docker run --name gpurunner -d --gpus all -e RUNNER_IDLE_TIEMOUT=1000 -e RUNNER_LABELS=cml,gpu -e RUNNER_REPO="https://github.com/ksmykala/wine_cml" -e repo_token=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 gpurunner