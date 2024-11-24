ARG CUDA_VERSION=11.3.1
ARG OS_VERSION=20.04

FROM nvcr.io/nvidia/cudagl:${CUDA_VERSION}-devel-ubuntu${OS_VERSION}


# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1
# Environment and system
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONUNBUFFERED 1

# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# RUN apt-get clean
# Change the sources list to South Korea mirrors (As it is slow to download from the original source)
RUN sed -i 's|http://archive.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list
RUN sed -i 's|http://security.ubuntu|http://mirror.kakao|g' /etc/apt/sources.list


RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN apt-get update && apt-get install -y software-properties-common
# Add Python ppa
# RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y libcudnn8=8.2.0.53-1+cuda11.3 \
    libcudnn8-dev=8.2.0.53-1+cuda11.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential \
    tensorrt

RUN apt-get update && apt-get install -y \
  apt ffmpeg git vim wget unrar nano \
  python3.8 python3.8-dev python3-pip  python3.8-distutils \
  && apt-get clean \
  &&  ln -s /usr/bin/python3.8 /usr/local/bin/python &&\
      ln -s /usr/bin/python3.8 /usr/local/bin/python3


# software rendering
# window rendering
RUN apt-get update && apt-get install -y \
  apt cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev \
  libgl1-mesa-glx libosmesa6 patchelf \
  libglfw3 libglew-dev \ 
  && apt-get clean

# MuJoCo
ENV MUJOCO_PY_MUJOCO_PATH /.mujoco/mujoco200
RUN mkdir -p /.mujoco
# download MuJoCo 2.1.0 for mujoco-py
RUN cd /.mujoco &&\
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210_linux.tar.gz &&\
    tar -xvzf mujoco210_linux.tar.gz -C /.mujoco/ &&\
    rm mujoco210_linux.tar.gz
# add MuJoCo 2.1.0 to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/.mujoco/mujoco210/bin

# download MuJoCo 2.1.1 for dm_control
RUN cd /.mujoco &&\
    wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O mujoco211_linux.tar.gz &&\
    tar -xvzf mujoco211_linux.tar.gz -C /.mujoco/ &&\
    rm mujoco211_linux.tar.gz

# add MuJoCo 2.1.1 to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/.mujoco/mujoco211/bin

# download MuJoCo 2.0.0 for dm_control

RUN cd /.mujoco &&\
    wget -q --show-progress https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip &&\
    unzip mujoco.zip -d /.mujoco &&\
    cp -r /.mujoco/mujoco200_linux /.mujoco/mujoco200 &&\
    rm mujoco.zip


# add MuJoCo 2.0.0 to LD_LIBRARY_PATH
COPY scripts_docker/mjkey.txt /.mujoco/mjkey.txt
ENV MUJOCO_PY_MJKEY_PATH /.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/.mujoco/mujoco200/bin

# for GPU rendering
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/nvidia


RUN pip install --upgrade pip
RUN pip install setuptools>=41.0.0

# Install torch
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip install --no-cache-dir numpy matplotlib pillow h5py==2.10.0 scikit-image funcsigs \
                               opencv-python moviepy pandas requests scipy absl-py ruamel.yaml\
                               ipdb colorlog tqdm slack_sdk opencv-python imageio imageio-ffmpeg

RUN pip install --no-cache-dir hydra-core wandb mpi4py mujoco-py gdown mujoco\
                               gym==0.23.1 setuptools==57.5.0

RUN pip install --no-cache-dir dm_control==0.0.364896371 git+https://github.com/denisyarats/dmc2gym.git

RUN apt-get update && apt-get install -y \
  freeglut3-dev \
  xvfb \
  xserver-xephyr \
  python-opengl \
  python3-opencv \
  && apt-get clean


RUN pip install torchsummary
RUN pip install plotly tensorboardX
# RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
RUN pip install pudb einops
RUN pip install jupyterlab notebook scikit-learn
ENV PYTHONBREAKPOINT "pudb.set_trace"

RUN pip install numpy==1.19.5 \
                absl-py==0.13.0 \
                pyparsing==2.4.7 \
                jupyterlab==3.0.14 \
                scikit-image==0.18.1 \
                termcolor==1.1.0 \
                dm-control==0.0.364896371 \
                tb-nightly==2.10.0a20220724 \
                imageio==2.9.0 \
                imageio-ffmpeg==0.4.4 \
                hydra-core==1.1.0 \
                hydra-submitit-launcher==1.1.5 \
                pandas==1.3.0 \
                ipdb==0.13.9 \
                yapf==0.31.0 \
                # mujoco_py==2.0.2.13 \
                sklearn==0.0 \
                matplotlib==3.4.2 \
                opencv-python \
                wandb==0.15.4 \
                moviepy==1.0.3 \
                git+https://github.com/rlworkgroup/metaworld.git@18118a28c06893da0f363786696cc792457b062b#egg=metaworld \
                pyglet==1.5.24 \
                imagehash==4.3.1 \
                hexhamming==2.2.3

# https://github.com/openai/mujoco-py/issues/773
RUN pip install cython==0.29.36
RUN pip install mujoco_py==2.0.2.13 --no-build-isolation

# https://github.com/pypa/setuptools/issues/3301
RUN pip install setuptools==59.8.0
RUN pip install scikit-image

COPY drstrategy_envs/ /drstrategy_envs/
RUN chown -R 1000:root /drstrategy_envs/ && chmod -R 775 /drstrategy_envs
WORKDIR /drstrategy_envs
RUN pip install --no-cache-dir -e .
WORKDIR /drstrategy_envs/drstrategy_envs/memory-maze
RUN pip install --no-cache-dir -e .

WORKDIR /

COPY scripts_docker/mjkey.txt /root/.mujoco/mjkey.txt
COPY scripts_docker/arial.ttf /tmp/arial.ttf
WORKDIR /root/

ENV MJLIB_PATH /.mujoco/mujoco200_linux/bin/libmujoco200.so
ENV MJKEY_PATH /.mujoco/mjkey.txt

RUN chmod -R 777 /usr/local/lib/python3.8/dist-packages/mujoco_py/generated/

ARG USER_ID
ARG GROUP_ID

RUN groupadd -g $GROUP_ID usergroup && \
    useradd -l -u $USER_ID -g usergroup USERNAME && \
    install -d -m 0755 -o USERNAME -g usergroup /home/USERNAME

USER USERNAME

RUN git config --global --add safe.directory '*'
