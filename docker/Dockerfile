# Docker image to build and test OpenMM-TensorRT plugin

# Start from CentOS 7 with CUDA
ARG PLUGIN_CUDA_VERSION
FROM nvidia/cuda:$PLUGIN_CUDA_VERSION-devel-centos7

# Set up CUDA
# Note: the following has to be repeated as it looses effect after FROM
ARG PLUGIN_CUDA_VERSION
ENV PLUGIN_CUDA_VERSION $PLUGIN_CUDA_VERSION
ARG PLUGIN_CUDA_LABEL
ENV PLUGIN_CUDA_LABEL $PLUGIN_CUDA_LABEL
ENV CUDA_PATH /usr/local/cuda

# Update & install system packages
RUN yum update -y &&\
    yum install -y epel-release &&\
    yum install -y conda \
                   wget \
                   which &&\
    yum clean all

# Install a custom GCC
RUN yum install -y file \
                   gmp-devel \
                   libmpc-devel \
                   make \
                   mpfr-devel \
                   texinfo &&\
    yum clean all
RUN curl https://gcc.gnu.org/pub/gcc/releases/gcc-7.3.0/gcc-7.3.0.tar.gz | tar xzvf - &&\
    cd gcc-7.3.0 &&\
    ./configure --disable-bootstrap \
                --disable-multilib \
                --enable-languages=c,c++ \
                --prefix=/opt/gcc &&\
    make -j 16 &&\
    make install &&\
    cd .. &&\
    rm -r gcc-7.3.0
ENV CC /opt/gcc/bin/gcc
ENV CXX /opt/gcc/bin/g++

# Install TensorRT
# Note: the download URL will work for CUDA 10.0 only
ENV TENSORRT_PACKAGE nv-tensorrt-repo-rhel7-cuda$PLUGIN_CUDA_VERSION-trt7.0.0.11-ga-20191216-1-1.x86_64
ENV TENSORRT_RPM $TENSORRT_PACKAGE.rpm
COPY $TENSORRT_RPM .
RUN yum install -y ./$TENSORRT_RPM &&\
    yum install -y tensorrt &&\
    yum remove -y $TENSORRT_PACKAGE &&\
    yum clean all &&\
    rm -f $TENSORRT_RPM

# Set root password :D
RUN echo root:tensorrt | chpasswd

# Create a user
ARG UID
ARG GID
RUN echo $UID $GID &&\
    groupadd --gid $GID user &&\
    useradd --uid $UID --gid $GID user
USER $UID:$GID

# Create conda enviroment
ENV OPENMM_VERSION 7.4.1.2020.01.06
ENV PYTHON_VERSION 3.7
ENV CONDA_NAME devel
ENV CONDA_CHANNEL_LIST "--channel conda-forge"
RUN conda create $CONDA_CHANNEL_LIST \
                 --name $CONDA_NAME \
                 --yes \
                 anaconda-client \
                 cmake \
                 conda-build \
                 conda-verify \
                 cudatoolkit==$PLUGIN_CUDA_VERSION.* \
                 git \
                 make \
                 nomkl \
                 python==$PYTHON_VERSION \
                 swig &&\
    conda clean --all --yes
RUN echo "conda activate $CONDA_NAME" >> ~/.bashrc

# Build a custom OpenMM
# Note: cannot use the OpennMM packages form Omnia channel because they are
#       built with the old C++ ABI.
# Note: conda-build needs a writable cache directory
RUN echo -e "pkgs_dirs:\n - ~/.conda/pkgs" > ~/.condarc
COPY --chown=$UID:$GID openmm /tmp/openmm
RUN . /etc/profile.d/conda.sh &&\
    conda activate $CONDA_NAME &&\
    conda build $CONDA_CHANNEL_LIST \
                --python $PYTHON_VERSION \
                /tmp/openmm &&\
    conda build purge &&\
    rm -r /tmp/openmm
ENV CONDA_CHANNEL_LIST "--channel file://home/user/.conda/envs/$CONDA_NAME/conda-bld \
                        $CONDA_CHANNEL_LIST"
RUN conda install $CONDA_CHANNEL_LIST \
                  --name $CONDA_NAME \
                  --yes \
                  openmm==$OPENMM_VERSION

# Set up environment
RUN cd && ln -s .conda/envs/devel/conda-bld/linux-64 packages
WORKDIR /home/user/openmm-tensorrt.git
CMD bash