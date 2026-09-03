###############################################################################
 # Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
 # and RAJA project contributors. See the RAJA/LICENSE file for details.
 #
 # SPDX-License-Identifier: (BSD-3-Clause)
 ###############################################################################

 FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc11

 ARG USER=rajadev
 ENV HOME /home/${USER}

 RUN apt-get update && apt-get install -y supervisor

 RUN useradd --create-home --shell /bin/bash ${USER}
 USER ${USER}

 WORKDIR $HOME
 RUN git clone --recursive -b develop https://github.com/llnl/raja

 WORKDIR $HOME/raja/build
 RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
     cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_ARCHITECTURES=70 -DENABLE_OPENMP=On .. 

 WORKDIR /opt/archives
 RUN curl -L https://github.com/gitpod-io/openvscode-server/releases/download/openvscode-server-v1.69.1/openvscode-server-v1.69.1-linux-x64.tar.gz > \
     /opt/archives/openvscode-server-v1.69.1-linux-x64.tar.gz
 RUN tar xzf openvscode-server-v1.69.1-linux-x64.tar.gz && chown -R ${USER}:${USER} openvscode-server-v1.69.1-linux-x64

 USER root
 ADD supervisord.conf /etc/supervisord.conf
 RUN sed -i "s/XXX/${USER}/g" /etc/supervisord.conf

 RUN touch /var/log/openvscode-server.log && chown -R ${USER}:${USER} /var/log/openvscode-server.log

 CMD ["/usr/bin/supervisord"]