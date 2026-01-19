###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

##
## Note that we build with 'make -j 16' for newer targets and 'make -j 6' 
## for older targets on GitHub Actions. This is reflected in the 'make' 
## commands below. This seems to work best for throughput.
##

FROM ghcr.io/llnl/radiuss:gcc-12-ubuntu-24.04 AS gcc12
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:gcc-12-ubuntu-24.04 AS gcc12_debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:gcc-13-ubuntu-24.04 AS gcc13
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:gcc-13-ubuntu-24.04 AS gcc13_desul
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On -DRAJA_ENABLE_DESUL_ATOMICS=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:clang-14-ubuntu-22.04 AS clang14_debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug  -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:clang-14-ubuntu-22.04 AS clang14_style
USER root
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN clang-format --version && \
    cmake -DENABLE_CLANGFORMAT=ON ../ && \
    make check

FROM ghcr.io/llnl/radiuss:clang-14-ubuntu-22.04 AS clang14_desul
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On -DRAJA_ENABLE_DESUL_ATOMICS=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure && \
    make clean

## Don't run tests due to failure in RAJA launch tests with new reducer interface.
FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024_0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    make clean"

## Don't run tests due to failure in RAJA launch tests with new reducer interface.
FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024_0_debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    make clean"

FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024_0_sycl
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=Off -DRAJA_ENABLE_SYCL=On -DRAJA_ENABLE_DESUL_ATOMICS=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    make clean"

FROM ghcr.io/llnl/radiuss:hip-6.4.3-ubuntu-24.04 AS rocm6_4_3_desul
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.3/bin/amdclang++ -DROCM_PATH=/opt/rocm-6.4.3 -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=On -DRAJA_ENABLE_DESUL_ATOMICS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off -DBLT_CXX_STD=c++17 .. && \
    make -j 16 &&\
    make clean 
