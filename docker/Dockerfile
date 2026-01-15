###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
#Builds and installs RAJA using the gcc8 compiler
#

FROM rajaorg/compiler:gcc8
MAINTAINER RAJA Development Team <raja-dev@llnl.gov>

COPY --chown=raja:raja . /home/raja/workspace

WORKDIR /home/raja/workspace

RUN  mkdir build && cd build && cmake -DENABLE_CUDA=OFF ..

RUN cd build && sudo make -j 3 && sudo make install

CMD ["bash"]
