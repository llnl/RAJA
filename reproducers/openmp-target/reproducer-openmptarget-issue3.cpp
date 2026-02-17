//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

/*
 *  Reproducer for compile error when using random number generators
 *
 * Compile lines that error
 *
 * /usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/bin/clang++ -std=c++14 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o reproducer-openmptarget-issue3.cpp.o -c reproducer-openmptarget-issue3.cpp
 *
 * /usr/tce/packages/clang/clang-10.0.1/bin/clang++ -std=c++14 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -o reproducer-openmptarget-issue3.cpp.o -c reproducer-openmptarget-issue3.cpp
 */

#include <random>

int main(int, char **)
{
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  double dval = dist(gen);

  return (dval > 1.0 || dval < 0.0) ? 1 : 0;
}
