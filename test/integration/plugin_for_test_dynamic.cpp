//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#include "RAJA/util/PluginStrategy.hpp"

#include <exception>

class ExceptionPlugin :
  public RAJA::util::PluginStrategy
{
  public:
  void preLaunch(const RAJA::util::PluginContext& RAJA_UNUSED_ARG(p)) override {
    throw std::runtime_error("preLaunch");
  }
};

extern "C" RAJA::util::PluginStrategy *RAJAGetPlugin()
{
  return new ExceptionPlugin;
}
