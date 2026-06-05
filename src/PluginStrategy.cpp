//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/PluginStrategy.hpp"

RAJA_INSTANTIATE_REGISTRY(PluginRegistry);

namespace RAJA
{
namespace util
{

PluginStrategy::PluginStrategy() = default;

void PluginStrategy::init(const PluginOptions&) {}

void PluginStrategy::preCapture(const PluginContext&) {}

void PluginStrategy::postCapture(const PluginContext&) {}

void PluginStrategy::preLaunch(const PluginContext&) {}

void PluginStrategy::postLaunch(const PluginContext&) {}

void PluginStrategy::finalize() {}

}  // namespace util
}  // namespace RAJA
