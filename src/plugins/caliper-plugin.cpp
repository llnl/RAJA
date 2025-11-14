//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/CaliperPlugin.hpp"

#include <caliper/cali.h>

namespace RAJA
{
namespace util
{

CaliperPlugin::CaliperPlugin()
{
  const std::string varName = "RAJA_CALIPER";
  const char* val           = std::getenv(varName.c_str());
  if (val == nullptr)
  {
    SetRAJACaliperProfiling(false);
    return;
  }

  SetRAJACaliperProfiling(std::stoi(val) != 0 ? true : false);
}

void CaliperPlugin::preLaunch(const RAJA::util::PluginContext& p)
{
  if (!p.kernel_name.empty() && RAJA_caliper_profile == true)
  {
    CALI_MARK_BEGIN(p.kernel_name.c_str());
  }
}

void CaliperPlugin::postLaunch(const RAJA::util::PluginContext& p)
{
  if (!p.kernel_name.empty() && RAJA_caliper_profile == true)
  {
    CALI_MARK_END(p.kernel_name.c_str());
  }
}

void linkCaliperPlugin() {}

}  // namespace util
}  // namespace RAJA

// Statically loading plugin.
static RAJA::util::PluginRegistry::add<RAJA::util::CaliperPlugin> P(
    "Caliper",
    "Enables Caliper Profiling");
