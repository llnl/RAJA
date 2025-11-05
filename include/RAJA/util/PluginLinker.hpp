//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Plugin_Linker_HPP
#define RAJA_Plugin_Linker_HPP

#include "RAJA/util/RuntimePluginLoader.hpp"
#include "RAJA/util/KokkosPluginLoader.hpp"
#include "RAJA/util/CaliperPlugin.hpp"

namespace
{
namespace anonymous_RAJA
{
struct pluginLinker
{
  inline pluginLinker()
  {
    (void)RAJA::util::linkRuntimePluginLoader();
    (void)RAJA::util::linkKokkosPluginLoader();
#if defined(RAJA_ENABLE_CALIPER)
    (void)RAJA::util::linkCaliperPlugin();
#endif
  }
} pluginLinker;
}  // namespace anonymous_RAJA
}  // namespace
#endif
