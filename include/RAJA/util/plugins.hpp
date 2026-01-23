//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugins_HPP
#define RAJA_plugins_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/PluginContext.hpp"
#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/PluginStrategy.hpp"
#if defined(RAJA_ENABLE_RUNTIME_PLUGINS)
#include "RAJA/util/RuntimePluginLoader.hpp"
#include "RAJA/util/KokkosPluginLoader.hpp"
#endif

namespace RAJA
{
namespace util
{

template<typename T>
RAJA_INLINE auto trigger_updates_before(T&& item) ->
    typename std::remove_reference<T>::type
{
  return item;
}

RAJA_INLINE
void callPreCapturePlugins(const PluginContext& p)
{
  for (auto plugin = PluginRegistry::begin(); plugin != PluginRegistry::end();
       ++plugin)
  {
    (*plugin).get()->preCapture(p);
  }
}

RAJA_INLINE
void callPostCapturePlugins(const PluginContext& p)
{
  for (auto plugin = PluginRegistry::begin(); plugin != PluginRegistry::end();
       ++plugin)
  {
    (*plugin).get()->postCapture(p);
  }
}

RAJA_INLINE
void callPreLaunchPlugins(const PluginContext& p)
{
  for (auto plugin = PluginRegistry::begin(); plugin != PluginRegistry::end();
       ++plugin)
  {
    (*plugin).get()->preLaunch(p);
  }
}

RAJA_INLINE
void callPostLaunchPlugins(const PluginContext& p)
{
  for (auto plugin = PluginRegistry::begin(); plugin != PluginRegistry::end();
       ++plugin)
  {
    (*plugin).get()->postLaunch(p);
  }
}

RAJA_INLINE
void callInitPlugins(const PluginOptions p)
{
  for (auto plugin = PluginRegistry::begin(); plugin != PluginRegistry::end();
       ++plugin)
  {
    (*plugin).get()->init(p);
  }
}

RAJA_INLINE
void init_plugins(const std::string& path)
{
  callInitPlugins(make_options(path));
}

RAJA_INLINE
void init_plugins() { callInitPlugins(make_options("")); }

RAJA_INLINE
void finalize_plugins()
{
  for (auto plugin = PluginRegistry::begin(); plugin != PluginRegistry::end();
       ++plugin)
  {
    (*plugin).get()->finalize();
  }
}

}  // namespace util
}  // namespace RAJA

#endif
