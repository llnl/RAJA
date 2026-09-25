//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_plugin_context_HPP
#define RAJA_plugin_context_HPP

#include <string>

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{
namespace util
{

class KokkosPluginLoader;

struct PluginContext
{
public:
  PluginContext(const Platform p, std::string&& name, resources::Resource res)
      : platform(p),
        kernel_name(std::move(name)),
        resource(std::move(res))
  {}

  Platform platform;
  std::string kernel_name;
  resources::Resource resource;

private:
  mutable uint64_t kID;

  friend class KokkosPluginLoader;
};

template<typename Policy, typename Resource>
PluginContext make_context(std::string&& name, Resource resource)
{
  return PluginContext {detail::get_platform<Policy>::value, std::move(name),
                        std::move(resource)};
}

template<typename Policy>
PluginContext make_context(std::string&& name)
{
  return make_context<Policy>(std::move(name),
                              resources::get_default_resource<Policy>());
}

}  // namespace util
}  // namespace RAJA

#endif
