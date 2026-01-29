//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Kokkos_Plugin_Loader_HPP
#define RAJA_Kokkos_Plugin_Loader_HPP

#include <memory>
#include <vector>

#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/PluginStrategy.hpp"

namespace RAJA
{
namespace util
{

class KokkosPluginLoader : public ::RAJA::util::PluginStrategy
{
public:
  using Parent = ::RAJA::util::PluginStrategy;
  typedef void (*init_function)(const int,
                                const uint64_t,
                                const uint32_t,
                                void*);
  typedef void (*pre_function)(const char*, const uint32_t, uint64_t*);
  typedef void (*post_function)(uint64_t);
  typedef void (*finalize_function)();

  KokkosPluginLoader();

  void preLaunch(const RAJA::util::PluginContext& p) override;

  void postLaunch(const RAJA::util::PluginContext& p) override;

  void finalize() override;

private:
  void initPlugin(const std::string& path);

  void initDirectory(const std::string& path);

  std::vector<init_function> init_functions;
  std::vector<pre_function> pre_functions;
  std::vector<post_function> post_functions;
  std::vector<finalize_function> finalize_functions;

};  // end KokkosPluginLoader class

void linkKokkosPluginLoader();

}  // end namespace util
}  // end namespace RAJA

#endif
