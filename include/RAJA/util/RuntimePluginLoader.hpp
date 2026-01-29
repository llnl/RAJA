//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Runtime_Plugin_Loader_HPP
#define RAJA_Runtime_Plugin_Loader_HPP

#include <memory>
#include <vector>

#include "RAJA/util/PluginOptions.hpp"
#include "RAJA/util/PluginStrategy.hpp"

namespace RAJA
{
namespace util
{

class RuntimePluginLoader : public RAJA::util::PluginStrategy
{
  using Parent = RAJA::util::PluginStrategy;

public:
  RuntimePluginLoader();

  void init(const RAJA::util::PluginOptions& p) override;

  void preCapture(const RAJA::util::PluginContext& p) override;

  void postCapture(const RAJA::util::PluginContext& p) override;

  void preLaunch(const RAJA::util::PluginContext& p) override;

  void postLaunch(const RAJA::util::PluginContext& p) override;

  void finalize() override;

private:
  void initPlugin(const std::string& path);

  void initDirectory(const std::string& path);

  std::vector<std::unique_ptr<Parent>> plugins;

};  // end RuntimePluginLoader class

void linkRuntimePluginLoader();

}  // end namespace util
}  // end namespace RAJA

#endif
