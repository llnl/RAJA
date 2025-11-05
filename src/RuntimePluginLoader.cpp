//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/RuntimePluginLoader.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#include <dirent.h>
#endif

RAJA_INLINE
bool isSharedObject(const std::string& filename)
{
  return (filename.size() > 3 &&
          !filename.compare(filename.size() - 3, 3, ".so"));
}

namespace RAJA
{
namespace util
{

// TODO (08/27/25): Add a controller plugin class that enables plugins
// based on environment variables. This would be its own class
//  we would add in the linkCaliperPlugin() bit as well as the
//  RAJA::util::PluginRegistry::add<RAJA::util::ControllerPlugin> call

RuntimePluginLoader::RuntimePluginLoader()
{
  char* env = ::getenv("RAJA_PLUGINS");
  if (nullptr == env)
  {
    return;
  }

  initDirectory(std::string(env));
}

void RuntimePluginLoader::init(const RAJA::util::PluginOptions& p)
{
  initDirectory(p.str);
  for (auto& plugin : plugins)
  {
    plugin->init(p);
  }
}

void RuntimePluginLoader::preCapture(const RAJA::util::PluginContext& p)
{
  for (auto& plugin : plugins)
  {
    plugin->preCapture(p);
  }
}

void RuntimePluginLoader::postCapture(const RAJA::util::PluginContext& p)
{
  for (auto& plugin : plugins)
  {
    plugin->postCapture(p);
  }
}

void RuntimePluginLoader::preLaunch(const RAJA::util::PluginContext& p)
{
  for (auto& plugin : plugins)
  {
    plugin->preLaunch(p);
  }
}

void RuntimePluginLoader::postLaunch(const RAJA::util::PluginContext& p)
{
  for (auto& plugin : plugins)
  {
    plugin->postLaunch(p);
  }
}

void RuntimePluginLoader::finalize()
{
  for (auto& plugin : plugins)
  {
    plugin->finalize();
  }
  plugins.clear();
}

// Initialize plugin from a shared object file specified by 'path'.
void RuntimePluginLoader::initPlugin(const std::string& path)
{
#ifndef _WIN32
  // clear out any existing dlerrors
  RAJA_UNUSED_VAR(dlerror());

  void* plugin = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!plugin)
  {
    printf("[RuntimePluginLoader]: dlopen failed: %s\n", dlerror());
  }

  RuntimePluginLoader::Parent* (*RAJAGetPlugin)() =
      (RuntimePluginLoader::Parent * (*)()) dlsym(plugin, "RAJAGetPlugin");

  if (!RAJAGetPlugin)
  {
    // clear the dlerror
    RAJA_UNUSED_VAR(dlerror());

    // try to find deprecated symbol
    RAJAGetPlugin =
        (RuntimePluginLoader::Parent * (*)()) dlsym(plugin, "getPlugin");
  }

  if (RAJAGetPlugin)
  {
    plugins.push_back(
        std::unique_ptr<RuntimePluginLoader::Parent>(RAJAGetPlugin()));
  }
  else
  {
    printf("[RuntimePluginLoader]: dlsym failed: %s\n", dlerror());
  }
#else
  RAJA_UNUSED_ARG(path);
#endif
}

// Initialize all plugins in a directory specified by 'path'.
void RuntimePluginLoader::initDirectory(const std::string& path)
{
#ifndef _WIN32
  if (isSharedObject(path))
  {
    initPlugin(path);
    return;
  }

  DIR* dir;
  struct dirent* file;

  if ((dir = opendir(path.c_str())) != NULL)
  {
    while ((file = readdir(dir)) != NULL)
    {
      if (isSharedObject(std::string(file->d_name)))
      {
        initPlugin(path + "/" + file->d_name);
      }
    }
    closedir(dir);
  }
  else
  {
    perror("[RuntimePluginLoader]: Could not open plugin directory");
  }
#else
  RAJA_UNUSED_ARG(path);
#endif
}

void linkRuntimePluginLoader() {}

}  // end namespace util
}  // end namespace RAJA

static RAJA::util::PluginRegistry::add<RAJA::util::RuntimePluginLoader> P(
    "RuntimePluginLoader",
    "Dynamically load RAJA plugins.");
