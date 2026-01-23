//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/KokkosPluginLoader.hpp"

#ifndef _WIN32
#include <dlfcn.h>
#include <dirent.h>
#endif

const uint64_t kokkos_interface_version = 20171029;

RAJA_INLINE
bool isSharedObject(const std::string& filename)
{
  return (filename.size() > 3 &&
          !filename.compare(filename.size() - 3, 3, ".so"));
}

template<typename function>
RAJA_INLINE void getFunction(void* plugin,
                             std::vector<function>& functions,
                             const char* fname)
{
#ifndef _WIN32
  function func = (function)dlsym(plugin, fname);
  if (func)
    functions.push_back(func);
  else
    printf("[KokkosPluginLoader]: dlsym failed: %s\n", dlerror());
#else
  RAJA_UNUSED_ARG(plugin);
  RAJA_UNUSED_ARG(functions);
  RAJA_UNUSED_ARG(fname);
#endif
}

namespace RAJA
{
namespace util
{

KokkosPluginLoader::KokkosPluginLoader()
{
  char* env = getenv("KOKKOS_PLUGINS");
  if (env == nullptr)
  {
    return;
  }
  initDirectory(std::string(env));

  for (auto& func : init_functions)
  {
    func(0, kokkos_interface_version, 0, nullptr);
  }
}

void KokkosPluginLoader::preLaunch(const RAJA::util::PluginContext& p)
{
  for (auto& func : pre_functions)
  {
    func("", 0, &(p.kID));
  }
}

void KokkosPluginLoader::postLaunch(const RAJA::util::PluginContext& p)
{
  for (auto& func : post_functions)
  {
    func(p.kID);
  }
}

void KokkosPluginLoader::finalize()
{
  for (auto& func : finalize_functions)
  {
    func();
  }
  init_functions.clear();
  pre_functions.clear();
  post_functions.clear();
  finalize_functions.clear();
}

// Initialize plugin from a shared object file specified by 'path'.
void KokkosPluginLoader::initPlugin(const std::string& path)
{
#ifndef _WIN32
  void* plugin = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!plugin)
  {
    printf("[KokkosPluginLoader]: dlopen failed: %s\n", dlerror());
  }

  // Getting and storing supported kokkos functions.
  getFunction<init_function>(plugin, init_functions, "kokkosp_init_library");

  getFunction<pre_function>(plugin, pre_functions,
                            "kokkosp_begin_parallel_for");

  getFunction<post_function>(plugin, post_functions,
                             "kokkosp_end_parallel_for");

  getFunction<finalize_function>(plugin, finalize_functions,
                                 "kokkosp_finalize_library");
#else
  RAJA_UNUSED_ARG(path);
#endif
}

// Initialize all plugins in a directory specified by 'path'.
void KokkosPluginLoader::initDirectory(const std::string& path)
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
    perror("[KokkosPluginLoader]: Could not open plugin directory");
  }
#else
  RAJA_UNUSED_ARG(path);
#endif
}

void linkKokkosPluginLoader() {}

}  // end namespace util
}  // end namespace RAJA

static RAJA::util::PluginRegistry::add<RAJA::util::KokkosPluginLoader> P(
    "KokkosPluginLoader",
    "Dynamically load plugins ported from the Kokkos library.");
