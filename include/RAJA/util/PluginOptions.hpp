//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Plugin_Options_HPP
#define RAJA_Plugin_Options_HPP

#include <string>

namespace RAJA
{
namespace util
{

struct PluginOptions
{
  PluginOptions(const std::string& newstr) : str(newstr) {};

  std::string str;
};

inline PluginOptions make_options(const std::string& newstr)
{
  return PluginOptions {newstr};
}

}  // namespace util
}  // namespace RAJA

#endif
