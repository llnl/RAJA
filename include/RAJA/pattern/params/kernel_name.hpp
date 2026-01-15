//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_KERNEL_NAME_HPP
#define RAJA_KERNEL_NAME_HPP

#include "RAJA/pattern/params/params_base.hpp"

namespace RAJA
{
namespace detail
{

struct Name : public RAJA::expt::detail::ForallParamBase
{
  RAJA_HOST_DEVICE Name() {}

  explicit Name(const char* name_in) : name(name_in) {}

  const char* name;
};

}  // namespace detail

inline auto Name(const char* n) { return detail::Name(n); }


}  //  namespace RAJA


#endif  // KERNEL_NAME_HPP
