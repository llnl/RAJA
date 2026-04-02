//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_region_sequential_HPP
#define RAJA_region_sequential_HPP

namespace RAJA
{
namespace policy
{
namespace sequential
{

/*!
 * \brief RAJA::region implementation for sequential
 *
 * Generates sequential region
 *
 * \code
 *
 * RAJA::region<seq_region>([=](){
 *
 *  // region body - may contain multiple loops
 *
 *  });
 *
 * \endcode
 *
 * \tparam Policy region policy
 *
 */

template<typename Func>
RAJA_INLINE void region_impl(const seq_region&, Func&& body)
{
  body();
}

}  // namespace sequential

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for header file include guard
