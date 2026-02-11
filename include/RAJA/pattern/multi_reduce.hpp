/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA reduction declarations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_multi_reduce_HPP
#define RAJA_multi_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{

//
// Forward declarations for multi reduction templates.
// Actual classes appear in forall_*.hxx header files.
//
// IMPORTANT: multi reduction policy parameter must be consistent with loop
//            execution policy type.
//
// Also, multiple multi reductions using different reduction operations may be
// combined in a single RAJA forall() construct.
//

/*!
 ******************************************************************************
 *
 * \brief  Min multi reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   Index_ptr bins = ...;
   Real_ptr min_vals = ...;

   MultiReduceMin<multi_reduce_policy, Real_type> my_mins(num_bins, init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_mins[bins[i]].min(data[i]);
   }

   for (size_t bin = 0; bin < num_bins; ++bin) {
      min_vals[bin] = my_mins[bin].get();
   }

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename MULTI_REDUCE_POLICY_T, typename T>
struct MultiReduceMin;

/*!
 ******************************************************************************
 *
 * \brief  Max multi reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   Index_ptr bins = ...;
   Real_ptr max_vals = ...;

   MultiReduceMax<multi_reduce_policy, Real_type> my_maxs(num_bins, init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_maxs[bins[i]].max(data[i]);
   }

   for (size_t bin = 0; bin < num_bins; ++bin) {
      max_vals[bin] = my_maxs[bin].get();
   }

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename MULTI_REDUCE_POLICY_T, typename T>
struct MultiReduceMax;

/*!
 ******************************************************************************
 *
 * \brief  Sum multi reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   Index_ptr bins = ...;
   Real_ptr sum_vals = ...;

   MultiReduceSum<multi_reduce_policy, Real_type> my_sums(num_bins, init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_sums[bins[i]] += (data[i]);
   }

   for (size_t bin = 0; bin < num_bins; ++bin) {
      sum_vals[bin] = my_sums[bin].get();
   }

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename MULTI_REDUCE_POLICY_T, typename T>
struct MultiReduceSum;

/*!
 ******************************************************************************
 *
 * \brief  Bitwise OR multi reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   Index_ptr bins = ...;
   Real_ptr bit_vals = ...;

   MultiReduceBitOr<multi_reduce_policy, Real_type> my_bits(num_bins, init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_bits[bins[i]] |= (data[i]);
   }

   for (size_t bin = 0; bin < num_bins; ++bin) {
      bit_vals[bin] = my_bits[bin].get();
   }

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename MULTI_REDUCE_POLICY_T, typename T>
struct MultiReduceBitOr;


/*!
 ******************************************************************************
 *
 * \brief  Bitwise AND multi reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   Index_ptr bins = ...;
   Real_ptr bit_vals = ...;

   MultiReduceBitAnd<multi_reduce_policy, Real_type> my_bits(num_bins,
 init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_bits[bins[i]] &= (data[i]);
   }

   for (size_t bin = 0; bin < num_bins; ++bin) {
      bit_vals[bin] = my_bits[bin].get();
   }

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename MULTI_REDUCE_POLICY_T, typename T>
struct MultiReduceBitAnd;

}  // namespace RAJA


#endif  // closing endif for header file include guard
