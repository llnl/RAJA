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

#ifndef RAJA_reduce_HPP
#define RAJA_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{

///
/// Macros for type agnostic reduction operations.
///
#define RAJA_MIN(a, b) (((b) < (a)) ? (b) : (a))
///
#define RAJA_MAX(a, b) (((b) > (a)) ? (b) : (a))

///
/// Macros to support unstructured minmaxloc operations
#define RAJA_MINLOC_UNSTRUCTURED(set_val, set_idx, a_val, a_idx, b_val, b_idx) \
  set_idx = ((b_val) < (a_val) ? (b_idx) : (a_idx));                           \
  set_val = ((b_val) < (a_val) ? (b_val) : (a_val));
///
#define RAJA_MAXLOC_UNSTRUCTURED(set_val, set_idx, a_val, a_idx, b_val, b_idx) \
  set_idx = ((b_val) > (a_val) ? (b_idx) : (a_idx));                           \
  set_val = ((b_val) > (a_val) ? (b_val) : (a_val));

//
// Forward declarations for reduction templates.
// Actual classes appear in forall_*.hxx header files.
//
// IMPORTANT: reduction policy parameter must be consistent with loop
//            execution policy type.
//
// Also, mutliple reductions using different reduction operations may be
// combined in a single RAJA forall() construct.
//

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMin<reduce_policy, Real_type> my_min(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_min.min(data[i]);
   }

   Real_type minval = my_min.get();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T>
class ReduceMin;

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMinLoc<reduce_policy, Real_type> my_min(init_val, -1);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_min.minloc(data[i], i);
   }

   Real_type minval = my_min.get();
   Index_type minloc = my_min.getLoc();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T, typename IndexType = Index_type>
class ReduceMinLoc;

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMax<reduce_policy, Real_type> my_max(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_max.max(data[i]);
   }

   Real_type maxval = my_max.get();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T>
class ReduceMax;

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceMaxLoc<reduce_policy, Real_type> my_max(init_val, -1);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_max.maxloc(data[i], i);
   }

   Real_type maxval = my_max.get();
   Index_type maxloc = my_max.getLoc();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T, typename IndexType = Index_type>
class ReduceMaxLoc;

/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceSum<reduce_policy, Real_type> my_sum(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_sum += data[i];
   }

   Real_type sum = my_sum.get();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T>
class ReduceSum;

/*!
 ******************************************************************************
 *
 * \brief  Bitwise OR reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceBitOr<reduce_policy, Real_type> my_bits(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_bits |= data[i];
   }

   Real_type finbits = my_bits.get();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T>
class ReduceBitOr;


/*!
 ******************************************************************************
 *
 * \brief  Bitwise AND reducer class template.
 *
 * Usage example:
 *
 * \verbatim

   Real_ptr data = ...;
   ReduceBitAnd<reduce_policy, Real_type> my_bits(init_val);

   forall<exec_policy>( ..., [=] (Index_type i) {
      my_bits &= data[i];
   }

   Real_type finbits = my_bits.get();

 * \endverbatim
 *
 ******************************************************************************
 */
template<typename REDUCE_POLICY_T, typename T>
class ReduceBitAnd;
}  // namespace RAJA


#endif  // closing endif for header file include guard
