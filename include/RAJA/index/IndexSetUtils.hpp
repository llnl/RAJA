/*!
 ******************************************************************************
 *
 * \file IndexSetUtils.hpp
 *
 * \brief   Header file containing generic RAJA index set and segment utility
 *          method templates.
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

#ifndef RAJA_IndexSetUtils_HPP
#define RAJA_IndexSetUtils_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/policy/sequential.hpp"

namespace RAJA
{

//@{
//!   @name Methods to gather indices of segment or index set into a container.
//!
//!   For each method, the given container must be templated on a data type,
//!   have default and copy ctors, push_back method, and value_type. Is is
//!   assumed that the container data type and segment or index set data type
//!   are compatible in the sense that the index set type can be converted to
//!   the container data type.

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given index set to given container.
 *
 ******************************************************************************
 */
template<typename CONTAINER_T, typename... SEG_TYPES>
RAJA_INLINE void getIndices(CONTAINER_T& con,
                            const TypedIndexSet<SEG_TYPES...>& iset)
{
  CONTAINER_T tcon;
  forall<ExecPolicy<seq_segit, seq_exec>>(
      iset, [&](typename CONTAINER_T::value_type idx) {
        tcon.push_back(idx);
      });
  con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given segment to given container.
 *
 ******************************************************************************
 */
template<typename CONTAINER_T, typename SEGMENT_T>
RAJA_INLINE void getIndices(CONTAINER_T& con, const SEGMENT_T& seg)
{
  CONTAINER_T tcon;
  forall<seq_exec>(seg, [&](typename CONTAINER_T::value_type idx) {
    tcon.push_back(idx);
  });
  con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given index set that satisfy
 *         given conditional to given container.
 *
 ******************************************************************************
 */
template<typename CONTAINER_T, typename... SEG_TYPES, typename CONDITIONAL>
RAJA_INLINE void getIndicesConditional(CONTAINER_T& con,
                                       const TypedIndexSet<SEG_TYPES...>& iset,
                                       CONDITIONAL conditional)
{
  CONTAINER_T tcon;
  forall<ExecPolicy<seq_segit, seq_exec>>(
      iset, [&](typename CONTAINER_T::value_type idx) {
        if (conditional(idx)) tcon.push_back(idx);
      });
  con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given segment that satisfy
 *         given conditional to given container.
 *
 ******************************************************************************
 */
template<typename CONTAINER_T, typename SEGMENT_T, typename CONDITIONAL>
RAJA_INLINE void getIndicesConditional(CONTAINER_T& con,
                                       const SEGMENT_T& seg,
                                       CONDITIONAL conditional)
{
  CONTAINER_T tcon;
  forall<seq_exec>(seg, [&](typename CONTAINER_T::value_type idx) {
    if (conditional(idx)) tcon.push_back(idx);
  });
  con = tcon;
}

//@}

}  // namespace RAJA

#endif  // closing endif for header file include guard
