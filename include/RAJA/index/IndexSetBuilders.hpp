/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for various index set builder methods.
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

#ifndef RAJA_IndexSetBuilders_HPP
#define RAJA_IndexSetBuilders_HPP

#include "RAJA/config.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/util/types.hpp"

#include "camp/resource.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief Generate an index set with aligned Range segments and List segments,
 *        as needed, from given array of indices.
 *
 *        Routine does no error-checking on argements and assumes
 *        RAJA::Index_type array contains valid indices.
 *
 *  \param iset reference to index set generated with aligned range segments
 *         and list segments. Method assumes index set is empty (no segments).
 *  \param work_res camp resource object that identifies the memory space in
 *         which list segment index data will live (passed to list segment
 *         ctor).
 *  \param indices_in pointer to start of input array of indices.
 *  \param length size of input index array.
 *  \param range_min_length min length of any range segment in index set
 *  \param range_align "alignment" value for range segments in index set.
 *         Starting index each range segment will be a multiple of this value.
 *
 ******************************************************************************
 */
void RAJASHAREDDLL_API buildIndexSetAligned(
    RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment>& iset,
    camp::resources::Resource work_res,
    const RAJA::Index_type* const indices_in,
    RAJA::Index_type length,
    RAJA::Index_type range_min_length,
    RAJA::Index_type range_align);


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// The following methods build "lock-free" index sets.
//
// Lock-free indexsets are designed to be used with coarse-grained OpenMP
// iteration policies.  The "lock-free" part here assumes interactions among
// the cell-complex associated with the space being partitioned are "tightly
// bound".
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/*!
 ******************************************************************************
 *
 * \brief Generate a lock-free "block" index set (planar division) containing
 *        range segments.
 *
 *        The method chunks a fastDim x midDim x slowDim mesh into blocks that
 *        can be dependency-scheduled, removing need for lock constructs.
 *
 *  \param iset reference to index set generated with range segments.
 *         Method assumes index set is empty (no segments).
 *  \param fastDim "fast" block dimension (see above).
 *  \param midDim  "mid" block dimension (see above).
 *  \param slowDim "slow" block dimension (see above).
 *
 ******************************************************************************
 */
void buildLockFreeBlockIndexset(RAJA::TypedIndexSet<RAJA::RangeSegment>& iset,
                                int fastDim,
                                int midDim,
                                int slowDim);

/*!
 ******************************************************************************
 *
 * \brief Generate a lock-free "color" index set containing range and list
 *        segments.
 *
 *        TThe domain-set is colored based on connectivity to the range-set.
 *        All elements in each segment are independent, and no two segments
 *        can be executed in parallel.
 *
 * \param iset reference to index set generated. Method assumes index set
 *        is empty (no segments).
 * \param work_res camp resource object that identifies the memory space in
 *         which list segment index data will live (passed to list segment
 *         ctor).
 *
 ******************************************************************************
 */
void buildLockFreeColorIndexset(
    RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment>& iset,
    camp::resources::Resource work_res,
    RAJA::Index_type const* domainToRange,
    int numEntity,
    int numRangePerDomain,
    int numEntityRange,
    RAJA::Index_type* elemPermutation  = nullptr,
    RAJA::Index_type* ielemPermutation = nullptr);

}  // namespace RAJA

#endif  // closing endif for header file include guard
