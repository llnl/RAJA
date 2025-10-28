/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining Layout, a N-dimensional index calculator
 *          with permuted stride orderings
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PERMUTEDLAYOUT_HPP
#define RAJA_PERMUTEDLAYOUT_HPP

#include "RAJA/config.hpp"

#include <iostream>

#include "RAJA/index/IndexValue.hpp"

#include "RAJA/util/Layout.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/Permutations.hpp"

namespace RAJA
{

/*!
 * @brief Creates a permuted Layout object.
 *
 * Allows control over the striding order when creating a Layout.
 *
 * For example:
 *
 *     // Create a layout object with the default striding order
 *     // The indices, left to right, have longest stride to stride-1
 *     Layout<3> layout(5,7,11);
 *
 *     // The above is equivalent to:
 *     Layout<3> default_layout = make_permuted_layout({5,7,11},
 * PERM_IJK::value);
 *
 *     // Create a layout object with permuted order
 *     // In this case, J is stride-1, and K has the longest stride
 *     Layout<3> perm_layout = make_permuted_layout({5,7,11}, PERM_KIJ::value);
 *
 *
 * Permutation of up to rank 5 are provided with PERM_I ... PERM_IJKLM
 * aliases in RAJA/util/Permutations.hpp as a convenience.
 *
 * Since permutations are represented using std::array<size_t, Rank> object,
 * arbitrary rank permutations can be used.
 *
 *
 */
template<size_t Rank, typename IdxLin = Index_type>
constexpr RAJA_INLINE RAJA_HOST_DEVICE auto make_permuted_layout(
    std::array<IdxLin, Rank> sizes,
    std::array<camp::idx_t, Rank> permutation) -> Layout<Rank, IdxLin>
{
  std::array<IdxLin, Rank> strides;
  std::array<IdxLin, Rank> folded_strides;
  for (size_t i = 0; i < Rank; ++i)
  {
    // If the size of dimension i is zero, then the stride is zero
    folded_strides[i] = sizes[permutation[i]] ? 1 : 0;
    for (size_t j = i + 1; j < Rank; ++j)
    {
      folded_strides[i] *= sizes[permutation[j]] ? sizes[permutation[j]] : 1;
    }
  }

  for (size_t i = 0; i < Rank; ++i)
  {
    strides[permutation[i]] = folded_strides[i];
  }


  // return Layout<Rank, IdxLin>(sizes, strides);
  auto ret = Layout<Rank, IdxLin>();
  for (size_t i = 0; i < Rank; ++i)
  {
    ret.sizes[i]       = sizes[i];
    ret.strides[i]     = strides[i];
    ret.inv_strides[i] = strides[i] ? strides[i] : 1;
    ret.inv_mods[i]    = sizes[i] ? sizes[i] : 1;
  }
  return ret;
}

template<camp::idx_t... Ints>
using Perm = camp::idx_seq<Ints...>;
template<camp::idx_t N>
using MakePerm = typename camp::make_idx_seq<N>::type;

}  // namespace RAJA

#endif
