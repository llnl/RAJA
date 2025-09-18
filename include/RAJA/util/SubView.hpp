/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining the SubView class
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SUBVIEW_HPP
#define RAJA_SUBVIEW_HPP

#include "RAJA/util/types.hpp"
#include "camp/number.hpp"
#include "camp/tuple.hpp"
#include "camp/array.hpp"

namespace RAJA
{

// Slice descriptors
struct RangeSlice { long start, end; };
struct FixedSlice { long idx; };
struct NoSlice { };

// Helper to count non-fixed dimensions at compile time
template <typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr size_t count_nonfixed_dims() {
    return (0 + ... + ((std::is_same_v<Slices, RangeSlice> || 
                        std::is_same_v<Slices, NoSlice>) ? 1 : 0));
}

template <typename T, size_t N, RAJA::Index_type... Is>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto array_to_tuple_impl(const camp::array<T, N>& arr, camp::idx_seq<Is...>) {
    return camp::make_tuple(arr[Is]...);
}

template <typename T, size_t N>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto array_to_tuple(const camp::array<T, N>& arr) {
    return array_to_tuple_impl(arr, camp::make_idx_seq_t<N>{});
}

template <typename ViewType, typename... Slices>
class SubView {
    ViewType view_;
    camp::tuple<Slices...> slices_;

    template <typename IndexType, IndexType... Is>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto map_indices(IndexType* idxs, camp::idx_seq<Is...>) const {
        camp::array<RAJA::Index_type, sizeof...(Is)> parent_indices{};
        RAJA::Index_type idx = 0;

        // For each slice, map subview index to parent index
        (
            (
                parent_indices[Is] = [&] {
                    const auto& s = camp::get<Is>(slices_);
                    if constexpr (std::is_same_v<std::decay_t<decltype(s)>, RangeSlice>) {
                        return s.start + idxs[idx++];
                    } else if constexpr (std::is_same_v<std::decay_t<decltype(s)>, FixedSlice>) {
                        return s.idx;
                    } else {
                        return idxs[idx++];
                    }
                }()
            ), ...
        );

        return parent_indices;
    }

public:
    RAJA_INLINE RAJA_HOST_DEVICE SubView(ViewType view, Slices... slices)
        : view_(view), slices_(slices...) {}

    template <typename... Idxs>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto operator()(Idxs... idxs) const {
        constexpr size_t nidx = count_nonfixed_dims<Slices...>();
        static_assert(sizeof...(idxs) == nidx, "Wrong number of indices for subview");
        camp::array<RAJA::Index_type, nidx> arr{idxs...};
        auto parent_indices = map_indices(arr.data(), camp::make_idx_seq_t<sizeof...(Slices)>());
        return camp::apply(view_, array_to_tuple(parent_indices));
    }
};

}  // namespace RAJA

#endif