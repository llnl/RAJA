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

template<typename IndexType = Index_type>
struct RangeSlice {
    IndexType start, end; 

    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(IndexType& idx) const {
        return start + idx;
    }
};

template<typename IndexType = Index_type>
struct FixedSlice { 
    IndexType idx; 

    static constexpr bool reduces_dimension = true;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(IndexType&) const {
        return idx;
    }
};

struct NoSlice { 
    static constexpr bool reduces_dimension = false;

    template<typename IndexType = Index_type>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(IndexType& idx) const {
        return idx;
    }
};

// Helper to count non-fixed dimensions at compile time
template <typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr size_t count_nonfixed_dims() {
    return (!Slices::reduces_dimension + ...);
}

template <typename T, size_t N, RAJA::Index_type... Is>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto array_to_tuple_impl(const camp::array<T, N>& arr, camp::idx_seq<Is...>) {
    return camp::make_tuple(arr[Is]...);
}

template <typename T, size_t N>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto array_to_tuple(const camp::array<T, N>& arr) {
    return array_to_tuple_impl(arr, camp::make_idx_seq_t<N>{});
}

template <typename ViewType, typename IndexType = Index_type, typename... Slices>
class SubView {
    ViewType view_;
    camp::tuple<Slices...> slices_;
    std::array<IndexType, sizeof...(Slices)> map_;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr void make_subview_index_map() {
        size_t sub_idx = 0;
        size_t i = 0;
        ((map_[i++] = (Slices::reduces_dimension ? -1 : sub_idx++)), ...);
    }

    template<IndexType I>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto map_subview_idx_to_parent(IndexType* idxs) const {
        return camp::get<I>(slices_).map_index(idxs[map_[I]]);
    }

    template <IndexType... Is>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto map_indices(IndexType* idxs, camp::idx_seq<Is...>) const {
        // For each slice, map subview index to parent index
        return camp::array{(map_subview_idx_to_parent<Is>(idxs))...};
    }

public:

    RAJA_INLINE RAJA_HOST_DEVICE constexpr SubView(ViewType view, Slices... slices)
        : view_(view), slices_(slices...) { make_subview_index_map(); }

    template <typename... Idxs>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType operator()(Idxs... idxs) const {
        constexpr size_t nidx = count_nonfixed_dims<Slices...>();
        static_assert(sizeof...(idxs) == nidx, "Wrong number of indices for subview");

        camp::array<RAJA::Index_type, nidx> arr{idxs...};
        auto parent_indices = map_indices(arr.data(), camp::make_idx_seq_t<sizeof...(Slices)>());

        return camp::apply(view_, array_to_tuple(parent_indices));
    }
};

}  // namespace RAJA

#endif