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

#include "RAJA/util/for_each.hpp"
#include "RAJA/util/types.hpp"
#include "camp/number.hpp"
#include "camp/tuple.hpp"
#include "camp/array.hpp"

namespace RAJA
{

// Slice descriptors

// template<typename SliceType>
// RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(IndexType& idx) const {
//     return start + idx;
// }

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

template<typename IndexType = Index_type>
struct NoSlice { 
    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(IndexType& idx) const {
        return idx;
    }
};

template <typename T, size_t N, RAJA::Index_type... Is>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto array_to_tuple_impl(const camp::array<T, N>& arr, camp::idx_seq<Is...>) {
    return camp::make_tuple(arr[Is]...);
}

template <typename T, size_t N>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto array_to_tuple(const camp::array<T, N>& arr) {
    return array_to_tuple_impl(arr, camp::make_idx_seq_t<N>{});
}

template <typename ViewType, typename SliceTypes, typename IndexType = Index_type>
class SubView;

template <typename ViewType, typename IndexType, typename... Slices>
class SubView<ViewType, camp::list<Slices...>, IndexType> {
    ViewType view_;
    camp::tuple<Slices...> slices_;
    std::array<IndexType, sizeof...(Slices)> map_;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_subview_index_map() {
        size_t sub_idx = 0;
        std::array<IndexType, sizeof...(Slices)> map;

        for_each_tuple_index( slices_, 
            [&](auto slice, auto index) { 
                map[index] = decltype(slice)::reduces_dimension ? -1 : sub_idx++;
            });

        return map;
    }

public:

    RAJA_INLINE RAJA_HOST_DEVICE constexpr SubView(ViewType view, Slices... slices)
        : view_(view), slices_(slices...), map_(make_subview_index_map()) { }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr void set_slices(Slices... slices) {
        slices_ = camp::tuple<Slices...>(slices...);
    }

    template<IndexType Index, typename Slice>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr void set_slice(Slice slice) {
        camp::get<Index>(slices_) = slice;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_slices() {
        return slices_;
    }

    template<IndexType Index>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_slice() {
        return camp::get<Index>(slices_);
    }

    template <typename... Idxs>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType operator()(Idxs... idxs) const {
        constexpr size_t nidx = ((Slices::reduces_dimension == false ? 1 : 0) + ...);
        static_assert(sizeof...(idxs) == nidx, "Wrong number of indices for subview");

        camp::array<Index_type, nidx> arr{idxs...};
        camp::array<Index_type, sizeof...(Slices)> parent_indices;

        for_each_tuple_index( slices_,
            [&](auto slice, auto index) { 
                parent_indices[index] = slice.map_index(arr[map_[index]]); 
            });

        return camp::apply(view_, array_to_tuple(parent_indices));
    }
};

template <typename ViewType, typename... Slices>
SubView(ViewType, Slices...) -> SubView<ViewType, camp::list<Slices...>>;

}  // namespace RAJA

#endif