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
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "camp/helpers.hpp"
#include "camp/number/number.hpp"
#include "camp/tuple.hpp"
#include "camp/array.hpp"
#include "camp/type_traits/is_same.hpp"

namespace RAJA
{


template<typename IndexType = Index_type>
struct RangeSlice {
    IndexType start_, end_; 

    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(const IndexType& idx) const {
        return start_ + idx;
    }

    template<IndexType DIM, typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const {
        return (end_ - start_ + 1);
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return 0;
    }
};

template<typename IndexType = Index_type>
struct FixedSlice { 
    IndexType idx_; 

    static constexpr bool reduces_dimension = true;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(const IndexType&) const {
        return idx_;
    }

    template<IndexType DIM, typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const {
        return 1;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return 0;
    }

};

template<typename IndexType = Index_type>
struct NoSlice { 
    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(const IndexType& idx) const {
        return idx;
    }

    template<IndexType DIM, typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType& layout) const {
        return layout.template get_dim_size<DIM>();
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return 0;
    }
};

template<typename IndexType = Index_type>
struct StridedSlice { 
    IndexType start_, end_, stride_;

    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(const IndexType& idx) const {
        return idx;
    }

    template<IndexType DIM, typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const {
        return (end_ - start_ + 1) / stride_;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return stride_;
    }
};

template<typename IndexType, typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_subview_index_map() {
    IndexType sub_idx = 0;
    IndexType i = 0;
    camp::array<IndexType, sizeof...(Slices)> map = {};
    ((map[i++] = (Slices::reduces_dimension ? -1 : sub_idx++)), ...);
    return map;
}

template <typename ViewType, typename SliceTypes, typename IndexType = Index_type>
class SubView;

template <typename ViewType, typename IndexType, typename... Slices>
class SubView<ViewType, camp::list<Slices...>, IndexType> {
    ViewType view_;
    camp::tuple<Slices...> slices_;
    static inline constexpr size_t num_slices_ = sizeof...(Slices);
    static inline constexpr IndexType num_sub_indices_ = ((Slices::reduces_dimension == false ? 1 : 0) + ...);
    static inline constexpr camp::array<IndexType, num_slices_> map_ = make_subview_index_map<IndexType, Slices...>();

public:

    RAJA_INLINE RAJA_HOST_DEVICE constexpr SubView(ViewType view, Slices... slices)
        : view_(view), slices_(slices...) { }

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

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size() {

        IndexType prod_dims = 1;
        for_each_tuple_index( slices_,
            [&](auto slice, auto index) {
                const IndexType dim_size = slice.template size<index>(view_.get_layout());
                prod_dims *= (dim_size == 0) ? 1 : dim_size;
            });

        return prod_dims;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size_noproj() {

        IndexType prod_dims = 1;
        for_each_tuple_index( slices_,
            [&](auto slice, auto index) {
                prod_dims *= slice.template size<index>(view_.get_layout());
            });

        return prod_dims;
    }

    template<IndexType DIM> 
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_size() {
        return camp::get<DIM>(slices_).template size<DIM>(view_.get_layout());
    }

    template<IndexType DIM> 
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_stride() {
        return camp::get<DIM>(slices_).template stride<DIM>();
    }

    template<IndexType DIM> 
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_layout() {
        return view_.get_layout();
    }

    template<IndexType DIM> 
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_data() {
        return view_.get_data();
    }

    template <typename... Idxs>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType operator()(Idxs... idxs) const {
        static_assert(sizeof...(idxs) == num_sub_indices_, "Wrong number of indices for subview");

        camp::array<IndexType, num_sub_indices_> arr{idxs...};
        camp::array<IndexType, num_slices_> parent_indices;

        for_each_tuple_index( slices_,
            [&](auto slice, auto index) {
                if (map_[index] >= 0) {
                    parent_indices[index] = slice.map_index(arr[map_[index]]); 
                } else {
                    // map_index will not need index values for dimension-reducing slices
                    // so we pass a "dummy" value.
                    constexpr IndexType dummy_value = -1;
                    parent_indices[index] = slice.map_index(dummy_value); 
                }
            });

        return camp::apply(view_, parent_indices);
    }
};

template <typename ViewType, typename... Slices>
SubView(ViewType, Slices...) -> SubView<ViewType, camp::list<Slices...>>;

}  // namespace RAJA

#endif