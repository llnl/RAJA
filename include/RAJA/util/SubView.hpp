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
#include "camp/tuple.hpp"
#include "camp/array.hpp"

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
        return 1;
    }
};

template<typename IndexType = Index_type>
struct RangeStartSlice {
    IndexType start_; 

    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(const IndexType& idx) const {
        return start_ + idx;
    }

    template<IndexType DIM, typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType& layout) const {
        return (layout.size() - start_);
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return 1;
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
        return 1;
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
        return 1;
    }
};

template<typename IndexType = Index_type>
struct StridedSlice { 
    IndexType start_, end_, stride_;

    static constexpr bool reduces_dimension = false;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(const IndexType& idx) const {
        return start_ + stride_ * idx;
    }

    template<IndexType DIM, typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const {
        return (end_ - start_) / stride_ + 1;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return stride_;
    }
};

template<typename IndexType, typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_slice_to_parent_index_map() {
    IndexType sub_idx = 0;
    IndexType i = 0;
    camp::array<IndexType, sizeof...(Slices)> map = {};
    ((map[i++] = (Slices::reduces_dimension ? -1 : sub_idx++)), ...);
    return map;
}

template<typename IndexType, size_t n_parent_dims, typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_parent_to_slice_index_map() {
    IndexType sub_idx = 0;
    IndexType i = 0;
    camp::array<IndexType, n_parent_dims> map = {};

    auto process_slice = [&](auto slice_type) constexpr {
        if constexpr (!decltype(slice_type)::reduces_dimension) {
            map[sub_idx++] = i++;
        } else {
            i++;
        }
    };

    (process_slice(Slices{}), ...);

    return map;
}

template <typename LayoutType, typename SliceTypes, typename IndexType = Index_type>
struct SubLayout;

template <typename LayoutType, typename IndexType, typename... Slices>
struct SubLayout<LayoutType, camp::list<Slices...>, IndexType> {

    using IndexLinear = IndexType;

    const LayoutType& parent_layout_;
    camp::tuple<Slices...> slices_;

    static inline constexpr size_t num_slices_ = sizeof...(Slices);

    static inline constexpr 
    IndexType n_dims = ((Slices::reduces_dimension == false ? 1 : 0) + ...);

    static inline constexpr 
    camp::array<IndexType, num_slices_> slice_to_parent_map_ = 
        make_slice_to_parent_index_map<IndexType, Slices...>();

    static inline constexpr 
    camp::array<IndexType, n_dims> parent_to_slice_map_ = 
        make_parent_to_slice_index_map<IndexType, n_dims, Slices...>();

    RAJA_INLINE RAJA_HOST_DEVICE constexpr SubLayout(const LayoutType& parent_layout, Slices... slices)
        : parent_layout_(parent_layout), slices_(slices...) { }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_parent_layout() const {
        return parent_layout_;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_slices() const {
        return slices_;
    }

    template<IndexType Index>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto& get_slice() const {
        return camp::get<Index>(slices_);
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size() const {

        IndexType prod_dims = 1;
        for_each_tuple_index( slices_,
            [&](auto slice, auto index) {
                const IndexType dim_size = slice.template size<index>(parent_layout_);
                prod_dims *= (dim_size == 0) ? 1 : dim_size;
            });

        return prod_dims;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size_noproj() const {

        IndexType prod_dims = 1;
        for_each_tuple_index( slices_,
            [&](auto slice, auto index) {
                prod_dims *= slice.template size<index>(parent_layout_);
            });

        return prod_dims;
    }

    template<IndexType DIM> 
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_size() const {
        constexpr auto SliceDim = parent_to_slice_map_[DIM];
        return camp::get<SliceDim>(slices_).template size<DIM>(parent_layout_);
    }

    template<IndexType DIM> 
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_stride() const {
        constexpr auto SliceDim = parent_to_slice_map_[DIM];
        return camp::get<SliceDim>(slices_).stride();
    }

    template <typename... Idxs>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto operator()(Idxs... idxs) const {
        static_assert(sizeof...(idxs) == n_dims, "Wrong number of indices for subview");

        camp::array<IndexType, n_dims> arr{idxs...};
        camp::array<IndexType, num_slices_> parent_indices;

        for_each_tuple_index( slices_,
            [&](auto slice, auto index) {
                if (slice_to_parent_map_[index] >= 0) {
                    parent_indices[index] = slice.map_index(arr[slice_to_parent_map_[index]]); 
                } else {
                    // map_index will not need index values for dimension-reducing slices
                    // so we pass a "dummy" value.
                    constexpr IndexType dummy_value = -1;
                    parent_indices[index] = slice.map_index(dummy_value); 
                }
            });

        return camp::apply(parent_layout_, parent_indices);
    }
};

template <typename LayoutType, typename... Slices>
SubLayout(LayoutType, Slices...) -> SubLayout<LayoutType, camp::list<Slices...>>;

}  // namespace RAJA

#endif