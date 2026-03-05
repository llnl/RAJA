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

    template<IndexType RAJA_UNUSED_ARG(DIM), typename LayoutType>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const {
        return (end_ - start_);
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
        return (layout.template get_dim_size<DIM>() - start_);
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return 1;
    }
};

template<typename IndexType = Index_type>
struct FixedSlice { 
    IndexType idx_; 

    static constexpr bool reduces_dimension = true;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index() const {
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
        if (stride_ == 0) {
            return 0;
        } else if (stride_ > 0) {
            if (start_ >= end_) {
                return 0;
            }
            return (end_ - start_ + stride_ - 1) / stride_;
        } else {
            if (start_ <= end_) {
                return 0;
            }
            return (start_ - end_ - stride_ - 1) / (-stride_);
        }
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const {
        return stride_;
    }
};

template<typename IndexType, typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_slice_to_parent_index_map() {
    IndexType sub_idx = 0;
    camp::array<IndexType, sizeof...(Slices)> map{{(Slices::reduces_dimension ? IndexType(0) : sub_idx++)...}};
    return map;
}

template<typename IndexType, typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_parent_to_slice_index_map() {

    constexpr IndexType n_dims = (!Slices::reduces_dimension + ...);
    IndexType sub_idx = 0;
    IndexType i = 0;
    camp::array<IndexType, n_dims> map{};

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
struct SubRegion;

/* SubLayout is a semantic alias for a SubRegion whose parent is a layout */
template <typename LayoutType, typename SliceTypes, typename IndexType = Index_type>
using SubLayout = SubRegion<LayoutType, SliceTypes, IndexType>;

/* SubView is a semantic alias for a SubRegion whose parent is a view */
template <typename LayoutType, typename SliceTypes, typename IndexType = Index_type>
using SubView = SubRegion<LayoutType, SliceTypes, IndexType>;

template <typename LayoutType, typename IndexType, typename... Slices>
struct SubRegion<LayoutType, camp::list<Slices...>, IndexType> {

    using IndexLinear = IndexType;

    static inline constexpr size_t s_num_slices = sizeof...(Slices);
    static_assert(s_num_slices == LayoutType::n_dims, "Wrong number of slices");

    static inline constexpr 
    IndexType n_dims = ((!Slices::reduces_dimension ? 1 : 0) + ...);

    static inline constexpr 
    camp::array<IndexType, s_num_slices> s_slice_to_parent_map = 
        make_slice_to_parent_index_map<IndexType, Slices...>();

    static inline constexpr 
    camp::array<IndexType, n_dims> s_parent_to_slice_map = 
        make_parent_to_slice_index_map<IndexType, Slices...>();

    const LayoutType m_parent;
    camp::tuple<Slices...> m_slices;

    RAJA_INLINE RAJA_HOST_DEVICE constexpr SubRegion(const LayoutType& parent, Slices... slices)
        : m_parent(parent), m_slices(slices...) { }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr const auto& get_parent() const {
        return m_parent;
    }

    RAJA_INLINE RAJA_HOST_DEVICE constexpr const auto& get_slices() const {
        return m_slices;
    }

    template<IndexType Index>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr const auto& get_slice() const {
        return camp::get<Index>(m_slices);
    }

	    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size() const {

	        IndexType prod_dims = 1;
	        for_each_tuple_index( m_slices,
	            [&](auto slice, auto index) {
	                const IndexType dim_size =
	                    decltype(slice)::reduces_dimension ? IndexType(1)
	                                                       : slice.template size<index>(m_parent);
	                prod_dims *= (dim_size == IndexType(0)) ? IndexType(1) : dim_size;
	            });

	        return prod_dims;
	    }

	    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size_noproj() const {

	        IndexType prod_dims = 1;
	        for_each_tuple_index( m_slices,
	            [&](auto slice, auto index) {
	                prod_dims *=
	                    decltype(slice)::reduces_dimension ? IndexType(1)
	                                                       : slice.template size<index>(m_parent);
	            });

	        return prod_dims;
	    }

	    template<IndexType DIM>
	    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_size() const {
	        static_assert(DIM < n_dims, "DIM out of bounds");
	        constexpr auto SliceDim = s_parent_to_slice_map[DIM];
	        return camp::get<SliceDim>(m_slices).template size<SliceDim>(m_parent);
	    }

	    template<IndexType DIM>
	    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_stride() const {
	        static_assert(DIM < n_dims, "DIM out of bounds");
	        constexpr auto SliceDim = s_parent_to_slice_map[DIM];
	        return m_parent.template get_dim_stride<SliceDim>() *
	               camp::get<SliceDim>(m_slices).stride();
	    }

    template <typename... Idxs>
    RAJA_INLINE RAJA_HOST_DEVICE constexpr auto operator()(Idxs... idxs) const {
        static_assert(sizeof...(idxs) == n_dims, "Wrong number of indices");

        camp::array<IndexType, n_dims> arr{idxs...};
        camp::array<IndexType, s_num_slices> parent_indices{};

        for_each_tuple_index( m_slices,
            [&](auto slice, auto index) {
                if constexpr (decltype(slice)::reduces_dimension) {
                    parent_indices[index] = slice.map_index();
                } else {
                    parent_indices[index] = slice.map_index(arr[s_slice_to_parent_map[index]]);
                }
            });

        return camp::apply(m_parent, parent_indices);
    }
};

template <typename LayoutType, typename... Slices>
SubRegion(LayoutType, Slices...) -> SubRegion<LayoutType, camp::list<Slices...>>;

}  // namespace RAJA

#endif
