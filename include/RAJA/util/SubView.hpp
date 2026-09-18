//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_SUBVIEW_HPP
#define RAJA_SUBVIEW_HPP

/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining the SubView class
 *
 ******************************************************************************
 */

#include "RAJA/util/for_each.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "camp/tuple.hpp"
#include "camp/array.hpp"

namespace RAJA
{

template<typename IndexType = Index_type>
struct RangeSlice
{
  static constexpr bool reduces_dimension = false;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr RangeSlice(IndexType start,
                                                    IndexType end)
      : m_start(start),
        m_end(end)
  {}

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx) const
  {
    return m_start + idx;
  }

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const
  {
    return (m_end - m_start);
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const { return 1; }

private:
  IndexType m_start, m_end;
};

template<typename IndexType = Index_type>
struct RangeStartSlice
{
  static constexpr bool reduces_dimension = false;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr explicit RangeStartSlice(
      IndexType start)
      : m_start(start)
  {}

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx) const
  {
    return m_start + idx;
  }

  template<size_t ParentDim, typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(
      const LayoutType& layout) const
  {
    return (layout.template get_dim_size<ParentDim>() - m_start);
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const { return 1; }

private:
  IndexType m_start;
};

template<typename IndexType = Index_type>
struct FixedSlice
{
  static constexpr bool reduces_dimension = true;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr explicit FixedSlice(IndexType idx)
      : m_idx(idx)
  {}

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index() const
  {
    return m_idx;
  }

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const
  {
    return 1;
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const { return 1; }

private:
  IndexType m_idx;
};

template<typename IndexType = Index_type>
struct NoSlice
{
  static constexpr bool reduces_dimension = false;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx) const
  {
    return idx;
  }

  template<size_t ParentDim, typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(
      const LayoutType& layout) const
  {
    return layout.template get_dim_size<ParentDim>();
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const { return 1; }
};

template<typename IndexType = Index_type>
struct StridedSlice
{
  static constexpr bool reduces_dimension = false;

  /*!
   * \brief Construct a slice over [start, end) using the given stride.
   *
   * \pre stride must be nonzero.
   */
  RAJA_INLINE RAJA_HOST_DEVICE constexpr StridedSlice(IndexType start,
                                                      IndexType end,
                                                      IndexType stride)
      : m_start(start),
        m_end(end),
        m_stride(stride)
  {}

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx) const
  {
    return m_start + m_stride * idx;
  }

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(const LayoutType&) const
  {
    if (m_stride > 0)
    {
      if (m_start >= m_end)
      {
        return 0;
      }
      return (m_end - m_start + m_stride - 1) / m_stride;
    }
    else
    {
      if (m_start <= m_end)
      {
        return 0;
      }
      return (m_start - m_end - m_stride - 1) / (-m_stride);
    }
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const
  {
    return m_stride;
  }

private:
  IndexType m_start, m_end, m_stride;
};

template<typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_parent_to_subregion_dim_map()
{
  size_t subregion_dim = 0;
  camp::array<size_t, sizeof...(Slices)> map {
      {(Slices::reduces_dimension ? size_t(0) : subregion_dim++)...}};
  return map;
}

template<typename... Slices>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto make_subregion_to_parent_dim_map()
{
  constexpr size_t n_dims = (!Slices::reduces_dimension + ...);
  size_t subregion_dim    = 0;
  size_t parent_dim       = 0;
  camp::array<size_t, n_dims> map {};

  auto process_slice = [&](bool reduces_dimension) constexpr {
    if (!reduces_dimension)
    {
      map[subregion_dim++] = parent_dim;
    }
    parent_dim++;
  };

  (process_slice(Slices::reduces_dimension), ...);

  return map;
}

template<typename ParentType,
         typename SliceTypes,
         typename IndexType = Index_type>
struct SlicingAdapter;

/* SubLayout is a semantic alias for a SlicingAdapter whose parent is a
 * layout */
template<typename LayoutType,
         typename SliceTypes,
         typename IndexType = Index_type>
using SubLayout = SlicingAdapter<LayoutType, SliceTypes, IndexType>;

/* SubView is a semantic alias for a SlicingAdapter whose parent is a view */
template<typename ViewType,
         typename SliceTypes,
         typename IndexType = Index_type>
using SubView = SlicingAdapter<ViewType, SliceTypes, IndexType>;

template<typename ParentType, typename IndexType, typename... Slices>
struct SlicingAdapter<ParentType, camp::list<Slices...>, IndexType>
{
  using IndexLinear = IndexType;

  static inline constexpr size_t n_dims =
      ((!Slices::reduces_dimension ? 1 : 0) + ...);

  RAJA_INLINE RAJA_HOST_DEVICE constexpr SlicingAdapter(
      const ParentType& parent,
      Slices... slices)
      : m_parent(parent),
        m_slices(slices...)
  {}

  RAJA_INLINE RAJA_HOST_DEVICE constexpr const auto& get_parent() const
  {
    return m_parent;
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr const auto& get_slices() const
  {
    return m_slices;
  }

  template<size_t ParentDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr const auto& get_slice() const
  {
    return camp::get<ParentDim>(m_slices);
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size() const
  {
    IndexType prod_dims = 1;
    for_each_tuple_index(m_slices, [&](auto slice, auto parent_dim) {
      const IndexType dim_size =
          decltype(slice)::reduces_dimension
              ? IndexType(1)
              : slice.template size<parent_dim>(m_parent);
      prod_dims *= (dim_size == IndexType(0)) ? IndexType(1) : dim_size;
    });

    return prod_dims;
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr auto size_noproj() const
  {
    IndexType prod_dims = 1;
    for_each_tuple_index(m_slices, [&](auto slice, auto parent_dim) {
      prod_dims *= decltype(slice)::reduces_dimension
                       ? IndexType(1)
                       : slice.template size<parent_dim>(m_parent);
    });

    return prod_dims;
  }

  template<size_t SubregionDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr auto get_dim_size() const
  {
    static_assert(SubregionDim < n_dims, "Dimension out of bounds");
    constexpr auto parent_dim = s_subregion_to_parent_dim[SubregionDim];
    return camp::get<parent_dim>(m_slices).template size<parent_dim>(m_parent);
  }

  template<size_t SubregionDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_stride() const
  {
    static_assert(SubregionDim < n_dims, "Dimension out of bounds");
    constexpr auto parent_dim = s_subregion_to_parent_dim[SubregionDim];
    return m_parent.template get_dim_stride<parent_dim>() *
           camp::get<parent_dim>(m_slices).stride();
  }

  template<typename... Idxs>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr auto operator()(Idxs... idxs) const
  {
    static_assert(sizeof...(idxs) == n_dims, "Wrong number of indices");

    camp::array<IndexType, n_dims> subregion_indices {idxs...};
    camp::array<IndexType, s_num_slices> parent_indices {};

    for_each_tuple_index(m_slices, [&](auto slice, auto parent_dim) {
      if constexpr (decltype(slice)::reduces_dimension)
      {
        parent_indices[parent_dim] = slice.map_index();
      }
      else
      {
        parent_indices[parent_dim] = slice.map_index(
            subregion_indices[s_parent_to_subregion_dim[parent_dim]]);
      }
    });

    return camp::apply(m_parent, parent_indices);
  }

private:
  static inline constexpr size_t s_num_slices = sizeof...(Slices);
  static_assert(s_num_slices == ParentType::n_dims, "Wrong number of slices");

  static inline constexpr camp::array<size_t, s_num_slices>
      s_parent_to_subregion_dim = make_parent_to_subregion_dim_map<Slices...>();

  static inline constexpr camp::array<size_t, n_dims>
      s_subregion_to_parent_dim = make_subregion_to_parent_dim_map<Slices...>();

  const ParentType m_parent;
  camp::tuple<Slices...> m_slices;
};

template<typename ParentType, typename... Slices>
SlicingAdapter(ParentType, Slices...)
    -> SlicingAdapter<ParentType, camp::list<Slices...>>;

}  // namespace RAJA

#endif
