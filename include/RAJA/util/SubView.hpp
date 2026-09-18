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
 * \brief   RAJA header file defining subview slicing utilities
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

namespace internal
{
template<typename ValueType, typename PointerType, typename LayoutType>
class ViewBase;
}

/*!
 * \brief Select a half-open range of parent indices without reducing the
 * dimension.
 */
template<typename IndexType = Index_type>
struct RangeSlice
{
  static constexpr bool reduces_dimension = false;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr RangeSlice(IndexType start,
                                                    IndexType end)
      : m_start(start),
        m_end(end)
  {}

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename ParentType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx,
      const ParentType&) const
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

/*!
 * \brief Select parent indices from start to the end of the parent dimension.
 */
template<typename IndexType = Index_type>
struct RangeStartSlice
{
  static constexpr bool reduces_dimension = false;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr explicit RangeStartSlice(
      IndexType start)
      : m_start(start)
  {}

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename ParentType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx,
      const ParentType&) const
  {
    return m_start + idx;
  }

  template<size_t ParentDim, typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(
      const LayoutType& layout) const
  {
    return (layout.template get_dim_begin<ParentDim>() +
            layout.template get_dim_size<ParentDim>() - m_start);
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const { return 1; }

private:
  IndexType m_start;
};

/*!
 * \brief Select one parent index and remove that dimension from the result.
 */
template<typename IndexType = Index_type>
struct FixedSlice
{
  static constexpr bool reduces_dimension = true;

  RAJA_INLINE RAJA_HOST_DEVICE constexpr explicit FixedSlice(IndexType idx)
      : m_idx(idx)
  {}

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename ParentType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      const ParentType&) const
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

/*!
 * \brief Select an entire parent dimension.
 */
template<typename IndexType = Index_type>
struct NoSlice
{
  static constexpr bool reduces_dimension = false;

  template<size_t ParentDim, typename ParentType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx,
      const ParentType& parent) const
  {
    return parent.template get_dim_begin<ParentDim>() + idx;
  }

  template<size_t ParentDim, typename LayoutType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType size(
      const LayoutType& layout) const
  {
    return layout.template get_dim_size<ParentDim>();
  }

  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType stride() const { return 1; }
};

/*!
 * \brief Select a strided half-open range of parent indices without reducing
 * the dimension.
 */
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

  template<size_t RAJA_UNUSED_ARG(ParentDim), typename ParentType>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexType map_index(
      IndexType idx,
      const ParentType&) const
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

namespace detail
{

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

}  // namespace detail

/*!
 * \brief Adapt a parent layout or view by applying a slice to each dimension.
 */
template<typename ParentType,
         typename SliceTypes,
         typename IndexType = Index_type>
struct SlicingAdapter;

/*!
 * \brief A SlicingAdapter whose parent is a layout.
 */
template<typename LayoutType,
         typename SliceTypes,
         typename IndexType = Index_type>
using SubLayout = SlicingAdapter<LayoutType, SliceTypes, IndexType>;

template<typename ParentType, typename IndexType, typename... Slices>
struct SlicingAdapter<ParentType, camp::list<Slices...>, IndexType>
{
  using IndexLinear = IndexType;

  static inline constexpr size_t n_dims =
      ((!Slices::reduces_dimension ? 1 : 0) + ...);

  /*!
   * \brief Construct an adapter from a parent and one slice per parent
   * dimension.
   */
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

  /*!
   * \brief Return the product of dimension sizes, treating zero-sized
   * dimensions as projected dimensions of size one.
   */
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

  /*!
   * \brief Return the product of dimension sizes, including zero-sized
   * dimensions.
   */
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

  template<size_t SubregionDim>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr IndexLinear get_dim_begin() const
  {
    static_assert(SubregionDim < n_dims, "Dimension out of bounds");
    return IndexLinear(0);
  }

  /*!
   * \brief Access the parent using indices mapped from the subregion index
   * space.
   */
  template<typename... Idxs>
  RAJA_INLINE RAJA_HOST_DEVICE constexpr decltype(auto) operator()(
      Idxs... idxs) const
  {
    static_assert(sizeof...(idxs) == n_dims, "Wrong number of indices");

    camp::array<IndexType, n_dims> subregion_indices {idxs...};
    camp::array<IndexType, s_num_slices> parent_indices {};

    for_each_tuple_index(m_slices, [&](auto slice, auto parent_dim) {
      if constexpr (decltype(slice)::reduces_dimension)
      {
        parent_indices[parent_dim] =
            slice.template map_index<parent_dim>(m_parent);
      }
      else
      {
        parent_indices[parent_dim] = slice.template map_index<parent_dim>(
            subregion_indices[s_parent_to_subregion_dim[parent_dim]], m_parent);
      }
    });

    return camp::apply(m_parent, parent_indices);
  }

private:
  static inline constexpr size_t s_num_slices = sizeof...(Slices);
  static_assert(s_num_slices == ParentType::n_dims, "Wrong number of slices");

  static inline constexpr camp::array<size_t, s_num_slices>
      s_parent_to_subregion_dim =
          detail::make_parent_to_subregion_dim_map<Slices...>();

  static inline constexpr camp::array<size_t, n_dims>
      s_subregion_to_parent_dim =
          detail::make_subregion_to_parent_dim_map<Slices...>();

  const ParentType m_parent;
  camp::tuple<Slices...> m_slices;
};

template<typename ParentType, typename... Slices>
RAJA_HOST_DEVICE SlicingAdapter(ParentType, Slices...)
    -> SlicingAdapter<ParentType, camp::list<Slices...>>;

/*!
 * \brief Create a sliced view whose layout maps indices into a parent view.
 *
 * One slice must be provided for each dimension of the parent view.
 *
 * \param view Parent view to slice.
 * \param slices Slice specification for each parent dimension.
 * \return A view over the parent data using a SubLayout.
 */
template<typename ValueType,
         typename LayoutType,
         typename PointerType,
         typename... Slices>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto make_subview(
    const internal::ViewBase<ValueType, PointerType, LayoutType>& view,
    Slices... slices)
{
  using SubLayoutType = SubLayout<LayoutType, camp::list<Slices...>,
                                  typename LayoutType::IndexLinear>;
  return internal::ViewBase<ValueType, PointerType, SubLayoutType>(
      view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

}  // namespace RAJA

#endif
