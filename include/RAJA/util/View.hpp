/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a multi-dimensional view class.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_VIEW_HPP
#define RAJA_VIEW_HPP

#include <array>
#include <type_traits>

#include "RAJA/config.hpp"
#include "RAJA/pattern/atomic.hpp"
#include "RAJA/util/IndexLayout.hpp"
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"
#include "RAJA/util/TypedViewBase.hpp"

namespace RAJA
{

// Helpers to convert
// layouts -> OffsetLayouts
// Typedlayouts -> TypedOffsetLayouts
template<typename layout>
struct add_offset
{
  using type = RAJA::OffsetLayout<layout::n_dims>;
};

template<typename IdxLin, typename... DimTypes>
struct add_offset<RAJA::TypedLayout<IdxLin, camp::tuple<DimTypes...>>>
{
  using type = RAJA::TypedOffsetLayout<IdxLin, camp::tuple<DimTypes...>>;
};

template<typename ValueType,
         typename LayoutType,
         typename PointerType = ValueType*>
using View = internal::ViewBase<ValueType, PointerType, LayoutType>;


template<typename ValueType, typename LayoutType, typename... IndexTypes>
using TypedView = internal::
    TypedViewBase<ValueType, ValueType*, LayoutType, camp::list<IndexTypes...>>;

template<typename IndexType, typename ValueType>
RAJA_HOST_DEVICE RAJA_INLINE constexpr View<ValueType, Layout<1, IndexType, 0>>
make_view(ValueType* ptr)
{
  return View<ValueType, Layout<1, IndexType, 0>>(ptr, 1);
}

template<size_t n_dims,
         typename IndexType,
         typename ValueType,
         typename... IndexTypes>
RAJA_HOST_DEVICE RAJA_INLINE constexpr View<
    ValueType,
    IndexLayout<n_dims, IndexType, IndexTypes...>>
make_index_view(ValueType* ptr,
                IndexLayout<n_dims, IndexType, IndexTypes...> index_layout)
{
  return View<ValueType, IndexLayout<n_dims, IndexType, IndexTypes...>>(
      ptr, index_layout);
}

// select certain indices from a tuple, given a curated index sequence
// returns linear index of layout(ar...)
template<typename Lay, typename Tup, camp::idx_t... Idxs>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto selecttuple(Lay lyout,
                                                        Tup&& tup,
                                                        camp::idx_seq<Idxs...>)
    -> decltype(lyout(camp::get<Idxs>(std::forward<Tup>(tup))...))
{
  return lyout(camp::get<Idxs>(std::forward<Tup>(tup))...);
}

// sequence combiner
template<typename Seq1, typename Seq2>
struct cat_seq;

template<camp::idx_t... Idxs1, camp::idx_t... Idxs2>
struct cat_seq<camp::idx_seq<Idxs1...>, camp::idx_seq<Idxs2...>>
{
  using type = camp::idx_seq<Idxs1..., Idxs2...>;
};

template<typename Seq1, typename Seq2>
using cat_seq_t = typename cat_seq<Seq1, Seq2>::type;

// sequence offsetter
template<camp::idx_t Offset, typename Seq>
struct offset_seq;

template<camp::idx_t Offset, camp::idx_t... Idxs>
struct offset_seq<Offset, camp::idx_seq<Idxs...>>
{
  using type = camp::idx_seq<(Idxs + Offset)...>;
};

template<camp::idx_t Offset, typename Seq>
using offset_seq_t = typename offset_seq<Offset, Seq>::type;

// remove the Nth index in a parameter pack
// returns linear index of layout(ar...)
template<typename Lay, RAJA::Index_type Nth = 0, typename Tup>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto removenth(Lay lyout, Tup&& tup)
    -> decltype(selecttuple<Lay>(
        lyout,
        std::forward<Tup>(tup),
        cat_seq_t<
            camp::make_idx_seq_t<Nth>,  // sequence up to Nth
            offset_seq_t<Nth + 1,       // after Nth
                         camp::make_idx_seq_t<camp::tuple_size<Tup>::value -
                                              Nth - 1>>  // sequence after Nth
            > {}))
{
  return selecttuple<Lay>(
      lyout, std::forward<Tup>(tup),
      cat_seq_t<camp::make_idx_seq_t<Nth>,  // sequence up to Nth
                offset_seq_t<Nth + 1,       // after Nth
                             camp::make_idx_seq_t<camp::tuple_size<Tup>::value -
                                                  Nth - 1>>  // sequence after
                                                             // Nth
                > {});
}

// P2Pidx represents the array-of-pointers index. This allows the position of
// the index into the array-of-pointers to be moved around in the MultiView
// operator(); see the operator overload. Default of 0 means that the p2p index
// is in the 0th position.
template<
    typename ValueType,
    typename LayoutType,
    RAJA::Index_type P2Pidx      = 0,
    typename PointerType         = ValueType**,
    typename NonConstPointerType = camp::type::ptr::add<  // adds *
        camp::type::ptr::add<camp::type::cv::rem<         // removes cv
            camp::type::ptr::rem<camp::type::ptr::rem<PointerType>  // removes
                                                                    // *
                                 >>>>>
struct MultiView
{
  using value_type      = ValueType;
  using pointer_type    = PointerType;
  using layout_type     = LayoutType;
  using nc_value_type   = camp::decay<value_type>;
  using nc_pointer_type = NonConstPointerType;
  using NonConstView =
      MultiView<nc_value_type, layout_type, P2Pidx, nc_pointer_type>;

  layout_type layout {};
  nc_pointer_type data = nullptr;

  MultiView() = default;

  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr MultiView(pointer_type data_ptr,
                                                   Args... dim_sizes)
      : layout(dim_sizes...),
        data(nc_pointer_type(data_ptr))
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr MultiView(pointer_type data_ptr,
                                                   layout_type const& ly)
      : layout(ly),
        data(nc_pointer_type(data_ptr))
  {}

  // TODO: Should be able to construct a const MultiView from non-const array of
  // arrays. For now, this becomes an ambiguous call to constructor error.
  // template<typename... Args,
  //         bool IsConstValue = std::is_const<value_type>::value>
  // RAJA_HOST_DEVICE RAJA_INLINE constexpr MultiView(
  //    std::enable_if_t<IsConstValue, NonConstPointerType> data_ptr,
  //    Args... dim_sizes)
  //    : layout(dim_sizes...),
  //      data(nc_pointer_type(data_ptr))
  //{}
  // template<bool IsConstValue = std::is_const<value_type>::value>
  // RAJA_HOST_DEVICE RAJA_INLINE constexpr MultiView(
  //    std::enable_if_t<IsConstValue, NonConstPointerType> data_ptr,
  //    layout_type const& ly)
  //    : layout(ly),
  //      data(nc_pointer_type(data_ptr))
  //{}

  RAJA_INLINE constexpr MultiView(MultiView const&)            = default;
  RAJA_INLINE constexpr MultiView(MultiView&&)                 = default;
  RAJA_INLINE constexpr MultiView& operator=(MultiView const&) = default;
  RAJA_INLINE constexpr MultiView& operator=(MultiView&&)      = default;

  template<bool IsConstView = std::is_const<value_type>::value>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr MultiView(
      std::enable_if_t<IsConstView, NonConstView> const& rhs)
      : layout(rhs.layout),
        data(nc_pointer_type(rhs.data))
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr void set_layout(layout_type const& ly)
  {
    layout = ly;
  }

  template<bool IsConstValue = std::is_const<value_type>::value>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr void set_data(
      std::enable_if_t<IsConstValue, NonConstPointerType> data_ptr)
  {
    data = nc_pointer_type(
        data_ptr);  // This data_ptr should already be non-const.
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr void set_data(pointer_type data_ptr)
  {
    data = nc_pointer_type(data_ptr);
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr layout_type const& get_layout() const
  {
    return layout;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr pointer_type get_data() const
  {
    return pointer_type(data);
  }

  // why doesn't this return a shifted copy instead of shifting this view?
  template<size_t n_dims = layout_type::n_dims, typename IdxLin = Index_type>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr RAJA::
      MultiView<ValueType, typename add_offset<layout_type>::type, P2Pidx>
      shift(const std::array<IdxLin, n_dims>& shift)
  {
    static_assert(n_dims == layout_type::n_dims,
                  "Dimension mismatch in MultiView shift");

    typename add_offset<layout_type>::type shift_layout(layout);
    shift_layout.shift(shift);

    return RAJA::MultiView<ValueType, typename add_offset<layout_type>::type,
                           P2Pidx>(pointer_type(data), shift_layout);
  }

  // Moving the position of the index into the array-of-pointers
  // is set by P2Pidx, which is defaulted to 0.
  // making this specifically typed would require unpacking the layout,
  // this is easier to maintain
  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE value_type& operator()(Args... ar) const
  {
    auto pidx =
        stripIndexType(camp::get<P2Pidx>(camp::forward_as_tuple(ar...)));

    if (pidx < 0)
    {
      RAJA_ABORT_OR_THROW(
          "Negative index while accessing array of pointers.\n");
    }
    auto idx = stripIndexType(
        removenth<LayoutType, P2Pidx>(layout, camp::forward_as_tuple(ar...)));
    return pointer_type(data)[pidx][idx];
  }
};

template<typename ViewType, typename AtomicPolicy = RAJA::auto_atomic>
struct AtomicViewWrapper
{
  using base_type    = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type   = typename base_type::value_type;
  using atomic_type  = RAJA::AtomicRef<value_type, AtomicPolicy>;

  base_type base_;

  RAJA_HOST_DEVICE RAJA_INLINE constexpr explicit AtomicViewWrapper(
      ViewType view)
      : base_(view)
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr void set_data(pointer_type data_ptr)
  {
    base_.set_data(data_ptr);
  }

  template<typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE atomic_type operator()(ARGS&&... args) const
  {
    return atomic_type(&base_.operator()(std::forward<ARGS>(args)...));
  }
};

/*
 * Specialized AtomicViewWrapper for seq_atomic that acts as pass-thru
 * for performance
 */
template<typename ViewType>
struct AtomicViewWrapper<ViewType, RAJA::seq_atomic>
{
  using base_type    = ViewType;
  using pointer_type = typename base_type::pointer_type;
  using value_type   = typename base_type::value_type;
  using atomic_type  = RAJA::AtomicRef<value_type, RAJA::seq_atomic>;

  base_type base_;

  RAJA_HOST_DEVICE RAJA_INLINE constexpr explicit AtomicViewWrapper(
      ViewType const& view)
      : base_ {view}
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr void set_data(pointer_type data_ptr)
  {
    base_.set_data(data_ptr);
  }

  template<typename... ARGS>
  RAJA_HOST_DEVICE RAJA_INLINE value_type& operator()(ARGS&&... args) const
  {
    return base_.operator()(std::forward<ARGS>(args)...);
  }
};

template<typename AtomicPolicy, typename ViewType>
RAJA_HOST_DEVICE RAJA_INLINE constexpr AtomicViewWrapper<ViewType, AtomicPolicy>
make_atomic_view(ViewType const& view)
{

  return RAJA::AtomicViewWrapper<ViewType, AtomicPolicy>(view);
}

struct layout_left
{};

struct layout_right
{};

namespace detail
{

template<typename meta_layout>
struct PermutedViewHelper;

template<typename T>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get_last_index(T last)
{
  return last;
}

template<typename T0, typename T1, typename... Args>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get_last_index(T0,
                                                           T1 t1,
                                                           Args... args)
{
  return get_last_index(t1, args...);
}

template<std::size_t... stride_order_idx>
struct PermutedViewHelper<std::index_sequence<stride_order_idx...>>
{
  template<typename IndexType, typename T, typename... Extents>
  static RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get(T* ptr,
                                                         Extents&&... extents)
  {
    constexpr int N = sizeof...(Extents);

    auto custom_layout = RAJA::make_permuted_layout(
        std::array<RAJA::idx_t, N> {std::forward<Extents>(extents)...},
        std::array<RAJA::idx_t, N> {stride_order_idx...});

    constexpr auto unit_stride = detail::get_last_index(stride_order_idx...);
    using view_t = RAJA::View<T, RAJA::Layout<N, IndexType, unit_stride>>;

    return view_t(ptr, custom_layout);
  }
};

template<>
struct PermutedViewHelper<layout_right>
{
  template<typename IndexType, typename T, typename... Extents>
  static RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get(T* ptr,
                                                         Extents&&... extents)
  {
    constexpr int N = sizeof...(Extents);
    using view_t    = RAJA::View<T, RAJA::Layout<N, IndexType, N - 1>>;

    return view_t(ptr, std::forward<Extents>(extents)...);
  }
};

template<std::size_t... idx>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto make_reverse_array(
    std::index_sequence<idx...>)
{
  return std::array<RAJA::idx_t, sizeof...(idx)> {sizeof...(idx) - 1U - idx...};
}

template<>
struct PermutedViewHelper<layout_left>
{
  template<typename IndexType, typename T, typename... Extents>
  static RAJA_HOST_DEVICE RAJA_INLINE constexpr auto get(T* ptr,
                                                         Extents&&... extents)
  {
    constexpr int N = sizeof...(Extents);

    auto reverse_layout = RAJA::make_permuted_layout(
        std::array<RAJA::idx_t, N> {std::forward<Extents>(extents)...},
        make_reverse_array(std::make_index_sequence<N> {}));
    using view_t = RAJA::View<T, RAJA::Layout<N, IndexType, 0U>>;

    return view_t(ptr, reverse_layout);
  }
};

}  // namespace detail

template<typename meta_layout,
         typename IndexType = RAJA::Index_type,
         typename T,
         typename... Extents>
RAJA_HOST_DEVICE RAJA_INLINE constexpr auto make_permuted_view(
    T* ptr,
    Extents&&... extents)
{
  return detail::PermutedViewHelper<meta_layout>::template get<IndexType>(
      ptr, std::forward<Extents>(extents)...);
}

}  // namespace RAJA

#endif
