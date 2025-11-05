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

#ifndef RAJA_util_TypedViewBase_HPP
#define RAJA_util_TypedViewBase_HPP

#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/pattern/atomic.hpp"

#if defined(RAJA_ENABLE_VECTORIZATION)
#include "RAJA/pattern/tensor.hpp"
#endif

#include "RAJA/util/Layout.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/OffsetLayout.hpp"

namespace RAJA
{

namespace internal
{

template<camp::idx_t, typename T>
struct IndexToType
{
  using type = T;
};

template<typename IdxSeq, typename T>
struct SequenceToType;

template<camp::idx_t... Perm, typename T>
struct SequenceToType<camp::idx_seq<Perm...>, T>
{
  using type = camp::list<typename IndexToType<Perm, T>::type...>;
};

template<typename Perm>
using getDefaultIndexTypes =
    typename SequenceToType<Perm, RAJA::Index_type>::type;

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


#if defined(RAJA_ENABLE_VECTORIZATION)
namespace detail
{
/*
 * Returns the argument number which contains a VectorIndex
 *
 * returns -1 if none of the arguments are VectorIndexs
 */

template<camp::idx_t DIM, typename ARGS, typename IDX_SEQ>
struct GetTensorArgIdxExpanded;

template<camp::idx_t DIM, typename... ARGS, camp::idx_t... IDX>
struct GetTensorArgIdxExpanded<DIM, camp::list<ARGS...>, camp::idx_seq<IDX...>>
{

  static constexpr camp::idx_t value = RAJA::max<camp::idx_t>(
      (internal::expt::isTensorIndex<ARGS>() &&
               internal::expt::getTensorDim<ARGS>() == DIM
           ? IDX
           : -1)...);
};


}  // namespace detail
#endif


/*
 * Returns the number of arguments which are VectorIndexs
 */
template<typename... ARGS>
struct count_num_tensor_args
{
  static constexpr camp::idx_t value =
#if defined(RAJA_ENABLE_VECTORIZATION)
      RAJA::sum<camp::idx_t>(
          (internal::expt::isTensorIndex<ARGS>() ? 1 : 0)...);
#else
      0;  // There should be 0 Tensor indices if not vectorizing.
#endif
};

#if defined(RAJA_ENABLE_VECTORIZATION)
/*
 * Returns which argument has a vector index
 */
template<camp::idx_t DIM, typename... ARGS>
struct GetTensorArgIdx
{
  static constexpr camp::idx_t value = detail::GetTensorArgIdxExpanded<
      DIM,
      camp::list<ARGS...>,
      camp::make_idx_seq_t<sizeof...(ARGS)>>::value;
};

template<camp::idx_t DIM, typename... ARGS>
struct GetTensorArgIdx<DIM, camp::list<ARGS...>>
{
  static constexpr camp::idx_t value = detail::GetTensorArgIdxExpanded<
      DIM,
      camp::list<ARGS...>,
      camp::make_idx_seq_t<sizeof...(ARGS)>>::value;
};

/*
 * Returns the beginning index in a vector argument
 */
template<camp::idx_t DIM, typename LAYOUT, typename... ARGS>
RAJA_INLINE RAJA_HOST_DEVICE static constexpr camp::idx_t get_tensor_args_begin(
    LAYOUT const& layout,
    ARGS... args)
{
  return RAJA::max<camp::idx_t>(
      internal::expt::getTensorDim<ARGS>() == DIM
          ? internal::expt::getTensorBegin<ARGS>(
                args, layout.template get_dim_begin<
                          GetTensorArgIdx<DIM, ARGS...>::value>())
          : 0 ...);
}

/*
 * Returns the number of elements in the vector argument
 */
template<camp::idx_t DIM, typename LAYOUT, typename... ARGS>
RAJA_INLINE RAJA_HOST_DEVICE static constexpr camp::idx_t get_tensor_args_size(
    LAYOUT const& layout,
    ARGS... args)
{
  return RAJA::max<camp::idx_t>(
      internal::expt::getTensorDim<ARGS>() == DIM
          ? internal::expt::getTensorSize<ARGS>(
                args, layout.template get_dim_size<
                          GetTensorArgIdx<DIM, ARGS...>::value>())
          : 0 ...);
}
#endif


namespace detail
{

/*!
 * Provides conversion of view data to a return type.
 *
 * For scalars, this just returns the scalar.
 *
 * In the future development, this may return SIMD vectors or matrices using
 * class specializations.
 */
template<typename VecSeq,
         typename Args,
         typename ElementType,
         typename PointerType,
         typename LinIdx,
         typename LayoutType>
struct ViewReturnHelper;

/*
 * Specialization for Scalar return types
 */
template<typename... Args,
         typename ElementType,
         typename PointerType,
         typename LinIdx,
         typename LayoutType>
struct ViewReturnHelper<camp::idx_seq<>,
                        camp::list<Args...>,
                        ElementType,
                        PointerType,
                        LinIdx,
                        LayoutType>
{
  using return_type = ElementType&;

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr return_type make_return(
      LayoutType const& layout,
      PointerType const& data,
      Args const&... args)
  {
    return data[stripIndexType(layout(args...))];
  }
};


#if defined(RAJA_ENABLE_VECTORIZATION)
/*
 * Specialization for Tensor return types
 */
template<camp::idx_t VecHead,
         camp::idx_t... VecSeq,
         typename... Args,
         typename ElementType,
         typename PointerType,
         typename LinIdx,
         typename LayoutType>
struct ViewReturnHelper<camp::idx_seq<VecHead, VecSeq...>,
                        camp::list<Args...>,
                        ElementType,
                        PointerType,
                        LinIdx,
                        LayoutType>
{

  static constexpr camp::idx_t s_num_dims = sizeof...(VecSeq) + 1;

  // This is the stride-one dimensions w.r.t. the tensor not the View
  // For example:
  //  For a vector, s_stride_one_dim is either 0 (packed) or -1 (strided)
  //  For a matrix, s_stride_one_dim is either:
  //                 -1 neither row nor column are packed
  //                 0 rows are stride-one
  //                 1 columns are stride-one
  static constexpr camp::idx_t s_stride_one_dim = RAJA::max<camp::idx_t>(
      (GetTensorArgIdx<VecHead, Args...>::value == LayoutType::stride_one_dim
           ? VecHead
           : -1),
      (GetTensorArgIdx<VecSeq, Args...>::value == LayoutType::stride_one_dim
           ? VecSeq
           : -1)...);


  using tensor_reg_type =
      typename camp::at_v<camp::list<Args...>,
                          GetTensorArgIdx<0, Args...>::value>::tensor_type;
  using ref_type = internal::expt::TensorRef<ElementType*,
                                             LinIdx,
                                             internal::expt::TENSOR_MULTIPLE,
                                             s_num_dims,
                                             s_stride_one_dim>;
  using return_type =
      internal::expt::ET::TensorLoadStore<tensor_reg_type, ref_type>;

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr return_type make_return(
      LayoutType const& layout,
      PointerType const& data,
      Args const&... args)
  {

    return return_type(ref_type {
        // data pointer
        &data[0] +
            layout(internal::expt::isTensorIndex<Args>()
                       ? LinIdx {0}
                       : (LinIdx)stripIndexType(
                             internal::expt::stripTensorIndexByValue(args))...),
        // strides
        {(LinIdx)layout.template get_dim_stride<
             GetTensorArgIdx<VecHead, Args...>::value>(),
         (LinIdx)layout.template get_dim_stride<
             GetTensorArgIdx<VecSeq, Args...>::value>()...},
        // tile
        {// begin
         {(LinIdx)(get_tensor_args_begin<VecHead>(layout, args...)),
          (LinIdx)(get_tensor_args_begin<VecSeq>(layout, args...))...},

         // size
         {(LinIdx)get_tensor_args_size<VecHead>(layout, args...),
          (LinIdx)get_tensor_args_size<VecSeq>(layout, args...)...}}});
  }
};

/*
 * Specialization for Tensor return types and static layout types
 */
template<camp::idx_t VecHead,
         camp::idx_t... VecSeq,
         typename... INDEX_TYPES,
         typename ElementType,
         typename PointerType,
         typename LinIdx,
         LinIdx... RangeInts,
         LinIdx... SizeInts,
         LinIdx... StrideInts,
         typename DIM_LIST>
struct ViewReturnHelper<
    camp::idx_seq<VecHead, VecSeq...>,
    camp::list<RAJA::expt::StaticTensorIndex<INDEX_TYPES>...>,
    ElementType,
    PointerType,
    LinIdx,
    RAJA::detail::StaticLayoutBase_impl<LinIdx,
                                        camp::int_seq<LinIdx, RangeInts...>,
                                        camp::int_seq<LinIdx, SizeInts...>,
                                        camp::int_seq<LinIdx, StrideInts...>,
                                        DIM_LIST>>
{
  static constexpr camp::idx_t s_num_dims = sizeof...(VecSeq) + 1;

  using index_list = camp::list<RAJA::expt::StaticTensorIndex<INDEX_TYPES>...>;

  using range_seq  = camp::int_seq<LinIdx, RangeInts...>;
  using size_seq   = camp::int_seq<LinIdx, SizeInts...>;
  using stride_seq = camp::int_seq<LinIdx, StrideInts...>;
  using LayoutType = RAJA::detail::
      StaticLayoutBase_impl<LinIdx, range_seq, size_seq, stride_seq, DIM_LIST>;

  // This is the stride-one dimensions w.r.t. the tensor not the View
  // For example:
  //  For a vector, s_stride_one_dim is either 0 (packed) or -1 (strided)
  //  For a matrix, s_stride_one_dim is either:
  //                 -1 neither row nor column are packed
  //                 0 rows are stride-one
  //                 1 columns are stride-one
  static constexpr camp::idx_t s_stride_one_dim = RAJA::max<camp::idx_t>(
      (GetTensorArgIdx<VecHead, index_list>::value == LayoutType::stride_one_dim
           ? VecHead
           : -1),
      (GetTensorArgIdx<VecSeq, index_list>::value == LayoutType::stride_one_dim
           ? VecSeq
           : -1)...);


  using new_begin_seq =
      camp::int_seq<LinIdx,
                    (LinIdx)get_tensor_args_begin<VecHead>(
                        LayoutType(),
                        RAJA::expt::StaticTensorIndex<INDEX_TYPES>()...),
                    (LinIdx)get_tensor_args_begin<VecSeq>(
                        LayoutType(),
                        RAJA::expt::StaticTensorIndex<INDEX_TYPES>()...)...>;
  using new_size_seq =
      camp::int_seq<LinIdx,
                    (LinIdx)get_tensor_args_size<VecHead>(
                        LayoutType(),
                        RAJA::expt::StaticTensorIndex<INDEX_TYPES>()...),
                    (LinIdx)get_tensor_args_size<VecSeq>(
                        LayoutType(),
                        RAJA::expt::StaticTensorIndex<INDEX_TYPES>()...)...>;

  using new_begin_type = internal::expt::StaticIndexArray<new_begin_seq>;
  using new_size_type  = internal::expt::StaticIndexArray<new_size_seq>;


  using tensor_reg_type =
      typename camp::at_v<index_list,
                          GetTensorArgIdx<0, index_list>::value>::tensor_type;
  using ref_type =
      internal::expt::StaticTensorRef<ElementType*,
                                      LinIdx,
                                      internal::expt::TENSOR_MULTIPLE,
                                      stride_seq,
                                      new_begin_seq,
                                      new_size_seq,
                                      s_stride_one_dim>;
  using return_type =
      internal::expt::ET::TensorLoadStore<tensor_reg_type, ref_type>;

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr return_type make_return(
      LayoutType const& layout,
      PointerType const& data,
      RAJA::expt::StaticTensorIndex<INDEX_TYPES> const&... args)
  {

    return return_type(ref_type {
        // data pointer
        &data[0] +
            layout(internal::expt::isTensorIndex<
                       typename RAJA::expt::StaticTensorIndex<
                           INDEX_TYPES>::base_type>()
                       ? LinIdx {0}
                       : (LinIdx)stripIndexType(
                             internal::expt::stripTensorIndexByValue(args))...),
        // strides
        typename ref_type::stride_type(),
        // tile
        {new_begin_type(), new_size_type()}});
  }
};
#endif


}  // namespace detail

/*
 * Computes the return type of a view.
 *
 * If any of the arguments are a VectorIndex, it creates a VectorRef
 * return type.
 *
 * Otherwise it produces the usual scalar reference return type
 */
template<typename ElementType,
         typename PointerType,
         typename LinIdx,
         typename LayoutType,
         typename... Args>
using view_return_type_t = typename detail::ViewReturnHelper<
    camp::make_idx_seq_t<count_num_tensor_args<Args...>::value>,
    camp::list<Args...>,
    ElementType,
    PointerType,
    LinIdx,
    LayoutType>::return_type;

/*
 * Creates the return value for a View
 *
 * If any of the arguments are a VectorIndex, it creates a VectorRef
 * return value.
 *
 * Otherwise it produces the usual scalar reference return value
 */
template<typename ElementType,
         typename LinIdx,
         typename LayoutType,
         typename PointerType,
         typename... Args>
RAJA_INLINE RAJA_HOST_DEVICE constexpr view_return_type_t<ElementType,
                                                          PointerType,
                                                          LinIdx,
                                                          LayoutType,
                                                          Args...>
view_make_return_value(LayoutType const& layout,
                       PointerType const& data,
                       Args const&... args)
{
  return detail::ViewReturnHelper<
      camp::make_idx_seq_t<count_num_tensor_args<Args...>::value>,
      camp::list<Args...>, ElementType, PointerType, LinIdx,
      LayoutType>::make_return(layout, data, args...);
}

namespace detail
{

/**
 * This class will help strip strongly typed indices
 *
 * This default implementation static_asserts that Expected==Arg, otherwise
 * it's an error.  This enforces types for the TypedView.
 *
 * Specialization where expected type is same as argument type.
 * In this case, there is no VectorIndex to unpack, just strip any strongly
 * typed indices.
 */
template<typename Expected, typename Arg>
struct MatchTypedViewArgHelper
{
  static_assert(std::is_convertible<strip_index_type_t<Arg>,
                                    strip_index_type_t<Expected>>::value,
                "Argument isn't compatible");

  using type = strip_index_type_t<Arg>;

  static RAJA_HOST_DEVICE RAJA_INLINE constexpr type extract(Arg arg)
  {
    return stripIndexType(arg);
  }
};


#if defined(RAJA_ENABLE_VECTORIZATION)
/**
 * Specialization where expected type is wrapped in a VectorIndex type
 *
 * In this case, there is no VectorIndex to unpack, just strip any strongly
 * typed indices.
 */
template<typename Expected, typename Arg, typename VectorType, camp::idx_t DIM>
struct MatchTypedViewArgHelper<Expected,
                               RAJA::expt::TensorIndex<Arg, VectorType, DIM>>
{

  static_assert(std::is_convertible<strip_index_type_t<Arg>,
                                    strip_index_type_t<Expected>>::value,
                "Argument isn't compatible");

  using arg_type = strip_index_type_t<Arg>;

  using type = RAJA::expt::TensorIndex<arg_type, VectorType, DIM>;

  static constexpr RAJA_HOST_DEVICE RAJA_INLINE type
  extract(RAJA::expt::TensorIndex<Arg, VectorType, DIM> vec_arg)
  {
    return type(stripIndexType(*vec_arg), vec_arg.size());
  }
};

/**
 * Specialization where expected type is wrapped in a StaticTensorIndex type
 *
 * In this case, there is no StaticTensorIndex to unpack, just strip any
 * strongly typed indices.
 */
template<typename Expected,
         typename Arg,
         typename VectorType,
         camp::idx_t DIM,
         Arg BEGIN,
         strip_index_type_t<Arg> LENGTH>
struct MatchTypedViewArgHelper<
    Expected,
    RAJA::expt::StaticTensorIndex<
        RAJA::expt::
            StaticTensorIndexInner<Arg, VectorType, DIM, BEGIN, LENGTH>>>
{

  static_assert(std::is_convertible<strip_index_type_t<Arg>,
                                    strip_index_type_t<Expected>>::value,
                "Argument isn't compatible");

  using arg_type = strip_index_type_t<Arg>;

  using type = RAJA::expt::StaticTensorIndex<
      RAJA::expt::
          StaticTensorIndexInner<arg_type, VectorType, DIM, BEGIN, LENGTH>>;

  static constexpr RAJA_HOST_DEVICE RAJA_INLINE type extract(
      RAJA::expt::StaticTensorIndex<
          RAJA::expt::
              StaticTensorIndexInner<Arg, VectorType, DIM, BEGIN, LENGTH>>
          RAJA_UNUSED_ARG(vec_arg))
  {
    return type();
  }
};
#endif

}  // namespace detail

template<typename Expected, typename Arg>
RAJA_HOST_DEVICE RAJA_INLINE constexpr
    typename detail::MatchTypedViewArgHelper<Expected, Arg>::type
    match_typed_view_arg(Arg const& arg)
{
  return detail::MatchTypedViewArgHelper<Expected, Arg>::extract(arg);
}

template<typename ValueType, typename PointerType, typename LayoutType>
class ViewBase
{

public:
  using value_type        = ValueType;
  using pointer_type      = PointerType;
  using layout_type       = LayoutType;
  using linear_index_type = typename layout_type::IndexLinear;
  using nc_value_type     = typename std::remove_const<value_type>::type;
  using nc_pointer_type = typename std::add_pointer<typename std::remove_const<
      typename std::remove_pointer<pointer_type>::type>::type>::type;

  using Self         = ViewBase<value_type, pointer_type, layout_type>;
  using NonConstView = ViewBase<nc_value_type, nc_pointer_type, layout_type>;

  using shifted_layout_type = typename add_offset<layout_type>::type;
  using ShiftedView = ViewBase<value_type, pointer_type, shifted_layout_type>;

protected:
  pointer_type m_data;
  layout_type const m_layout;

public:
  /*
   * Defaulted operators (AJK):
   *
   * OpenMP Target currently needs the View classes to be trivially copyable,
   * which means that we need to use the default ctor's and assignment
   * operators.
   *
   * These defaulted operators cause issues with some versions of CUDA, so
   * in the case that CUDA is enabled, we switch to explicitly defined
   * operators.
   */
#if (defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_CLANG_CUDA))
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr ViewBase() {};

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ViewBase(ViewBase const& c)
      : m_layout(c.m_layout),
        m_data(c.m_data)
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr ViewBase& operator=(ViewBase const& c)
  {
    m_layout = c.m_layout;
    m_data   = c.m_data;
  }
#else
  constexpr ViewBase()                                       = default;
  RAJA_INLINE constexpr ViewBase(ViewBase const&)            = default;
  RAJA_INLINE constexpr ViewBase(ViewBase&&)                 = default;
  RAJA_INLINE constexpr ViewBase& operator=(ViewBase const&) = default;
  RAJA_INLINE constexpr ViewBase& operator=(ViewBase&&)      = default;

#endif

  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr ViewBase(pointer_type data, layout_type&& layout)
      : m_data(data),
        m_layout(layout)
  {}

  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr ViewBase(pointer_type data,
                                                  Args... dim_sizes)
      : m_data(data),
        m_layout(dim_sizes...)
  {}

  template<bool IsConstView = std::is_const<value_type>::value>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr ViewBase(
      typename std::enable_if<IsConstView, NonConstView>::type const& rhs)
      : m_data(rhs.get_data()),
        m_layout(rhs.get_layout())
  {}

  RAJA_HOST_DEVICE RAJA_INLINE constexpr void set_data(PointerType data_ptr)
  {
    m_data = data_ptr;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr pointer_type const& get_data() const
  {
    return m_data;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr layout_type const& get_layout() const
  {
    return m_layout;
  }

  RAJA_HOST_DEVICE RAJA_INLINE constexpr linear_index_type size() const
  {
    return m_layout.size();
  }

  template<camp::idx_t DIM>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr linear_index_type get_dim_size() const
  {
    return m_layout.template get_dim_size<DIM>();
  }

  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr view_return_type_t<value_type,
                                                            pointer_type,
                                                            linear_index_type,
                                                            layout_type,
                                                            Args...>
  operator()(Args... args) const
  {
    return view_make_return_value<value_type, linear_index_type>(
        m_layout, m_data, args...);
  }

  /*
   * Compatibility note (AJK):
   * We are using variadic arguments even though operator[] takes exactly 1
   * argument This gets around a template instantiation bug in CUDA/nvcc 9.1,
   * which seems to have been fixed in CUDA 9.2+
   */
  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr view_return_type_t<value_type,
                                                            pointer_type,
                                                            linear_index_type,
                                                            layout_type,
                                                            Args...>
  operator[](Args... args) const
  {
    return view_make_return_value<value_type, linear_index_type>(
        m_layout, m_data, args...);
  }

  // why doesn't this return a shifted copy instead of shifting this view?
  template<size_t n_dims   = layout_type::n_dims,
           typename IdxLin = linear_index_type>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr ShiftedView shift(
      const std::array<IdxLin, n_dims>& shift)
  {
    static_assert(n_dims == layout_type::n_dims,
                  "Dimension mismatch in ViewBase shift");

    shifted_layout_type shift_layout(m_layout);
    shift_layout.shift(shift);

    return ShiftedView(m_data, shift_layout);
  }
};


template<typename ValueType,
         typename PointerType,
         typename LayoutType,
         typename IndexTypes>
class TypedViewBase;

template<typename ValueType,
         typename PointerType,
         typename LayoutType,
         typename... IndexTypes>
class TypedViewBase<ValueType,
                    PointerType,
                    LayoutType,
                    camp::list<IndexTypes...>>
    : public ViewBase<ValueType, PointerType, LayoutType>
{

public:
  using value_type        = ValueType;
  using pointer_type      = PointerType;
  using layout_type       = LayoutType;
  using linear_index_type = typename layout_type::IndexLinear;
  using nc_value_type     = typename std::remove_const<value_type>::type;
  using nc_pointer_type = typename std::add_pointer<typename std::remove_const<
      typename std::remove_pointer<pointer_type>::type>::type>::type;

  using Base         = ViewBase<ValueType, PointerType, LayoutType>;
  using Self         = TypedViewBase<value_type,
                             pointer_type,
                             layout_type,
                             camp::list<IndexTypes...>>;
  using NonConstView = TypedViewBase<nc_value_type,
                                     nc_pointer_type,
                                     layout_type,
                                     camp::list<IndexTypes...>>;

  using shifted_layout_type = typename add_offset<layout_type>::type;
  using ShiftedView         = TypedViewBase<value_type,
                                    pointer_type,
                                    shifted_layout_type,
                                    camp::list<IndexTypes...>>;

  static constexpr size_t n_dims = sizeof...(IndexTypes);

  using Base::Base;

  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr view_return_type_t<value_type,
                                                            pointer_type,
                                                            linear_index_type,
                                                            layout_type,
                                                            Args...>
  operator()(Args... args) const
  {
    return view_make_return_value<value_type, linear_index_type>(
        Base::m_layout, Base::m_data,
        match_typed_view_arg<IndexTypes>(args)...);
  }

  /*
   * Compatibility note (AJK):
   * We are using variadic arguments even though operator[] takes exactly 1
   * argument This gets around a template instantiation bug in CUDA/nvcc 9.1,
   * which seems to have been fixed in CUDA 9.2+
   */
  template<typename... Args>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr view_return_type_t<value_type,
                                                            pointer_type,
                                                            linear_index_type,
                                                            layout_type,
                                                            Args...>
  operator[](Args... args) const
  {
    return view_make_return_value<value_type, linear_index_type>(
        Base::m_layout, Base::m_data,
        match_typed_view_arg<IndexTypes>(args)...);
  }

  // why doesn't this return a shifted copy instead of shifting this view?
  template<size_t n_dims   = sizeof...(IndexTypes),
           typename IdxLin = linear_index_type>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr ShiftedView shift(
      const std::array<IdxLin, n_dims>& shift)
  {
    static_assert(n_dims == layout_type::n_dims,
                  "Dimension mismatch in TypedViewBase shift");

    shifted_layout_type shift_layout(Base::get_layout());
    shift_layout.shift(shift);

    return ShiftedView(Base::get_data(), shift_layout);
  }
};


}  // namespace internal

}  // namespace RAJA

#endif
