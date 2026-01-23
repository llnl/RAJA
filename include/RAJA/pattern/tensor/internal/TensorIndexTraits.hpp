/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_TensorIndexTraits_HPP
#define RAJA_pattern_tensor_TensorIndexTraits_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/TensorIndex.hpp"

namespace RAJA
{

namespace internal
{
/* Partial specialization for the strip_index_type_t helper in
   IndexValue.hpp
*/
template<typename IDX, typename VECTOR_TYPE, camp::idx_t DIM>
struct StripIndexTypeT<RAJA::expt::TensorIndex<IDX, VECTOR_TYPE, DIM>>
{
  using type =
      typename RAJA::expt::TensorIndex<IDX, VECTOR_TYPE, DIM>::value_type;
};

namespace expt
{


// Helper that strips the Vector type from an argument
template<typename ARG>
struct TensorIndexTraits
{
  using arg_type   = ARG;
  using value_type = strip_index_type_t<ARG>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool isTensorIndex() { return false; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr arg_type const& strip(arg_type const& arg) { return arg; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr arg_type const strip_by_value(arg_type const arg)
  {
    return arg;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type size(arg_type const&) { return 1; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type begin(arg_type const&) { return 0; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type dim() { return 0; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type num_elem() { return 1; }
};

template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
struct TensorIndexTraits<RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>>
{
  using index_type = RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>;
  using arg_type   = IDX;
  using value_type = strip_index_type_t<IDX>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool isTensorIndex() { return true; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr arg_type const& strip(index_type const& arg) { return *arg; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr arg_type const strip_by_value(index_type const arg)
  {
    return (arg_type)arg;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type size(index_type const& arg) { return arg.size(); }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type begin(index_type const& arg)
  {
    return arg.begin();
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type dim() { return DIM; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type num_elem()
  {
    return TENSOR_TYPE::s_dim_elem(DIM);
  }
};

template<typename IDX,
         typename TENSOR_TYPE,
         camp::idx_t DIM,
         IDX INDEX_VALUE,
         strip_index_type_t<IDX> LENGTH_VALUE>
struct TensorIndexTraits<RAJA::expt::StaticTensorIndex<
    RAJA::expt::StaticTensorIndexInner<IDX,
                                       TENSOR_TYPE,
                                       DIM,
                                       INDEX_VALUE,
                                       LENGTH_VALUE>>>
{
  using base_type  = RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>;
  using index_type = RAJA::expt::StaticTensorIndex<
      RAJA::expt::StaticTensorIndexInner<IDX,
                                         TENSOR_TYPE,
                                         DIM,
                                         INDEX_VALUE,
                                         LENGTH_VALUE>>;
  using arg_type   = IDX;
  using value_type = strip_index_type_t<IDX>;

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr bool isTensorIndex() { return true; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr arg_type const strip_by_value(index_type const)
  {
    return INDEX_VALUE;
  }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type size(index_type const&) { return LENGTH_VALUE; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type begin(index_type const&) { return INDEX_VALUE; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type dim() { return DIM; }

  RAJA_INLINE

  RAJA_HOST_DEVICE
  static constexpr value_type num_elem()
  {
    return TENSOR_TYPE::s_dim_elem(DIM);
  }
};

/*
 * Returns vector size of argument.
 *
 * For scalars, always returns 1.
 *
 * For VectorIndex types, returns the number of vector lanes.
 */
template<typename ARG>
RAJA_INLINE RAJA_HOST_DEVICE constexpr bool isTensorIndex()
{
  return TensorIndexTraits<ARG>::isTensorIndex();
}

template<typename ARG>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto stripTensorIndex(ARG const& arg) ->
    typename TensorIndexTraits<ARG>::arg_type const&
{
  return TensorIndexTraits<ARG>::strip(arg);
}

template<typename ARG>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto stripTensorIndexByValue(
    ARG const arg) -> typename TensorIndexTraits<ARG>::arg_type const
{
  return TensorIndexTraits<ARG>::strip_by_value(arg);
}

/*
 * Returns tensor dimension size of argument.
 *
 * For VectorIndex types, returns the number of vector lanes.
 */
template<typename ARG, typename IDX>
RAJA_INLINE RAJA_HOST_DEVICE constexpr IDX getTensorSize(ARG const& arg,
                                                         IDX dim_size)
{
  return TensorIndexTraits<ARG>::size(arg) >= 0
             ? IDX(TensorIndexTraits<ARG>::size(arg))
             : dim_size;
}

/*
 * Returns tensor dimenson beginning index of an argument.
 *
 */
template<typename ARG, typename IDX>
RAJA_INLINE RAJA_HOST_DEVICE constexpr IDX getTensorBegin(ARG const& arg,
                                                          IDX dim_minval)
{
  return TensorIndexTraits<ARG>::begin(arg) >= 0
             ? IDX(TensorIndexTraits<ARG>::begin(arg))
             : dim_minval;
}

/*
 * Returns vector dim of argument.
 *
 * For scalars, always returns 0.
 *
 * For VectorIndex types, returns the DIM argument.
 * For vector_exec, this is always 0
 *
 * For matrices, DIM means:
 *   0 : Row
 *   1 : Column
 */
template<typename ARG>
RAJA_INLINE RAJA_HOST_DEVICE constexpr auto getTensorDim()
    -> decltype(TensorIndexTraits<ARG>::dim())
{
  return TensorIndexTraits<ARG>::dim();
}

}  // namespace expt

/*
 * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
 * includes the vector length with them
 */
template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t id>
struct LambdaSegExtractor<RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>, id>
{

  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static RAJA::expt::
      TensorIndex<IDX, TENSOR_TYPE, DIM>
      extract(Data&& data)
  {
    return RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>(
        camp::get<id>(data.segment_tuple)
            .begin()[camp::get<id>(data.offset_tuple)],
        camp::get<id>(data.vector_sizes));
  }
};

/*
 * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
 * includes the vector length with them
 */
template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t id>
struct LambdaOffsetExtractor<RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>, id>
{

  template<typename Data>
  RAJA_HOST_DEVICE RAJA_INLINE constexpr static RAJA::expt::
      TensorIndex<IDX, TENSOR_TYPE, DIM>
      extract(Data&& data)
  {
    return RAJA::expt::TensorIndex<IDX, TENSOR_TYPE, DIM>(
        IDX(camp::get<id>(data.offset_tuple)),  // convert offset type to IDX
        camp::get<id>(data.vector_sizes));
  }
};

}  // namespace internal
}  // namespace RAJA


#endif
