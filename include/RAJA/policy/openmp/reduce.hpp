/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_omp_reduce_HPP
#define RAJA_omp_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <memory>
#include <vector>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/detail/reduce.hpp"
#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

namespace detail
{
template<typename T, typename Reduce>
class ReduceOMP
    : public reduce::detail::BaseCombinable<T, Reduce, ReduceOMP<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceOMP>;

public:
  using Base::Base;
  //! prohibit compiler-generated default ctor
  ReduceOMP() = delete;

  ~ReduceOMP()
  {
    if (Base::parent)
    {
#pragma omp critical(ompReduceCritical)
      Reduce()(Base::parent->local(), Base::my_data);
      Base::my_data = Base::identity;
    }
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(omp_reduce, detail::ReduceOMP)

///////////////////////////////////////////////////////////////////////////////
//
// Old ordered reductions are included below.
//
///////////////////////////////////////////////////////////////////////////////

namespace detail
{
template<typename T, typename Reduce>
class ReduceOMPOrdered
    : public reduce::detail::
          BaseCombinable<T, Reduce, ReduceOMPOrdered<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceOMPOrdered>;
  std::shared_ptr<std::vector<T>> data;

public:
  ReduceOMPOrdered() { reset(T(), T()); }

  //! constructor requires a default value for the reducer
  explicit ReduceOMPOrdered(T init_val, T identity_)
  {
    reset(init_val, identity_);
  }

  void reset(T init_val, T identity_)
  {
    Base::reset(init_val, identity_);
    data = std::shared_ptr<std::vector<T>>(
        std::make_shared<std::vector<T>>(omp_get_max_threads(), identity_));
  }

  ~ReduceOMPOrdered()
  {
    Reduce {}((*data)[omp_get_thread_num()], Base::my_data);
    Base::my_data = Base::identity;
  }

  T get_combined() const
  {
    if (Base::my_data != Base::identity)
    {
      Reduce {}((*data)[omp_get_thread_num()], Base::my_data);
      Base::my_data = Base::identity;
    }

    T res = Base::identity;
    for (size_t i = 0; i < data->size(); ++i)
    {
      Reduce {}(res, (*data)[i]);
    }
    return res;
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(omp_reduce_ordered, detail::ReduceOMPOrdered)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
