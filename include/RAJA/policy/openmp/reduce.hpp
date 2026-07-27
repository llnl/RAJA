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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
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
class ReduceOMP  // This is a Combinable and is the first layer of that
                 // implementation detail used in Reducers
    : public reduce::detail::BaseCombinable<T, Reduce, ReduceOMP<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceOMP>;

public:
  using Base::Base;
  using Base::reset;

  ReduceOMP(Policy p, T init_val, T identity_) : Base(init_val, identity_)
  {
    policy_supported_or_throw(
        "OpenMPReduce", reduction_supported_policies_t<Policy::openmp> {}, p);
  }

  void reset(Policy p, T init_val, T identity_)
  {
    policy_supported_or_throw("OpenMPReduce::reset",
                              reduction_supported_policies_t<Policy::openmp> {},
                              p);
    Base::reset(init_val, identity_);
  }

  ~ReduceOMP()
  {
    if (Base::parent && Base::my_data != Base::identity)
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
class ReduceOMPOrdered  // This is a Combinable and is the first layer of that
                        // implementation detail used in Reducers
    : public reduce::detail::
          BaseCombinable<T, Reduce, ReduceOMPOrdered<T, Reduce>>
{
  using Base = reduce::detail::BaseCombinable<T, Reduce, ReduceOMPOrdered>;
  std::shared_ptr<std::vector<T>> data;

public:
  ReduceOMPOrdered(Policy p, T init_val, T identity_)
      : ReduceOMPOrdered(init_val, identity_)
  {
    policy_supported_or_throw("OpenMPReduceOrdered",
                              reduction_supported_policies_t<Policy::openmp> {},
                              p);
  }

  ReduceOMPOrdered(T init_val, T identity_)
      : Base(init_val, identity_),
        data(std::make_shared<std::vector<T>>(omp_get_max_threads(), identity_))
  {}

  void reset(T init_val, T identity_)
  {
    Base::reset(init_val, identity_);
    for (T& data_i : *data)
    {
      data_i = Base::identity;
    }
  }

  void reset(Policy p, T init_val, T identity_)
  {
    policy_supported_or_throw("OpenMPReduceOrdered::reset",
                              reduction_supported_policies_t<Policy::openmp> {},
                              p);
    reset(init_val, identity_);
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
    for (T const& data_i : *data)
    {
      Reduce {}(res, data_i);
    }
    return res;
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_REDUCERS(omp_reduce_ordered, detail::ReduceOMPOrdered)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
