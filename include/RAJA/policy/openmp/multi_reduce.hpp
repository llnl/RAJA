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

#ifndef RAJA_omp_multi_reduce_HPP
#define RAJA_omp_multi_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_OPENMP)

#include <memory>
#include <vector>

#include <omp.h>

#include "RAJA/util/types.hpp"
#include "RAJA/util/reduce.hpp"
#include "RAJA/util/RepeatView.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/multi_reduce.hpp"
#include "RAJA/pattern/multi_reduce.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

namespace detail
{

/*!
 **************************************************************************
 *
 * \brief  OMP multi-reduce data class template.
 *
 * In this class memory is owned by the parent object
 *
 **************************************************************************
 */
template<typename T, typename t_MultiReduceOp, typename tuning>
struct MultiReduceDataOMP;

/*!
 **************************************************************************
 *
 * \brief  OMP multi-reduce data class template using combine on destruction.
 *
 * In this class memory is owned by each copy of the object
 *
 **************************************************************************
 */
template<typename T, typename t_MultiReduceOp>
struct MultiReduceDataOMP<
    T,
    t_MultiReduceOp,
    RAJA::omp::MultiReduceTuning<
        RAJA::omp::multi_reduce_algorithm::combine_on_destruction>>
{
  using value_type    = T;
  using MultiReduceOp = t_MultiReduceOp;

  MultiReduceDataOMP() = delete;

  template<typename Container>
  MultiReduceDataOMP(Policy p, Container const& container, T identity)
      : m_parent(nullptr),
        m_num_bins(container.size()),
        m_data(nullptr),
        m_identity(identity),
        m_create_data_on_copy(p != Policy::sequential)
  {
    m_data = create_data(container, m_num_bins);
  }

  MultiReduceDataOMP(MultiReduceDataOMP const& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_num_bins(other.m_num_bins),
        m_data(nullptr),
        m_identity(other.m_identity),
        m_create_data_on_copy(other.m_create_data_on_copy)
  {
    if (m_create_data_on_copy)
    {
      m_data = create_data(
          RepeatView<value_type>(other.m_identity, other.m_num_bins),
          other.m_num_bins);
    }
    else
    {
      m_data = other.m_data;
    }
  }

  MultiReduceDataOMP(MultiReduceDataOMP&&)                 = delete;
  MultiReduceDataOMP& operator=(MultiReduceDataOMP const&) = delete;
  MultiReduceDataOMP& operator=(MultiReduceDataOMP&&)      = delete;

  ~MultiReduceDataOMP()
  {
    if (m_data)
    {
      if (m_parent && m_create_data_on_copy)
      {
        if (m_num_bins != size_t(0))
        {
#pragma omp critical(ompMultiReduceCritical)
          {
            for (size_t bin = 0; bin < m_num_bins; ++bin)
            {
              MultiReduceOp {}(m_parent->m_data[bin], m_data[bin]);
            }
          }
        }
      }
      if (!m_parent || m_create_data_on_copy)
      {
        destroy_data(m_data, m_num_bins);
      }
    }
  }

  template<typename Container>
  void reset(Container const& container, T identity)
  {
    m_identity          = identity;
    size_t new_num_bins = container.size();
    if (new_num_bins != m_num_bins)
    {
      destroy_data(m_data, m_num_bins);
      m_num_bins = new_num_bins;
      m_data     = create_data(container, new_num_bins);
    }
    else
    {
      size_t bin = 0;
      for (auto const& value : container)
      {
        m_data[bin] = value;
        ++bin;
      }
    }
  }

  template<typename Container>
  void reset(Policy p, Container const& container, T identity)
  {
    reset(container, identity);
    m_create_data_on_copy = (p != Policy::sequential);
  }

  size_t num_bins() const { return m_num_bins; }

  T identity() const { return m_identity; }

  void combine(size_t bin, T const& val) { MultiReduceOp {}(m_data[bin], val); }

  T get(size_t bin) const { return m_data[bin]; }

private:
  MultiReduceDataOMP const* m_parent;
  size_t m_num_bins;
  T* m_data;
  T m_identity;
  bool m_create_data_on_copy;

  template<typename Container>
  static T* create_data(Container const& container, size_t num_bins)
  {
    if (num_bins == size_t(0))
    {
      return nullptr;
    }
    auto data =
        RAJA::allocate_aligned_type<T>(RAJA::DATA_ALIGN, num_bins * sizeof(T));
    size_t bin = 0;
    for (auto const& value : container)
    {
      new (&data[bin]) T(value);
      ++bin;
    }
    return data;
  }

  static void destroy_data(T*& data, size_t num_bins)
  {
    if (num_bins == size_t(0))
    {
      return;
    }
    for (size_t bin = num_bins; bin > 0; --bin)
    {
      data[bin - 1].~T();
    }
    RAJA::free_aligned(data);
    data = nullptr;
  }
};

/*!
 **************************************************************************
 *
 * \brief  OMP multi-reduce data class template using combine on get.
 *
 * In this class memory is owned by each copy of the object
 *
 **************************************************************************
 */
template<typename T, typename t_MultiReduceOp>
struct MultiReduceDataOMP<
    T,
    t_MultiReduceOp,
    RAJA::omp::MultiReduceTuning<
        RAJA::omp::multi_reduce_algorithm::combine_on_get>>
{
  using value_type    = T;
  using MultiReduceOp = t_MultiReduceOp;

  MultiReduceDataOMP() = delete;

  template<typename Container>
  MultiReduceDataOMP(Policy p, Container const& container, T identity)
      : m_parent(nullptr),
        m_data_helper(p, container.size()),
        m_data(nullptr),
        m_identity(identity)
  {
    policy_supported_or_throw("OpenMPMultiReduce",
                              reduction_supported_policies_t<Policy::openmp> {},
                              p);
    m_data = create_data(container, identity, m_data_helper);
  }

  MultiReduceDataOMP(MultiReduceDataOMP const& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_data_helper(other.m_data_helper),
        m_data(other.m_data),
        m_identity(other.m_identity)
  {}

  MultiReduceDataOMP(MultiReduceDataOMP&&)                 = delete;
  MultiReduceDataOMP& operator=(MultiReduceDataOMP const&) = delete;
  MultiReduceDataOMP& operator=(MultiReduceDataOMP&&)      = delete;

  ~MultiReduceDataOMP()
  {
    if (m_data)
    {
      if (!m_parent)
      {
        destroy_data(m_data, m_data_helper);
      }
    }
  }

  template<typename Container>
  void reset(Container const& container, T identity)
  {
    DataHelper new_data_helper = m_data_helper;
    new_data_helper.reset(container.size());

    reset_impl(container, identity, new_data_helper);
  }

  template<typename Container>
  void reset(Policy p, Container const& container, T identity)
  {
    policy_supported_or_throw("OpenMPMultiReduce::reset",
                              reduction_supported_policies_t<Policy::openmp> {},
                              p);

    DataHelper new_data_helper(p, container.size());

    reset_impl(container, identity, new_data_helper);
  }

  size_t num_bins() const { return m_data_helper.m_num_bins; }

  T identity() const { return m_identity; }

  void combine(size_t bin, T const& val)
  {
    size_t thread_idx =
        (m_data_helper.m_max_threads > 1) ? omp_get_thread_num() : 0;
    MultiReduceOp {}(m_data[m_data_helper.index(bin, thread_idx)], val);
  }

  T get(size_t bin) const
  {
    ::RAJA::HighAccuracyReduce<T, typename MultiReduceOp::operator_type>
        reducer(m_identity);
    for (size_t thread_idx = 0; thread_idx < m_data_helper.m_max_threads;
         ++thread_idx)
    {
      reducer.combine(m_data[m_data_helper.index(bin, thread_idx)]);
    }
    return reducer.get_and_reset();
  }

private:
  struct DataHelper
  {
    static size_t get_max_threads(Policy p)
    {
      if (policy_supported(PolicyList<Policy::openmp> {}, p))
      {
        return omp_get_max_threads();
      }
      return 1;
    }

    static constexpr size_t pad_bins(size_t num_bins)
    {
      size_t num_cache_lines =
          RAJA_DIVIDE_CEILING_INT(num_bins * sizeof(T), RAJA::DATA_ALIGN);
      return RAJA_DIVIDE_CEILING_INT(num_cache_lines * RAJA::DATA_ALIGN,
                                     sizeof(T));
    }

    static constexpr size_t pad_threads(size_t max_threads)
    {
      return max_threads;
    }

    size_t m_max_threads;
    size_t m_num_bins;
    size_t m_padded_threads;
    size_t m_padded_bins;

    DataHelper(Policy p, size_t num_bins)
        : m_max_threads(get_max_threads(p)),
          m_num_bins(num_bins),
          m_padded_threads(pad_threads(m_max_threads)),
          m_padded_bins(pad_bins(num_bins))
    {}

    constexpr void reset(size_t num_bins)
    {
      m_num_bins    = num_bins;
      m_padded_bins = pad_bins(num_bins);
    }

    constexpr size_t index(size_t bin, size_t thread_idx) const
    {
      return bin + thread_idx * m_padded_bins;
    }

    bool operator==(DataHelper const&) const = default;
  };

  MultiReduceDataOMP const* m_parent;
  DataHelper m_data_helper;
  T* m_data;
  T m_identity;

  template<typename Container>
  void reset_impl(Container const& container,
                  T identity,
                  DataHelper new_data_helper)
  {
    m_identity = identity;

    if (new_data_helper != m_data_helper)
    {
      destroy_data(m_data, m_data_helper);
      m_data_helper = new_data_helper;
      m_data        = create_data(container, identity, new_data_helper);
    }
    else
    {
      if (m_data)
      {
        {
          size_t thread_idx = 0;
          size_t bin        = 0;
          for (auto const& value : container)
          {
            m_data[m_data_helper.index(bin, thread_idx)] = value;
            ++bin;
          }
        }
        for (size_t thread_idx = 1; thread_idx < m_data_helper.m_max_threads;
             ++thread_idx)
        {
          for (size_t bin = 0; bin < m_data_helper.m_num_bins; ++bin)
          {
            m_data[m_data_helper.index(bin, thread_idx)] = identity;
          }
        }
      }
    }
  }

  template<typename Container>
  static T* create_data(Container const& container,
                        T identity,
                        DataHelper const& data_helper)
  {
    if (data_helper.m_num_bins == size_t(0))
    {
      return nullptr;
    }
    auto data = RAJA::allocate_aligned_type<T>(
        RAJA::DATA_ALIGN,
        data_helper.m_padded_threads * data_helper.m_padded_bins * sizeof(T));
    if (data_helper.m_max_threads > 0)
    {
      {
        size_t thread_idx = 0;
        size_t bin        = 0;
        for (auto const& value : container)
        {
          new (&data[data_helper.index(bin, thread_idx)]) T(value);
          ++bin;
        }
      }
      for (size_t thread_idx = 1; thread_idx < data_helper.m_max_threads;
           ++thread_idx)
      {
        for (size_t bin = 0; bin < data_helper.m_num_bins; ++bin)
        {
          new (&data[data_helper.index(bin, thread_idx)]) T(identity);
        }
      }
    }
    return data;
  }

  static void destroy_data(T*& data, DataHelper const& data_helper)
  {
    if (data_helper.m_num_bins == size_t(0))
    {
      return;
    }
    for (size_t thread_idx = data_helper.m_max_threads; thread_idx > 0;
         --thread_idx)
    {
      for (size_t bin = data_helper.m_num_bins; bin > 0; --bin)
      {
        data[data_helper.index(bin - 1, thread_idx - 1)].~T();
      }
    }
    RAJA::free_aligned(data);
    data = nullptr;
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_MULTI_REDUCERS(policy::omp::omp_multi_reduce_policy,
                                detail::MultiReduceDataOMP)

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
