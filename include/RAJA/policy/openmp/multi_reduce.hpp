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

  template<typename Container,
           std::enable_if_t<
               !std::is_same<Container, MultiReduceDataOMP>::value>* = nullptr>
  MultiReduceDataOMP(Container const& container, T identity)
      : m_parent(nullptr),
        m_num_bins(container.size()),
        m_identity(identity),
        m_data(nullptr)
  {
    m_data = create_data(container, m_num_bins);
  }

  MultiReduceDataOMP(MultiReduceDataOMP const& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_num_bins(other.m_num_bins),
        m_identity(other.m_identity),
        m_data(nullptr)
  {
    m_data =
        create_data(RepeatView<value_type>(other.m_identity, other.m_num_bins),
                    other.m_num_bins);
  }

  MultiReduceDataOMP(MultiReduceDataOMP&&)                 = delete;
  MultiReduceDataOMP& operator=(MultiReduceDataOMP const&) = delete;
  MultiReduceDataOMP& operator=(MultiReduceDataOMP&&)      = delete;

  ~MultiReduceDataOMP()
  {
    if (m_data)
    {
      if (m_parent && (m_num_bins != size_t(0)))
      {
#pragma omp critical(ompMultiReduceCritical)
        {
          for (size_t bin = 0; bin < m_num_bins; ++bin)
          {
            MultiReduceOp {}(m_parent->m_data[bin], m_data[bin]);
          }
        }
      }
      destroy_data(m_data, m_num_bins);
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
      m_data     = create_data(container, m_num_bins);
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

  size_t num_bins() const { return m_num_bins; }

  T identity() const { return m_identity; }

  void combine(size_t bin, T const& val) { MultiReduceOp {}(m_data[bin], val); }

  T get(size_t bin) const { return m_data[bin]; }

private:
  MultiReduceDataOMP const* m_parent;
  size_t m_num_bins;
  T m_identity;
  T* m_data;

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

  template<typename Container,
           std::enable_if_t<
               !std::is_same<Container, MultiReduceDataOMP>::value>* = nullptr>
  MultiReduceDataOMP(Container const& container, T identity)
      : m_parent(nullptr),
        m_max_threads(omp_get_max_threads()),
        m_num_bins(container.size()),
        m_padded_threads(pad_threads(m_max_threads)),
        m_padded_bins(pad_bins(m_num_bins)),
        m_identity(identity),
        m_data(nullptr)
  {
    m_data = create_data(container, identity, m_num_bins, m_max_threads,
                         m_padded_bins, m_padded_threads);
  }

  MultiReduceDataOMP(MultiReduceDataOMP const& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_num_bins(other.m_num_bins),
        m_padded_threads(other.m_padded_threads),
        m_padded_bins(other.m_padded_bins),
        m_identity(other.m_identity),
        m_data(other.m_data)
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
        destroy_data(m_data, m_num_bins, m_max_threads, m_padded_bins,
                     m_padded_threads);
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
      destroy_data(m_data, m_num_bins, m_max_threads, m_padded_bins,
                   m_padded_threads);
      m_num_bins    = new_num_bins;
      m_padded_bins = pad_bins(m_num_bins);
      m_data = create_data(container, identity, m_num_bins, m_max_threads,
                           m_padded_bins, m_padded_threads);
    }
    else
    {
      if (m_max_threads > 0)
      {
        {
          size_t thread_idx = 0;
          size_t bin        = 0;
          for (auto const& value : container)
          {
            m_data[index_data(bin, thread_idx, m_padded_bins,
                              m_padded_threads)] = value;
            ++bin;
          }
        }
        for (size_t thread_idx = 1; thread_idx < m_max_threads; ++thread_idx)
        {
          for (size_t bin = 0; bin < m_num_bins; ++bin)
          {
            m_data[index_data(bin, thread_idx, m_padded_bins,
                              m_padded_threads)] = identity;
          }
        }
      }
    }
  }

  size_t num_bins() const { return m_num_bins; }

  T identity() const { return m_identity; }

  void combine(size_t bin, T const& val)
  {
    size_t thread_idx = omp_get_thread_num();
    MultiReduceOp {}(
        m_data[index_data(bin, thread_idx, m_padded_bins, m_padded_threads)],
        val);
  }

  T get(size_t bin) const
  {
    ::RAJA::detail::HighAccuracyReduce<T, typename MultiReduceOp::operator_type>
        reducer(m_identity);
    for (size_t thread_idx = 0; thread_idx < m_max_threads; ++thread_idx)
    {
      reducer.combine(
          m_data[index_data(bin, thread_idx, m_padded_bins, m_padded_threads)]);
    }
    return reducer.get_and_clear();
  }

private:
  MultiReduceDataOMP const* m_parent;
  size_t m_max_threads;
  size_t m_num_bins;
  size_t m_padded_threads;
  size_t m_padded_bins;
  T m_identity;
  T* m_data;

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

  static constexpr size_t index_data(size_t bin,
                                     size_t thread_idx,
                                     size_t padded_bins,
                                     size_t RAJA_UNUSED_ARG(padded_threads))
  {
    return bin + thread_idx * padded_bins;
  }

  template<typename Container>
  static T* create_data(Container const& container,
                        T identity,
                        size_t num_bins,
                        size_t max_threads,
                        size_t padded_bins,
                        size_t padded_threads)
  {
    if (num_bins == size_t(0))
    {
      return nullptr;
    }
    auto data = RAJA::allocate_aligned_type<T>(
        RAJA::DATA_ALIGN, padded_threads * padded_bins * sizeof(T));
    if (max_threads > 0)
    {
      {
        size_t thread_idx = 0;
        size_t bin        = 0;
        for (auto const& value : container)
        {
          new (&data[index_data(bin, thread_idx, padded_bins, padded_threads)])
              T(value);
          ++bin;
        }
      }
      for (size_t thread_idx = 1; thread_idx < max_threads; ++thread_idx)
      {
        for (size_t bin = 0; bin < num_bins; ++bin)
        {
          new (&data[index_data(bin, thread_idx, padded_bins, padded_threads)])
              T(identity);
        }
      }
    }
    return data;
  }

  static void destroy_data(T*& data,
                           size_t num_bins,
                           size_t max_threads,
                           size_t padded_bins,
                           size_t padded_threads)
  {
    if (num_bins == size_t(0))
    {
      return;
    }
    for (size_t thread_idx = max_threads; thread_idx > 0; --thread_idx)
    {
      for (size_t bin = num_bins; bin > 0; --bin)
      {
        data[index_data(bin - 1, thread_idx - 1, padded_bins, padded_threads)]
            .~T();
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
