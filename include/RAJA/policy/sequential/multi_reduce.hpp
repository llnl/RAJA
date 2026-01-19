/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
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

#ifndef RAJA_sequential_multi_reduce_HPP
#define RAJA_sequential_multi_reduce_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "RAJA/pattern/detail/multi_reduce.hpp"
#include "RAJA/pattern/multi_reduce.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{

/*!
 **************************************************************************
 *
 * \brief  Seq multi-reduce data class template.
 *
 * In this class memory is owned by the parent object
 *
 **************************************************************************
 */
template<typename T, typename t_MultiReduceOp, typename tuning>
struct MultiReduceDataSeq;

/*!
 **************************************************************************
 *
 * \brief  Seq multi-reduce data class template using left_fold reductions.
 *
 * In this class memory is owned by the parent object
 *
 **************************************************************************
 */
template<typename T, typename t_MultiReduceOp>
struct MultiReduceDataSeq<
    T,
    t_MultiReduceOp,
    RAJA::sequential::MultiReduceTuning<
        RAJA::sequential::multi_reduce_algorithm::left_fold>>
{
  using value_type    = T;
  using MultiReduceOp = t_MultiReduceOp;

  MultiReduceDataSeq() = delete;

  template<typename Container,
           std::enable_if_t<
               !std::is_same<Container, MultiReduceDataSeq>::value>* = nullptr>
  MultiReduceDataSeq(Container const& container, T identity)
      : m_parent(nullptr),
        m_num_bins(container.size()),
        m_identity(identity),
        m_data(nullptr)
  {
    m_data = create_data(container, m_num_bins);
  }

  MultiReduceDataSeq(MultiReduceDataSeq const& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_num_bins(other.m_num_bins),
        m_identity(other.m_identity),
        m_data(other.m_data)
  {}

  MultiReduceDataSeq(MultiReduceDataSeq&&)                 = delete;
  MultiReduceDataSeq& operator=(MultiReduceDataSeq const&) = delete;
  MultiReduceDataSeq& operator=(MultiReduceDataSeq&&)      = delete;

  ~MultiReduceDataSeq()
  {
    if (m_data)
    {
      if (!m_parent)
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
  MultiReduceDataSeq const* m_parent;
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

    auto data  = static_cast<T*>(malloc(num_bins * sizeof(T)));
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

    for (size_t bin = 0; bin < num_bins; ++bin)
    {
      data[bin].~T();
    }
    free(data);
    data = nullptr;
  }
};

}  // namespace detail

RAJA_DECLARE_ALL_MULTI_REDUCERS(policy::sequential::seq_multi_reduce_policy,
                                detail::MultiReduceDataSeq)

}  // namespace RAJA

#endif  // closing endif for header file include guard
