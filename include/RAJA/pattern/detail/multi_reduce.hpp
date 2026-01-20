/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Base types used in common for RAJA reducer objects.
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

#ifndef RAJA_PATTERN_DETAIL_MULTI_REDUCE_HPP
#define RAJA_PATTERN_DETAIL_MULTI_REDUCE_HPP

#include "RAJA/pattern/detail/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/RepeatView.hpp"


#define RAJA_DECLARE_MULTI_REDUCER(OP_NAME, OP, POL, DATA)                     \
  template<typename tuning, typename T>                                        \
  struct MultiReduce##OP_NAME<POL<tuning>, T>                                  \
      : reduce::detail::BaseMultiReduce##OP_NAME<                              \
            DATA<T, RAJA::reduce::OP<T>, tuning>>                              \
  {                                                                            \
    using policy = POL<tuning>;                                                \
    using Base   = reduce::detail::BaseMultiReduce##OP_NAME<                   \
        DATA<T, RAJA::reduce::OP<T>, tuning>>;                               \
    using Base::Base;                                                          \
    using typename Base::value_type;                                           \
    using typename Base::reference;                                            \
                                                                               \
    RAJA_SUPPRESS_HD_WARN                                                      \
    RAJA_HOST_DEVICE                                                           \
    reference operator[](size_t bin) const { return reference(*this, bin); }   \
  };

#define RAJA_DECLARE_ALL_MULTI_REDUCERS(POL, DATA)                             \
  RAJA_DECLARE_MULTI_REDUCER(Sum, sum, POL, DATA)                              \
  RAJA_DECLARE_MULTI_REDUCER(Min, min, POL, DATA)                              \
  RAJA_DECLARE_MULTI_REDUCER(Max, max, POL, DATA)                              \
  RAJA_DECLARE_MULTI_REDUCER(BitOr, or_bit, POL, DATA)                         \
  RAJA_DECLARE_MULTI_REDUCER(BitAnd, and_bit, POL, DATA)

namespace RAJA
{

namespace reduce
{

namespace detail
{

template<typename t_MultiReduceData>
struct BaseMultiReduce
{
  using MultiReduceData = t_MultiReduceData;
  using MultiReduceOp   = typename t_MultiReduceData::MultiReduceOp;
  using value_type      = typename t_MultiReduceData::value_type;

  BaseMultiReduce()
      : BaseMultiReduce {RepeatView<value_type>(MultiReduceOp::identity(), 0)}
  {}

  explicit BaseMultiReduce(size_t num_bins,
                           value_type init_val = MultiReduceOp::identity(),
                           value_type identity = MultiReduceOp::identity())
      : BaseMultiReduce {RepeatView<value_type>(init_val, num_bins), identity}
  {}

  template<typename Container,
           concepts::enable_if_t<
               type_traits::is_range<Container>,
               concepts::negate<std::is_convertible<Container, size_t>>,
               concepts::negate<std::is_base_of<BaseMultiReduce, Container>>>* =
               nullptr>
  explicit BaseMultiReduce(Container const& container,
                           value_type identity = MultiReduceOp::identity())
      : data {container, identity}
  {}

  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduce(BaseMultiReduce const&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduce(BaseMultiReduce&&)                 = default;
  BaseMultiReduce& operator=(BaseMultiReduce const&) = delete;
  BaseMultiReduce& operator=(BaseMultiReduce&&)      = delete;
  RAJA_SUPPRESS_HD_WARN
  ~BaseMultiReduce() = default;

  void reset()
  {
    reset(RepeatView<value_type>(MultiReduceOp::identity(), size()));
  }

  void reset(size_t num_bins,
             value_type init_val = MultiReduceOp::identity(),
             value_type identity = MultiReduceOp::identity())
  {
    reset(RepeatView<value_type>(init_val, num_bins), identity);
  }

  template<typename Container,
           concepts::enable_if_t<type_traits::is_range<Container>>* = nullptr>
  void reset(Container const& container,
             value_type identity = MultiReduceOp::identity())
  {
    for (size_t bin = 0; bin < data.num_bins(); ++bin)
    {
      RAJA_UNUSED_VAR(get(bin));  // automatic get() before reset
    }
    data.reset(container, identity);
  }

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  size_t size() const { return data.num_bins(); }

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  BaseMultiReduce const& combine(size_t bin, value_type const& other) const
  {
    data.combine(bin, other);
    return *this;
  }

  //! Get the calculated reduced value for a bin
  value_type get(size_t bin) const { return data.get(bin); }

  //! Get the calculated reduced value for each bin and store it in container
  template<typename Container,
           concepts::enable_if_t<type_traits::is_range<Container>>* = nullptr>
  void get_all(Container& container) const
  {
    RAJA_EXTRACT_BED_IT(container);
    if (size_t(distance_it) != data.num_bins())
    {
      RAJA_ABORT_OR_THROW("MultiReduce::get_all container has different size "
                          "than multi reducer");
    }
    size_t bin = 0;
    for (auto& val : container)
    {
      val = data.get(bin);
      ++bin;
    }
  }

private:
  MultiReduceData mutable data;
};

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template.
 *
 ******************************************************************************
 */
template<typename MultiReduceData>
class BaseMultiReduceMin : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using Base::Base;
  using typename Base::value_type;

  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceMin(BaseMultiReduceMin const&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceMin(BaseMultiReduceMin&&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceMin& operator=(BaseMultiReduceMin const&) = delete;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceMin& operator=(BaseMultiReduceMin&&) = delete;
  RAJA_SUPPRESS_HD_WARN
  ~BaseMultiReduceMin() = default;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceMin const& base, size_t bin)
        : m_base(base),
          m_bin(bin)
    {}

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& min(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const { return m_base.get(m_bin); }

  private:
    BaseMultiReduceMin const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template.
 *
 **************************************************************************
 */
template<typename MultiReduceData>
class BaseMultiReduceMax : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceMax(BaseMultiReduceMax const&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceMax(BaseMultiReduceMax&&)                 = default;
  BaseMultiReduceMax& operator=(BaseMultiReduceMax const&) = delete;
  BaseMultiReduceMax& operator=(BaseMultiReduceMax&&)      = delete;
  RAJA_SUPPRESS_HD_WARN
  ~BaseMultiReduceMax() = default;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceMax const& base, size_t bin)
        : m_base(base),
          m_bin(bin)
    {}

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& max(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const { return m_base.get(m_bin); }

  private:
    BaseMultiReduceMax const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template.
 *
 **************************************************************************
 */
template<typename MultiReduceData>
class BaseMultiReduceSum : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceSum(BaseMultiReduceSum const&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceSum(BaseMultiReduceSum&&)                 = default;
  BaseMultiReduceSum& operator=(BaseMultiReduceSum const&) = delete;
  BaseMultiReduceSum& operator=(BaseMultiReduceSum&&)      = delete;
  RAJA_SUPPRESS_HD_WARN
  ~BaseMultiReduceSum() = default;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceSum const& base, size_t bin)
        : m_base(base),
          m_bin(bin)
    {}

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& operator+=(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const { return m_base.get(m_bin); }

  private:
    BaseMultiReduceSum const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Bitwise OR reducer class template.
 *
 **************************************************************************
 */
template<typename MultiReduceData>
class BaseMultiReduceBitOr : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceBitOr(BaseMultiReduceBitOr const&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceBitOr(BaseMultiReduceBitOr&&)                 = default;
  BaseMultiReduceBitOr& operator=(BaseMultiReduceBitOr const&) = delete;
  BaseMultiReduceBitOr& operator=(BaseMultiReduceBitOr&&)      = delete;
  RAJA_SUPPRESS_HD_WARN
  ~BaseMultiReduceBitOr() = default;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceBitOr const& base, size_t bin)
        : m_base(base),
          m_bin(bin)
    {}

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& operator|=(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const { return m_base.get(m_bin); }

  private:
    BaseMultiReduceBitOr const& m_base;
    size_t m_bin;
  };
};

/*!
 **************************************************************************
 *
 * \brief  Bitwise AND reducer class template.
 *
 **************************************************************************
 */
template<typename MultiReduceData>
class BaseMultiReduceBitAnd : public BaseMultiReduce<MultiReduceData>
{
public:
  using Base = BaseMultiReduce<MultiReduceData>;
  using typename Base::value_type;

  using Base::Base;

  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceBitAnd(BaseMultiReduceBitAnd const&) = default;
  RAJA_SUPPRESS_HD_WARN
  BaseMultiReduceBitAnd(BaseMultiReduceBitAnd&&)                 = default;
  BaseMultiReduceBitAnd& operator=(BaseMultiReduceBitAnd const&) = delete;
  BaseMultiReduceBitAnd& operator=(BaseMultiReduceBitAnd&&)      = delete;
  RAJA_SUPPRESS_HD_WARN
  ~BaseMultiReduceBitAnd() = default;

  struct reference
  {
    RAJA_HOST_DEVICE
    reference(BaseMultiReduceBitAnd const& base, size_t bin)
        : m_base(base),
          m_bin(bin)
    {}

    //! reducer function; updates the current instance's state
    RAJA_HOST_DEVICE
    reference const& operator&=(value_type rhs) const
    {
      m_base.combine(m_bin, rhs);
      return *this;
    }

    value_type get() const { return m_base.get(m_bin); }

  private:
    BaseMultiReduceBitAnd const& m_base;
    size_t m_bin;
  };
};

}  // namespace detail

}  // namespace reduce

}  // namespace RAJA

#endif /* RAJA_PATTERN_DETAIL_MULTI_REDUCE_HPP */
