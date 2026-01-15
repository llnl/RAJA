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

#ifndef RAJA_PATTERN_DETAIL_REDUCE_HPP
#define RAJA_PATTERN_DETAIL_REDUCE_HPP

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

#define RAJA_DECLARE_REDUCER(OP, POL, COMBINER)                                \
  template<typename T>                                                         \
  class Reduce##OP<POL, T>                                                     \
      : public reduce::detail::BaseReduce##OP<T, COMBINER>                     \
  {                                                                            \
  public:                                                                      \
    using Base = reduce::detail::BaseReduce##OP<T, COMBINER>;                  \
    using Base::Base;                                                          \
  };

#define RAJA_DECLARE_INDEX_REDUCER(OP, POL, COMBINER)                          \
  template<typename T, typename IndexType>                                     \
  class Reduce##OP<POL, T, IndexType>                                          \
      : public reduce::detail::BaseReduce##OP<T, IndexType, COMBINER>          \
  {                                                                            \
  public:                                                                      \
    using Base = reduce::detail::BaseReduce##OP<T, IndexType, COMBINER>;       \
    using Base::Base;                                                          \
  };

#define RAJA_DECLARE_ALL_REDUCERS(POL, COMBINER)                               \
  RAJA_DECLARE_REDUCER(Sum, POL, COMBINER)                                     \
  RAJA_DECLARE_REDUCER(Min, POL, COMBINER)                                     \
  RAJA_DECLARE_REDUCER(Max, POL, COMBINER)                                     \
  RAJA_DECLARE_INDEX_REDUCER(MinLoc, POL, COMBINER)                            \
  RAJA_DECLARE_INDEX_REDUCER(MaxLoc, POL, COMBINER)                            \
  RAJA_DECLARE_REDUCER(BitOr, POL, COMBINER)                                   \
  RAJA_DECLARE_REDUCER(BitAnd, POL, COMBINER)

namespace RAJA
{

namespace reduce
{

#if defined(RAJA_ENABLE_TARGET_OPENMP)
#pragma omp declare target
#endif

namespace detail
{

template<typename T, template<typename...> class Op>
struct op_adapter : private Op<T, T, T>
{
  using operator_type = Op<T, T, T>;

  RAJA_HOST_DEVICE static constexpr T identity()
  {
    return operator_type::identity();
  }

  RAJA_HOST_DEVICE RAJA_INLINE void operator()(T& val, const T v) const
  {
    val = operator_type::operator()(val, v);
  }
};
}  // namespace detail

template<typename T>
struct sum : detail::op_adapter<T, RAJA::operators::plus>
{};

template<typename T>
struct min : detail::op_adapter<T, RAJA::operators::minimum>
{};

template<typename T>
struct max : detail::op_adapter<T, RAJA::operators::maximum>
{};

template<typename T>
struct or_bit : detail::op_adapter<T, RAJA::operators::bit_or>
{};

template<typename T>
struct and_bit : detail::op_adapter<T, RAJA::operators::bit_and>
{};


#if defined(RAJA_ENABLE_TARGET_OPENMP)
#pragma omp end declare target
#endif

namespace detail
{

template<typename T, bool = std::is_integral<T>::value>
struct DefaultLoc
{};

template<typename T>
struct DefaultLoc<T, false>  // any non-integral type
{
  RAJA_HOST_DEVICE constexpr T value() const { return T(); }
};

template<typename T>
struct DefaultLoc<T, true>
{
  RAJA_HOST_DEVICE constexpr T value() const { return -1; }
};

template<typename T, typename IndexType, bool doing_min = true>
class ValueLoc
{
public:
  T val = doing_min ? operators::limits<T>::max() : operators::limits<T>::min();
  IndexType loc = DefaultLoc<IndexType>().value();

#if __NVCC__ && defined(CUDART_VERSION) && CUDART_VERSION < 9020 ||            \
    defined(__HIPCC__)
  RAJA_HOST_DEVICE constexpr ValueLoc() {}

  RAJA_HOST_DEVICE constexpr ValueLoc(ValueLoc const& other)
      : val {other.val},
        loc {other.loc}
  {}

  RAJA_HOST_DEVICE
  ValueLoc& operator=(ValueLoc const& other)
  {
    val = other.val;
    loc = other.loc;
    return *this;
  }
#else
  constexpr ValueLoc()                 = default;
  constexpr ValueLoc(ValueLoc const&)  = default;
  ValueLoc& operator=(ValueLoc const&) = default;
#endif

  RAJA_HOST_DEVICE constexpr ValueLoc(T const& val_)
      : val {val_},
        loc {DefaultLoc<IndexType>().value()}
  {}

  RAJA_HOST_DEVICE constexpr ValueLoc(T const& val_, IndexType const& loc_)
      : val {val_},
        loc {loc_}
  {}

  RAJA_HOST_DEVICE operator T() const { return val; }

  RAJA_HOST_DEVICE IndexType getLoc() { return loc; }

  RAJA_HOST_DEVICE bool operator<(ValueLoc const& rhs) const
  {
    return val < rhs.val;
  }

  RAJA_HOST_DEVICE bool operator>(ValueLoc const& rhs) const
  {
    return val > rhs.val;
  }
};

}  // namespace detail

}  // namespace reduce

namespace operators
{
template<typename T, typename IndexType, bool B>
struct limits<::RAJA::reduce::detail::ValueLoc<T, IndexType, B>>
{
  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ::RAJA::reduce::detail::
      ValueLoc<T, IndexType, B>
      min()
  {
    return ::RAJA::reduce::detail::ValueLoc<T, IndexType, B>(limits<T>::min());
  }

  RAJA_INLINE RAJA_HOST_DEVICE static constexpr ::RAJA::reduce::detail::
      ValueLoc<T, IndexType, B>
      max()
  {
    return ::RAJA::reduce::detail::ValueLoc<T, IndexType, B>(limits<T>::max());
  }
};
}  // namespace operators

namespace reduce
{

namespace detail
{

template<typename T,
         template<typename>
         class Reduce_,
         template<typename, typename>
         class Combiner_>
class BaseReduce
{
  using Reduce = Reduce_<T>;
  // NOTE: the _t here is to appease MSVC
  using Combiner_t = Combiner_<T, Reduce>;
  Combiner_t mutable c;

public:
  using value_type  = T;
  using reduce_type = Reduce;

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  BaseReduce() : c {T(), Reduce::identity()} {}

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  BaseReduce(T init_val, T identity_ = Reduce::identity())
      : c {init_val, identity_}
  {}

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  void reset(T val, T identity_ = Reduce::identity())
  {
    operator T();  // automatic get() before reset
    c.reset(val, identity_);
  }

  //! prohibit compiler-generated copy assignment
  BaseReduce& operator=(const BaseReduce&) = delete;

  //! compiler-generated copy constructor
  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  BaseReduce(const BaseReduce& copy) : c(copy.c) {}

  //! compiler-generated move constructor
  RAJA_SUPPRESS_HD_WARN
  RAJA_HOST_DEVICE

  RAJA_INLINE
  BaseReduce(BaseReduce&& copy) : c(std::move(copy.c)) {}

  //! compiler-generated move assignment
  BaseReduce& operator=(BaseReduce&&) = default;

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  void combine(T const& other) const { c.combine(other); }

  T& local() const { return c.local(); }

  //! Get the calculated reduced value
  operator T() const { return c.get(); }

  //! Get the calculated reduced value
  T get() const { return c.get(); }
};

template<typename T, typename Reduce, typename Derived>
class BaseCombinable
{
protected:
  BaseCombinable const* parent = nullptr;
  T identity;
  T mutable my_data;

public:
  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  constexpr BaseCombinable() : identity {T()}, my_data {T()} {}

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  constexpr BaseCombinable(T init_val, T identity_ = T())
      : identity {identity_},
        my_data {init_val}
  {}

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  void reset(T init_val, T identity_)
  {
    my_data  = init_val;
    identity = identity_;
  }

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  constexpr BaseCombinable(BaseCombinable const& other)
      : parent {other.parent ? other.parent : &other},
        identity {other.identity},
        my_data {identity}
  {}

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  ~BaseCombinable()
  {
    if (parent && my_data != identity)
    {
      Reduce()(parent->my_data, my_data);
    }
  }

  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  void combine(T const& other) { Reduce {}(my_data, other); }

  /*!
   *  \return the calculated reduced value
   */
  T get() const { return derived().get_combined(); }

  /*!
   *  \return reference to the local value
   */
  T& local() const { return my_data; }

  T get_combined() const { return my_data; }

private:
  // Convenience method for CRTP
  const Derived& derived() const
  {
    return *(static_cast<const Derived*>(this));
  }

  Derived& derived() { return *(static_cast<Derived*>(this)); }
};

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template.
 *
 ******************************************************************************
 */
template<typename T, template<typename, typename> class Combiner>
class BaseReduceMin : public BaseReduce<T, RAJA::reduce::min, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::min, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const BaseReduceMin& min(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  MinLoc reducer class template.
 *
 **************************************************************************
 */
template<typename T,
         typename IndexType,
         template<typename, typename>
         class Combiner>
class BaseReduceMinLoc
    : public BaseReduce<ValueLoc<T, IndexType>, RAJA::reduce::min, Combiner>
{
public:
  using Base = BaseReduce<ValueLoc<T, IndexType>, RAJA::reduce::min, Combiner>;
  using value_type  = typename Base::value_type;
  using reduce_type = typename Base::reduce_type;
  using Base::Base;

  constexpr BaseReduceMinLoc() : Base(value_type(T(), IndexType())) {}

  constexpr BaseReduceMinLoc(
      T init_val,
      IndexType init_idx,
      T identity_val_         = reduce_type::identity(),
      IndexType identity_loc_ = DefaultLoc<IndexType>().value())
      : Base(value_type(init_val, init_idx),
             value_type(identity_val_, identity_loc_))
  {}

  void reset(T init_val,
             IndexType init_idx,
             T identity_val_         = reduce_type::identity(),
             IndexType identity_loc_ = DefaultLoc<IndexType>().value())
  {
    operator T();  // automatic get() before reset
    Base::reset(value_type(init_val, init_idx),
                value_type(identity_val_, identity_loc_));
  }

  /// \brief reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const BaseReduceMinLoc& minloc(T rhs, IndexType loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  IndexType getLoc() const { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() const { return Base::get(); }
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template.
 *
 **************************************************************************
 */
template<typename T, template<typename, typename> class Combiner>
class BaseReduceMax : public BaseReduce<T, RAJA::reduce::max, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::max, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const BaseReduceMax& max(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template.
 *
 **************************************************************************
 */
template<typename T, template<typename, typename> class Combiner>
class BaseReduceSum : public BaseReduce<T, RAJA::reduce::sum, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::sum, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  const BaseReduceSum& operator+=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  Bitwise OR reducer class template.
 *
 **************************************************************************
 */
template<typename T, template<typename, typename> class Combiner>
class BaseReduceBitOr : public BaseReduce<T, RAJA::reduce::or_bit, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::or_bit, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  const BaseReduceBitOr& operator|=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  Bitwise AND reducer class template.
 *
 **************************************************************************
 */
template<typename T, template<typename, typename> class Combiner>
class BaseReduceBitAnd : public BaseReduce<T, RAJA::reduce::and_bit, Combiner>
{
public:
  using Base = BaseReduce<T, RAJA::reduce::and_bit, Combiner>;
  using Base::Base;

  //! reducer function; updates the current instance's state
  RAJA_SUPPRESS_HD_WARN

  RAJA_HOST_DEVICE
  const BaseReduceBitAnd& operator&=(T rhs) const
  {
    this->combine(rhs);
    return *this;
  }
};

/*!
 **************************************************************************
 *
 * \brief  MaxLoc reducer class template.
 *
 **************************************************************************
 */
template<typename T,
         typename IndexType,
         template<typename, typename>
         class Combiner>
class BaseReduceMaxLoc : public BaseReduce<ValueLoc<T, IndexType, false>,
                                           RAJA::reduce::max,
                                           Combiner>
{
public:
  using Base =
      BaseReduce<ValueLoc<T, IndexType, false>, RAJA::reduce::max, Combiner>;
  using value_type  = typename Base::value_type;
  using reduce_type = typename Base::reduce_type;
  using Base::Base;

  constexpr BaseReduceMaxLoc() : Base(value_type(T(), IndexType())) {}

  constexpr BaseReduceMaxLoc(
      T init_val,
      IndexType init_idx,
      T identity_val_         = reduce_type::identity(),
      IndexType identity_loc_ = DefaultLoc<IndexType>().value())
      : Base(value_type(init_val, init_idx),
             value_type(identity_val_, identity_loc_))
  {}

  void reset(T init_val,
             IndexType init_idx,
             T identity_val_         = reduce_type::identity(),
             IndexType identity_loc_ = DefaultLoc<IndexType>().value())
  {
    operator T();  // automatic get() before reset
    Base::reset(value_type(init_val, init_idx),
                value_type(identity_val_, identity_loc_));
  }

  //! reducer function; updates the current instance's state
  RAJA_HOST_DEVICE
  const BaseReduceMaxLoc& maxloc(T rhs, IndexType loc) const
  {
    this->combine(value_type(rhs, loc));
    return *this;
  }

  //! Get the calculated reduced value
  IndexType getLoc() const { return Base::get().getLoc(); }

  //! Get the calculated reduced value
  operator T() const { return Base::get(); }
};

}  // namespace detail

}  // namespace reduce

}  // namespace RAJA

#endif /* RAJA_PATTERN_DETAIL_REDUCE_HPP */
