//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PARAMS_BASE
#define RAJA_PARAMS_BASE

namespace RAJA
{
namespace expt
{

template<typename T, typename IndexType = RAJA::Index_type>
struct ValLoc
{
  using index_type = IndexType;
  using value_type = T;

  ValLoc() = default;

  RAJA_HOST_DEVICE constexpr explicit ValLoc(value_type v) : val(v) {}

  RAJA_HOST_DEVICE constexpr ValLoc(value_type v, index_type l) : val(v), loc(l)
  {}

  ValLoc(ValLoc const&)            = default;
  ValLoc(ValLoc&&)                 = default;
  ValLoc& operator=(ValLoc const&) = default;
  ValLoc& operator=(ValLoc&&)      = default;

  RAJA_HOST_DEVICE constexpr bool operator<(const ValLoc& rhs) const
  {
    return val < rhs.val;
  }

  RAJA_HOST_DEVICE constexpr bool operator>(const ValLoc& rhs) const
  {
    return val > rhs.val;
  }

  RAJA_HOST_DEVICE constexpr const value_type& getVal() const { return val; }

  RAJA_HOST_DEVICE constexpr const index_type& getLoc() const { return loc; }

  RAJA_HOST_DEVICE void set(T inval, IndexType inindex)
  {
    val = inval;
    loc = inindex;
  }

  RAJA_HOST_DEVICE void setVal(T inval) { val = inval; }

  RAJA_HOST_DEVICE void setLoc(IndexType inindex) { loc = inindex; }

  value_type val;
  index_type loc = -1;
};

template<typename T, template<typename, typename, typename> class Op>
struct ValOp
{
  using value_type = T;
  using op_type    = Op<T, T, T>;

  ValOp() = default;

  RAJA_HOST_DEVICE constexpr explicit ValOp(value_type v) : val(v) {}

  ValOp(ValOp const&)            = default;
  ValOp(ValOp&&)                 = default;
  ValOp& operator=(ValOp const&) = default;
  ValOp& operator=(ValOp&&)      = default;

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::minimum<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& min(value_type v)
  {
    if (v < val)
    {
      val = v;
    }
    return *this;
  }

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::maximum<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& max(value_type v)
  {
    if (v > val)
    {
      val = v;
    }
    return *this;
  }

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::plus<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& operator+=(const value_type& rhs)
  {
    val += rhs;
    return *this;
  }

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::bit_and<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& operator&=(const value_type& rhs)
  {
    val &= rhs;
    return *this;
  }

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::bit_or<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& operator|=(const value_type& rhs)
  {
    val |= rhs;
    return *this;
  }

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::bit_and<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE ValOp& operator&=(value_type& rhs)
  {
    val &= rhs;
    return *this;
  }

  template<
      typename U = op_type,
      std::enable_if_t<
          std::is_same<U, RAJA::operators::bit_or<T, T, T>>::value>* = nullptr>
  RAJA_HOST_DEVICE ValOp& operator|=(value_type& rhs)
  {
    val |= rhs;
    return *this;
  }

  RAJA_HOST_DEVICE constexpr bool operator<(const ValOp& rhs) const
  {
    val < rhs.val;
    return *this;
  }

  RAJA_HOST_DEVICE constexpr bool operator>(const ValOp& rhs) const
  {
    val > rhs.val;
    return *this;
  }

  value_type val = op_type::identity();
};

template<typename T,
         typename IndexType,
         template<typename, typename, typename>
         class Op>
struct ValOp<ValLoc<T, IndexType>, Op>
{
  using index_type        = IndexType;
  using value_type        = ValLoc<T, index_type>;
  using op_type           = Op<value_type, value_type, value_type>;
  using valloc_value_type = typename value_type::value_type;
  using valloc_index_type = typename value_type::index_type;

  ValOp() = default;

  RAJA_HOST_DEVICE constexpr explicit ValOp(value_type v) : val(v) {}

  RAJA_HOST_DEVICE constexpr ValOp(valloc_value_type v, valloc_index_type l)
      : val(v, l)
  {}

  ValOp(ValOp const&)            = default;
  ValOp(ValOp&&)                 = default;
  ValOp& operator=(ValOp const&) = default;
  ValOp& operator=(ValOp&&)      = default;

  template<typename U                   = op_type,
           std::enable_if_t<std::is_same<
               U,
               RAJA::operators::minimum<value_type, value_type, value_type>>::
                                value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& min(value_type v)
  {
    if (v < val)
    {
      val = v;
    }
    return *this;
  }

  template<typename U                   = op_type,
           std::enable_if_t<std::is_same<
               U,
               RAJA::operators::maximum<value_type, value_type, value_type>>::
                                value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& max(value_type v)
  {
    if (v > val)
    {
      val = v;
    }
    return *this;
  }

  template<typename U                   = op_type,
           std::enable_if_t<std::is_same<
               U,
               RAJA::operators::minimum<value_type, value_type, value_type>>::
                                value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& minloc(valloc_value_type v,
                                           valloc_index_type l)
  {
    return min(value_type(v, l));
  }

  template<typename U                   = op_type,
           std::enable_if_t<std::is_same<
               U,
               RAJA::operators::maximum<value_type, value_type, value_type>>::
                                value>* = nullptr>
  RAJA_HOST_DEVICE constexpr ValOp& maxloc(valloc_value_type v,
                                           valloc_index_type l)
  {
    return max(value_type(v, l));
  }

  RAJA_HOST_DEVICE constexpr bool operator<(const ValOp& rhs) const
  {
    return val < rhs.val;
  }

  RAJA_HOST_DEVICE constexpr bool operator>(const ValOp& rhs) const
  {
    return val > rhs.val;
  }

  value_type val = op_type::identity();
};

template<typename T,
         typename IndexType,
         template<typename, typename, typename>
         class Op>
using ValLocOp = ValOp<ValLoc<T, IndexType>, Op>;

namespace detail
{

struct ParamBase
{
  // Some of this can be made virtual in c++20, for now must be defined in each
  // child class if any arguments to the forall lambda are needed (e.g.
  // Name is excluded.)
  using ARG_TUP_T  = camp::tuple<>;
  using ARG_T      = ParamBase;
  using ARG_LIST_T = typename ARG_TUP_T::TList;

  RAJA_HOST_DEVICE ARG_TUP_T get_lambda_arg_tup() { return camp::make_tuple(); }

  RAJA_HOST_DEVICE ARG_T* get_lambda_arg() { return this; }

  static constexpr size_t num_lambda_args = camp::tuple_size<ARG_TUP_T>::value;
};

struct ForallParamBase : public ParamBase
{};

// Convert a tuple of parameter types to their respective arg type,
// EG Reduction<Double> -> Valop<Double>
template<typename T>
struct ParamToArgHelper
{};

template<typename ParamType, typename Enable = void>
struct GetArgType
{
  using type = ParamType;
};

template<typename ParamType>
struct GetArgType<
    ParamType,
    std::enable_if_t<std::is_base_of<ParamBase, ParamType>::value>>
{
  using type = typename ParamType::ARG_T;
};

template<typename... Params>
struct ParamToArgHelper<camp::tuple<Params...>>
{
  using type = camp::tuple<typename GetArgType<Params>::type...>;
};

template<typename T>
RAJA_HOST_DEVICE std::enable_if_t<
    std::is_base_of<ParamBase, T>::value,
    std::add_lvalue_reference_t<typename T::ARG_T>>
get_lambda_arg(T& Param)
{
  return *Param.get_lambda_arg();
}

template<typename T>
RAJA_HOST_DEVICE std::enable_if_t<!std::is_base_of<ParamBase, T>::value,
                                  std::add_lvalue_reference_t<T>>
get_lambda_arg(T& Param)
{
  return Param;
}

}  // namespace detail

}  // namespace expt

}  //  namespace RAJA

#endif  //  RAJA_PARAMS_BASE
