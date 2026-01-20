/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining callback helpers for RAJA::messages.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJA_MSG_CALLBACK_HPP
#define RAJA_MSG_CALLBACK_HPP

#include <type_traits>

#include "RAJA/util/msg_header.hpp"

namespace RAJA
{
class imsg_callback
{
public:
  virtual ~imsg_callback() = default;

  virtual std::size_t hash() const { return typeid(void).hash_code(); }

  virtual void operator()(char*) const = 0;
};

template<typename Callable, typename Signature>
class msg_callback;

template<typename Callable, typename Ret, typename... Args>
class msg_callback<Callable, Ret(Args...)> : public imsg_callback
{
public:
  using return_t = Ret;

  template<typename Object>
  explicit msg_callback(const Object& obj) : m_callable {obj}
  {}

  template<typename Object>
  explicit msg_callback(Object&& obj) : m_callable {std::move(obj)}
  {}

  std::size_t hash() const final override
  {
    return typeid(Callable).hash_code();
  }

  void operator()(char* args_buf) const final override
  {
    auto& args = *std::launder(
        reinterpret_cast<msg_args<std::decay_t<Args>...>*>(args_buf));
    camp::apply(m_callable, args);
    args.~msg_args<std::decay_t<Args>...>();
  }

private:
  Callable m_callable;
};

template<typename Fn>
struct get_signature;

template<typename R, typename C, typename... Args>
struct get_signature<R (C::*)(Args...)>
{
  using type = R(Args...);
};

template<typename R, typename C, typename... Args>
struct get_signature<R (C::*)(Args...) const>
{
  using type = R(Args...);
};

template<typename R, typename C, typename... Args>
struct get_signature<R (C::*)(Args...) noexcept>
{
  using type = R(Args...);
};

template<typename R, typename C, typename... Args>
struct get_signature<R (C::*)(Args...) const noexcept>
{
  using type = R(Args...);
};

template<typename R, typename... Args>
msg_callback(R (*)(Args...)) -> msg_callback<R (*)(Args...), R(Args...)>;

template<typename Object,
         typename Signature =
             typename get_signature<decltype(&Object::operator())>::type>
msg_callback(const Object&) -> msg_callback<Object, Signature>;

template<typename Object,
         typename Signature =
             typename get_signature<decltype(&Object::operator())>::type>
msg_callback(Object&&) -> msg_callback<Object, Signature>;
}  // namespace RAJA

#endif  // RAJA_MSG_CALLBACK_HPP
