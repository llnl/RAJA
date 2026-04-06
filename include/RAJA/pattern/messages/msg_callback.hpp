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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MSG_CALLBACK_HPP
#define RAJA_MSG_CALLBACK_HPP

#include <typeindex>
#include <type_traits>

#include "RAJA/pattern/messages/msg_header.hpp"
#include "RAJA/util/FunctionTypeTraits.hpp"

namespace RAJA
{
class imsg_callback
{
public:
  virtual ~imsg_callback() = default;

  virtual std::type_index get_type() const { return typeid(void); }

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

  std::type_index get_type() const final override { return typeid(Callable); }

  void operator()(char* args_buf) const final override
  {
    auto& msg = *std::launder(
        reinterpret_cast<msg_args<std::decay_t<Args>...>*>(args_buf));
    camp::apply(m_callable, msg.args);
    msg.~msg_args<std::decay_t<Args>...>();
  }

private:
  Callable m_callable;
};

template<typename R, typename... Args>
msg_callback(R (*)(Args...)) -> msg_callback<R (*)(Args...), R(Args...)>;

template<
    typename Object,
    typename Signature = internal::signature_t<decltype(&Object::operator())>>
msg_callback(const Object&) -> msg_callback<Object, Signature>;

template<
    typename Object,
    typename Signature = internal::signature_t<decltype(&Object::operator())>>
msg_callback(Object&&) -> msg_callback<Object, Signature>;
}  // namespace RAJA

#endif  // RAJA_MSG_CALLBACK_HPP
