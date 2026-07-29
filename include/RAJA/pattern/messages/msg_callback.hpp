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
class IMsgCallback
{
public:
  virtual ~IMsgCallback() = default;

  virtual std::type_index get_type() const { return typeid(void); }

  virtual void operator()(char*) const   = 0;
  virtual void destroy_args(char*) const = 0;
};

template<typename Callable, typename Signature>
class MsgCallback;

template<typename Callable, typename Ret, typename... Args>
class MsgCallback<Callable, Ret(Args...)> : public IMsgCallback
{
public:
  using return_t = Ret;

  explicit MsgCallback(const Callable& callable) : m_callable {callable} {}

  explicit MsgCallback(Callable&& callable) : m_callable {std::move(callable)}
  {}

  std::type_index get_type() const final { return typeid(Callable); }

  void operator()(char* args_buf) const final override
  {
    auto& msg = *std::launder(
        reinterpret_cast<MsgArgs<std::decay_t<Args>...>*>(args_buf));
    camp::apply(m_callable, msg.args);
  }

  void destroy_args(char* args_buf) const final override
  {
    auto& msg = *std::launder(
        reinterpret_cast<MsgArgs<std::decay_t<Args>...>*>(args_buf));
    msg.~MsgArgs<std::decay_t<Args>...>();
  }

private:
  Callable m_callable;
};

template<typename R, typename... Args>
MsgCallback(R (*)(Args...)) -> MsgCallback<R (*)(Args...), R(Args...)>;

template<
    typename Callable,
    typename Signature = internal::signature_t<decltype(&Callable::operator())>>
MsgCallback(const Callable&) -> MsgCallback<Callable, Signature>;

template<
    typename Callable,
    typename Signature = internal::signature_t<decltype(&Callable::operator())>>
MsgCallback(Callable&&) -> MsgCallback<Callable, Signature>;
}  // namespace RAJA

#endif  // RAJA_MSG_CALLBACK_HPP
