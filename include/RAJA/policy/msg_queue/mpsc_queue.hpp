/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing implementation for a MPSC
 *          message queue policy. By SPSC, means multi-producer
 *          single-consumer. In other words, messages produced
 *          could be from multiple thread may require atomics.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_mpsc_queue_HPP
#define RAJA_mpsc_queue_HPP

#include "RAJA/util/concepts.hpp"
#include "RAJA/pattern/atomic.hpp"

#include "RAJA/util/msg_header.hpp"
#include "RAJA/policy/msg_queue/policy.hpp"

namespace RAJA
{
namespace messages
{

template<typename Container, typename... Args>
class queue<Container, RAJA::mpsc_queue, RAJA::msg_args<Args...>>
{
public:
  using policy = RAJA::mpsc_queue;

  using args_type = camp::tuple<Args...>;
  using size_type = typename Container::size_type;

  queue(std::size_t id, Container& container)
      : m_id {id},
        m_container {&container}
  {}

  queue(std::size_t id, Container* container)
      : m_id {id},
        m_container {container}
  {}

  std::size_t get_id() const noexcept { return m_id; }

  /// Posts message to queue. This is marked `const` to pass to lambda by
  /// copy. This throws away messages that are over the capacity of the
  /// container.
  template<typename... Ts>
  RAJA_HOST_DEVICE bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      constexpr size_type header_sz = align(sizeof(msg_header));
      constexpr size_type args_sz   = align(sizeof(msg_args<Args...>));
      constexpr size_type msg_sz    = header_sz + args_sz;
      auto local_size =
          RAJA::atomicAdd<auto_atomic>(&(m_container->m_end), msg_sz);
      if (m_container->m_data != nullptr &&
          local_size + msg_sz <= m_container->m_capacity)
      {
        char* buf = m_container->m_data + local_size;
        new (buf) msg_header {args_sz, m_id, buf + header_sz};
        new (buf + header_sz)
            msg_args<Args...> {args_type(std::forward<Ts>(args)...)};

        return true;
      }

      // If message is not stored, update end to have the correct number of
      // bytes and/or allow for additional smaller messages to be stored
      RAJA::atomicSub<auto_atomic>(&(m_container->m_end), msg_sz);
    }

    return false;
  }

private:
  std::size_t m_id;
  Container* m_container;
};

// TODO: turning off for now
// need to relook at logic
#if 0
template<typename Container>
class queue<Container, RAJA::mpsc_queue_overwrite>
{
public:
  using policy = RAJA::mpsc_queue_overwrite;

  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  queue(Container& container) : m_container {&container} {}

  queue(Container* container) : m_container {container} {}

  /// Posts message to queue. This is marked `const` to pass to lambda by
  /// copy. This overwrites previously stored messages once the number of
  /// messages are over the capacity of the container.
  template<typename... Ts>
  RAJA_HOST_DEVICE bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      auto local_size = RAJA::atomicInc<auto_atomic>(&(m_container->m_size));
      if (m_container->m_data != nullptr)
      {
        m_container->m_data[local_size % m_container->m_capacity] =
            value_type(std::forward<Ts>(args)...);
        return true;
      }
    }

    return false;
  }

private:
  Container* m_container;
};
#endif

}  // namespace messages
}  // namespace RAJA

#endif  // closing endif for header file include guard
