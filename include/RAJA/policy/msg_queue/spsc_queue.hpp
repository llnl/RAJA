/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing implementation for a SPSC
 *          message queue policy. By SPSC, means single-producer
 *          single-consumer. In other words, messages will be
 *          produced from one thread and no atomics needed.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_spsc_queue_HPP
#define RAJA_spsc_queue_HPP

#include <utility>

#include "RAJA/util/concepts.hpp"
#include "RAJA/pattern/atomic.hpp"

#include "RAJA/pattern/messages/msg_header.hpp"
#include "RAJA/policy/msg_queue/policy.hpp"

namespace RAJA
{
namespace messages
{

template<typename Container, typename... Args>
class queue<Container, RAJA::spsc_queue, RAJA::msg_args<Args...>>
{
public:
  using policy = RAJA::spsc_queue;

  using args_type = camp::tuple<Args...>;
  using size_type = typename Container::size_type;

  queue(std::pair<std::size_t, std::size_t> id, Container& container)
      : m_type {id.first},
        m_hash {id.second},
        m_container {&container}
  {}

  queue(std::pair<std::size_t, std::size_t> id, Container* container)
      : m_type {id.first},
        m_hash {id.second},
        m_container {container}
  {}

  auto get_id() const noexcept { return std::make_pair(m_type, m_hash); }

  /// Posts message to queue. This is marked `const` to pass to lambda by
  /// copy. This throws away messages that are over the capacity of the
  /// container.
  template<typename... Ts>
  bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      constexpr size_type header_sz = align_sz(sizeof(msg_header));
      constexpr size_type args_sz   = align_sz(sizeof(msg_args<Args...>));
      constexpr size_type msg_sz    = header_sz + args_sz;
      auto local_size               = m_container->m_end;
      if (m_container->m_data != nullptr &&
          local_size + msg_sz <= m_container->m_capacity)
      {
        m_container->m_end += msg_sz;
        char* buf = m_container->m_data + local_size;
        new (buf) msg_header {args_sz, m_type, m_hash, buf + header_sz};
        new (buf + header_sz)
            msg_args<Args...> {args_type(std::forward<Ts>(args)...)};

        return true;
      }
    }

    return false;
  }

private:
  std::size_t m_type;
  std::size_t m_hash;
  Container* m_container;
};

// TODO: turning off for now
// need to relook at logic
// Previously, this logic was expecting a different msg_queue
// per type of message. In other words, this expected all messages
// to be the same size. Since a more generic version is being
// used, messages cannot easily overwrite old messages with different
// sizes. Therefore, this was commented out for now.
//
// The goal with the `_overwrite` type is to use a circular buffer
// to overwrite old messages if the buffer is full. (4/7/2026)
#if 0
template<typename Container>
class queue<Container, RAJA::spsc_queue_overwrite>
{
public:
  using policy = RAJA::spsc_queue_overwrite;

  using value_type = typename Container::value_type;
  using size_type  = typename Container::size_type;

  queue(Container& container) : m_container {&container} {}

  queue(Container* container) : m_container {container} {}

  /// Posts message to queue. This is marked `const` to pass to lambda by
  /// copy. This overwrites previously stored messages once the number of
  /// messages are over the capacity of the container.
  template<typename... Ts>
  bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      auto local_size = m_container->m_size++;
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
