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
class Queue<Container, RAJA::spsc_queue, RAJA::MsgArgs<Args...>>
{
public:
  using policy = RAJA::spsc_queue;

  using args_type = camp::tuple<Args...>;
  using size_type = typename Container::size_type;

  Queue(std::pair<std::size_t, std::size_t> id, Container& container)
      : m_type {id.first},
        m_hash {id.second},
        m_container {&container}
  {}

  Queue(std::pair<std::size_t, std::size_t> id, Container* container)
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
      constexpr size_type header_sz = align_sz(sizeof(MsgHeader));
      constexpr size_type args_sz   = align_sz(sizeof(MsgArgs<Args...>));
      constexpr size_type msg_sz    = header_sz + args_sz;
      auto local_size               = m_container->m_end;
      if (m_container->m_data != nullptr &&
          local_size + msg_sz <= m_container->m_capacity)
      {
        m_container->m_end += msg_sz;
        char* buf = m_container->m_data + local_size;
        new (buf) MsgHeader {args_sz, m_type, m_hash, buf + header_sz};
        new (buf + header_sz)
            MsgArgs<Args...> {args_type(std::forward<Ts>(args)...)};

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

}  // namespace messages
}  // namespace RAJA

#endif  // closing endif for header file include guard
