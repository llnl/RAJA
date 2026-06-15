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

#include <utility>

#include "RAJA/util/align.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/pattern/atomic.hpp"

#include "RAJA/pattern/messages/msg_header.hpp"
#include "RAJA/policy/msg_queue/policy.hpp"

namespace RAJA
{
namespace messages
{

template<typename Container, typename... Args>
class Queue<Container, RAJA::mpsc_queue, RAJA::MsgArgs<Args...>>
{
public:
  using policy = RAJA::mpsc_queue;

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
  RAJA_HOST_DEVICE bool try_post_message(Ts&&... args) const
  {
    if (m_container != nullptr)
    {
      constexpr size_type header_sz = align_sz(sizeof(MsgHeader));
      constexpr size_type args_sz   = align_sz(sizeof(MsgArgs<Args...>));
      constexpr size_type msg_sz    = header_sz + args_sz;

      size_type local_sz;
      size_type old_sz = RAJA::atomicLoad<auto_atomic>(&(m_container->m_end));

      // Checks if message can fit in queue. If so, adds msg_sz to end of queue
      // to reserve space. Otherwise, message doesn't fit and no space is
      // reserved. In other words, the CAS-loop below performs the follwing
      // operation:
      // (*address + msg_sz <= capacity) ?  (*address + msg_sz) : *address;
      do
      {
        size_type new_sz = old_sz + msg_sz;
        local_sz         = old_sz;  // offset to start of message
        // Checks if fits in queue
        if (new_sz <= m_container->m_capacity)
        {
          old_sz = RAJA::atomicCAS<auto_atomic>(&(m_container->m_end), local_sz,
                                                new_sz);
        }
      } while (local_sz != old_sz);

      if (m_container->m_data != nullptr &&
          local_sz + msg_sz <= m_container->m_capacity)
      {
        char* buf = m_container->m_data + local_sz;
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
