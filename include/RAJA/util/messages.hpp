/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a GPU to CPU message handler class.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MESSAGES_HPP
#define RAJA_MESSAGES_HPP

#include <algorithm>
#include <functional>
#include <memory>

#include "RAJA/util/msg_header.hpp"
#include "RAJA/policy/msg_queue.hpp"

#include "camp/resource.hpp"

namespace RAJA
{
///
/// Owning wrapper for a message queue. This is used for ownership
/// of the message queue and is a move-only class. For getting a view-like
/// class, use the `get_queue` member function, which allows copying.
///
template<typename T>
class message_bus;

///
/// Specialized case from message bus.
/// This will store a msg_header and arguments in a char* buffer. These
/// are later reinterpretted to the correct message arguments.
///
template<>
class message_bus<char>
{
private:
  // Internal classes
  struct queue
  {
    using value_type     = char;
    using size_type      = unsigned long long;
    using pointer        = value_type*;
    using const_pointer  = const value_type*;
    using iterator       = value_type*;
    using const_iterator = const value_type*;

    size_type m_begin {0};
    size_type m_end {0};
    size_type m_capacity {0};
    size_type m_size {0};
    pointer m_data {nullptr};
  };

  struct msg_iterator
  {
    using value_type        = char;
    using pointer           = value_type*;
    using reference         = value_type&;
    using difference_type   = std::ptrdiff_t;
    using iterator_categroy = std::forward_iterator_tag;

    msg_iterator(pointer ptr) : cur_ptr(ptr) {}

    msg_header& operator*() const
    {
      return *std::launder(reinterpret_cast<msg_header*>(cur_ptr));
    }

    msg_header* operator->() const
    {
      return std::launder(reinterpret_cast<msg_header*>(cur_ptr));
    }

    msg_iterator& operator++()
    {
      msg_header& msg = *std::launder(reinterpret_cast<msg_header*>(cur_ptr));
      cur_ptr += msg.sz + align(sizeof(msg_header));

      return (*this);
    }

    msg_iterator operator++(int)
    {
      msg_iterator temp = *this;
      ++(*this);
      return temp;
    }

    bool operator==(const msg_iterator& other) const
    {
      return (cur_ptr == other.cur_ptr);
    }

    bool operator!=(const msg_iterator& other) const
    {
      return !(*this == other);
    }

  private:
    pointer cur_ptr;
  };

  struct resource_deleter
  {
  public:
    using resource_type = camp::resources::Resource;

    template<typename Resource>
    resource_deleter(Resource res) : m_res {res}
    {}

    void operator()(queue* ptr)
    {
      m_res.wait();
      ptr->~queue();
      m_res.deallocate(ptr, camp::resources::MemoryAccess::Pinned);
    }

  private:
    resource_type m_res;
  };

public:
  using value_type     = char;
  using size_type      = unsigned long long;
  using pointer        = value_type*;
  using const_pointer  = const value_type*;
  using iterator       = msg_iterator;
  using const_iterator = const iterator;
  using resource_type  = resource_deleter::resource_type;

  message_bus()
      : m_res {camp::resources::Host()},
        m_bus {new (m_res.allocate<queue>(
                   1,
                   camp::resources::MemoryAccess::Pinned)) queue {},
               resource_deleter {m_res}}
  {}

  template<typename Resource>
  message_bus(Resource res)
      : m_res {res},
        m_bus {new (m_res.allocate<queue>(
                   1,
                   camp::resources::MemoryAccess::Pinned)) queue {},
               resource_deleter {m_res}}
  {}

  template<typename Resource>
  message_bus(const size_type bus_sz, Resource res) : message_bus {res}
  {
    reserve(bus_sz);
  }

  ~message_bus() { reset(); }

  // Copy ctor/operator
  message_bus(const message_bus&)            = delete;
  message_bus& operator=(const message_bus&) = delete;

  // Move ctor/operator
  message_bus(message_bus&&)            = default;
  message_bus& operator=(message_bus&&) = default;

  void reserve(size_type bus_sz)
  {
    reset();
    m_bus->m_data = m_res.allocate<value_type>(
        bus_sz, camp::resources::MemoryAccess::Pinned);
    m_bus->m_capacity = bus_sz;
  }

  void reset()
  {
    // Verify that queue is not in use
    if (m_bus->m_data != nullptr)
    {
      m_res.wait();
      m_res.deallocate(m_bus->m_data, camp::resources::MemoryAccess::Pinned);
      m_bus->m_data = nullptr;
    }
    m_bus->m_capacity = 0;
    m_bus->m_size     = 0;
    m_bus->m_end      = 0;
    m_bus->m_begin    = 0;
  }

  bool has_pending_messages() { return get_num_pending_messages() != 0; }

  size_type get_num_pending_messages()
  {
    m_res.wait();
    return m_bus->m_size;
  }

  void clear_messages()
  {
    m_res.wait();
    m_bus->m_size  = 0;
    m_bus->m_end   = 0;
    m_bus->m_begin = 0;
  }

  template<typename Policy, typename... Args>
  auto get_queue(int id) noexcept
  {
    return RAJA::messages::queue<queue, Policy, RAJA::msg_args<Args...>> {
        id, m_bus.get()};
  }

  template<typename Policy, typename... Args>
  auto get_queue(int id) const noexcept
  {
    return RAJA::messages::queue<queue, Policy, RAJA::msg_args<Args...>> {
        id, m_bus.get()};
  }

  iterator begin() noexcept { return iterator {m_bus->m_data}; }

  iterator end() noexcept
  {
    return iterator {m_bus->m_data + get_num_pending_messages()};
  }

private:
  resource_type m_res;
  std::unique_ptr<queue, resource_deleter> m_bus;
};

///
/// Provides a way to handle messages from a GPU. This currently
/// stores messages from the GPU and then calls a callback
/// function from the host.
///
/// Note:
/// Currently, this forces a synchronize prior to calling
/// the callback function or testing if there are any messages.
///
class message_manager
{
public:
  using callback_type = std::function<void(char*)>;
  using msg_id        = int;
  using msg_bus       = message_bus<char>;

  template<typename T>
  using msg_decay_t = std::decay_t<T>;

public:
  template<typename Resource>
  message_manager(const std::size_t bus_sz, Resource res) : m_bus {bus_sz, res}
  {}

  ~message_manager() = default;

  // Doesn't support copying
  message_manager(const message_manager&)            = delete;
  message_manager& operator=(const message_manager&) = delete;

  // Move ctor/operator
  message_manager(message_manager&&)            = default;
  message_manager& operator=(message_manager&&) = default;

  template<typename Policy, typename Callable>
  auto get_queue(msg_id id, Callable&& c)
  {
    return get_queue_impl<Policy>(id, std::forward<Callable>(c),
                                  std::function {std::forward<Callable>(c)});
  }

  void clear() { m_bus.clear_messages(); }

  bool test_any() { return m_bus.has_pending_messages(); }

  void wait_all()
  {
    if (test_any())
    {
      for (auto& msg : m_bus)
      {
        m_callbacks[msg.id](msg.args);
      }
      clear();
    }
  }

private:
  // TODO: create small wrapper for callables
  template<typename Policy, typename Callable, typename R, typename... Args>
  auto get_queue_impl(msg_id id, Callable&& c, std::function<R(Args...)>)
  {
    m_callbacks[id] = callback_type {[=](char* msg_args_buf) {
      msg_args<msg_decay_t<Args>...>& aligned_args = *std::launder(
          reinterpret_cast<msg_args<msg_decay_t<Args>...>*>(msg_args_buf));
      camp::apply(c, aligned_args);
      aligned_args.~msg_args<msg_decay_t<Args>...>();
    }};
    return m_bus.template get_queue<Policy, msg_decay_t<Args>...>(id);
  }

  msg_bus m_bus;
  std::map<msg_id, callback_type> m_callbacks;
};

template<typename Resource>
auto make_message_manager(std::size_t bus_sz, Resource r)
{
  return RAJA::message_manager(bus_sz, r);
}

template<typename ExecPol>
auto make_message_manager(std::size_t bus_sz)
{
  auto r = RAJA::resources::get_default_resource<ExecPol>();
  return RAJA::message_manager(bus_sz, r);
}

}  // namespace RAJA

#endif /* RAJA_MESSAGES_HPP */
