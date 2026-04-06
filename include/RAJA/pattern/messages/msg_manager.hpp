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
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MSG_MANAGER_HPP
#define RAJA_MSG_MANAGER_HPP

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "RAJA/util/align.hpp"

#include "RAJA/pattern/messages/msg_header.hpp"
#include "RAJA/pattern/messages/msg_callback.hpp"
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
  // Internal classes
public:
  // queue is public due to limitation with extended lambdas
  // in nvcc
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
    pointer m_data {nullptr};
  };

private:
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
      cur_ptr += msg.sz + align_sz(sizeof(msg_header));

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

  message_bus() : message_bus(camp::resources::Host()) {}

  template<typename Resource>
  message_bus(Resource res)
      : m_res {res},
        m_bus {
            new(m_res.allocate<queue>(1, camp::resources::MemoryAccess::Pinned))
                queue {},
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
    m_bus->m_end      = 0;
    m_bus->m_begin    = 0;
  }

  bool has_pending_messages()
  {
    m_res.wait();
    return (m_bus->m_end != 0);
  }

  void clear_messages()
  {
    m_res.wait();
    m_bus->m_end   = 0;
    m_bus->m_begin = 0;
  }

  template<typename Policy, typename... Args>
  auto get_queue(std::size_t id) noexcept
  {
    return RAJA::messages::queue<queue, Policy, RAJA::msg_args<Args...>> {
        id, m_bus.get()};
  }

  template<typename Policy, typename... Args>
  auto get_queue(std::size_t id) const noexcept
  {
    return RAJA::messages::queue<queue, Policy, RAJA::msg_args<Args...>> {
        id, m_bus.get()};
  }

  iterator begin() noexcept { return iterator {m_bus->m_data}; }

  iterator begin() const noexcept { return iterator {m_bus->m_data}; }

  iterator end() noexcept { return iterator {m_bus->m_data + m_bus->m_end}; }

  iterator end() const noexcept
  {
    return iterator {m_bus->m_data + m_bus->m_end};
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
  using msg_callback_t = std::unique_ptr<RAJA::imsg_callback>;
  using msg_fn_list_t  = std::vector<msg_callback_t>;
  using msg_id         = std::size_t;
  using msg_bus        = message_bus<char>;

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
  auto subscribe(Callable&& c)
  {
    msg_id id = m_callback_map.size();

    // Create new callback list
    m_callback_map.emplace_back();

    return get_queue_impl<Policy>(
        id, RAJA::msg_callback {std::forward<Callable>(c)});
  }

  template<typename Callable>
  void subscribe(msg_id id, Callable&& c)
  {
    RAJA::msg_callback callback {std::forward<Callable>(c)};
    auto& fn_list = m_callback_map.at(id);
    auto it = std::find_if(fn_list.begin(), fn_list.end(), [](const auto& fn) {
      return std::type_index {typeid(Callable)} == fn->get_type();
    });

    using msg_callback_t = decltype(callback);
    if (it != fn_list.end())
    {
      // TODO: would it be better to throw or just replace old one?
      *it = std::make_unique<msg_callback_t>(std::move(callback));
    }
    else
    {
      fn_list.emplace_back(
          std::make_unique<msg_callback_t>(std::move(callback)));
    }
  }

  template<typename Callable>
  void unsubscribe(msg_id id, Callable&&)
  {
    auto& fn_list = m_callback_map.at(id);
    auto it = std::find_if(fn_list.begin(), fn_list.end(), [](const auto& fn) {
      return std::type_index {typeid(Callable)} == fn->get_type();
    });

    if (it != fn_list.end())
    {
      fn_list.erase(it);
    }
    else
    {
      throw std::runtime_error("Callable is not subscribed");
    }
  }

  void unsubscribe_all(msg_id id) { m_callback_map.at(id).clear(); }

  void unsubscribe_all() { m_callback_map.clear(); }

  void clear() { m_bus.clear_messages(); }

  bool test_any() { return m_bus.has_pending_messages(); }

  auto get_messages()
  {
    std::vector<const msg_header*> messages;

    if (test_any())
    {
      for (const auto& msg : m_bus)
      {
        messages.emplace_back(&msg);
      }
    }

    return messages;
  }

  /**
   * This takes in a container of messages and applies them to the
   * callbacks. Once messages are handled, then container is cleared.
   */
  template<typename Container>
  void handle_all(Container& messages)
  {
    if (!m_callback_map.empty())
    {
      for (const auto& msg : messages)
      {
        for (auto& callback : m_callback_map[msg->id])
        {
          (*callback)(msg->args);
        }
        msg->~msg_header();
      }
      messages.clear();
    }
    clear();
  }

  void wait_all()
  {
    if (!m_callback_map.empty())
    {
      auto messages = get_messages();
      for (const auto& msg : messages)
      {
        for (auto& callback : m_callback_map[msg->id])
        {
          (*callback)(msg->args);
        }
        msg->~msg_header();
      }
      messages.clear();
    }
    clear();
  }

private:
  template<typename Policy, typename Callable, typename R, typename... Args>
  auto get_queue_impl(msg_id id, msg_callback<Callable, R(Args...)>&& c)
  {
    using msg_callback_t = msg_callback<Callable, R(Args...)>;

    m_callback_map[id].emplace_back(
        std::make_unique<msg_callback_t>(std::move(c)));

    return m_bus.template get_queue<Policy, std::decay_t<Args>...>(id);
  }

  msg_bus m_bus;
  std::vector<msg_fn_list_t> m_callback_map;
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

#endif /* RAJA_MSG_MANAGER_HPP */
