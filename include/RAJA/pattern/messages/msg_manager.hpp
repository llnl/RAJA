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
#include <unordered_map>

#include "RAJA/util/align.hpp"
#include "RAJA/util/HashCombiner.hpp"
#include "RAJA/util/ResourceAllocator.hpp"

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
template<typename T, typename Allocator>
class MessageBus;

///
/// Specialized case from message bus.
/// This will store a MsgHeader and arguments in a char* buffer. These
/// are later reinterpretted to the correct message arguments.
///
template<typename Allocator>
class MessageBus<char, Allocator>
{
  // Internal classes
public:
  // Queue is public due to limitation with extended lambdas
  // in nvcc
  struct Queue
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
  struct MsgIterator
  {
    using value_type        = char;
    using pointer           = value_type*;
    using reference         = value_type&;
    using difference_type   = std::ptrdiff_t;
    using iterator_categroy = std::forward_iterator_tag;

    MsgIterator(pointer ptr) : cur_ptr(ptr) {}

    MsgHeader& operator*() const
    {
      return *std::launder(reinterpret_cast<MsgHeader*>(cur_ptr));
    }

    MsgHeader* operator->() const
    {
      return std::launder(reinterpret_cast<MsgHeader*>(cur_ptr));
    }

    MsgIterator& operator++()
    {
      MsgHeader& msg = *std::launder(reinterpret_cast<MsgHeader*>(cur_ptr));
      cur_ptr += msg.sz + align_sz(sizeof(MsgHeader));

      return (*this);
    }

    MsgIterator operator++(int)
    {
      MsgIterator temp = *this;
      ++(*this);
      return temp;
    }

    bool operator==(const MsgIterator& other) const
    {
      return (cur_ptr == other.cur_ptr);
    }

    bool operator!=(const MsgIterator& other) const
    {
      return !(*this == other);
    }

  private:
    pointer cur_ptr;
  };

  struct ResourceDeleter
  {
  public:
    using resource_type = camp::resources::Resource;
    using allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<Queue>;

    template<typename Resource>
    ResourceDeleter(Resource res, allocator_type alloc)
        : m_res {res},
          m_alloc {alloc}
    {}

    void operator()(Queue* ptr)
    {
      m_res.wait();
      ptr->~Queue();
      m_alloc.deallocate(ptr, 1);
    }

    allocator_type& get_allocator() noexcept { return m_alloc; }

    resource_type& get_resource() noexcept { return m_res; }

  private:
    resource_type m_res;
    allocator_type m_alloc;
  };

public:
  using value_type     = char;
  using size_type      = unsigned long long;
  using pointer        = value_type*;
  using const_pointer  = const value_type*;
  using iterator       = MsgIterator;
  using const_iterator = const iterator;
  using resource_type  = typename ResourceDeleter::resource_type;
  // Allocator for queue buffer
  using allocator_type = Allocator;
  // Allocator for queue struct
  using queue_allocator = typename ResourceDeleter::allocator_type;

  MessageBus() : MessageBus(camp::resources::Host()) {}

  template<typename Resource>
  MessageBus(Resource res, Allocator alloc = Allocator {})
      : m_bus {new(queue_allocator(alloc).allocate(1)) Queue {},
               ResourceDeleter {res, alloc}}
  {}

  template<typename Resource>
  MessageBus(const size_type bus_sz,
             Resource res,
             Allocator alloc = Allocator {})
      : MessageBus {res, alloc}
  {
    reserve(bus_sz);
  }

  ~MessageBus() { reset(); }

  // Copy ctor/operator
  MessageBus(const MessageBus&)            = delete;
  MessageBus& operator=(const MessageBus&) = delete;

  // Move ctor/operator
  MessageBus(MessageBus&&) = default;

  MessageBus& operator=(MessageBus&& other)
  {
    using alloc_traits = std::allocator_traits<allocator_type>;

    if constexpr (alloc_traits::propagate_on_container_move_assignment)
    {
      m_bus = std::move(other.m_bus);
    }
    else
    {
      if (get_allocator() == other.get_allocator())
      {
        m_bus->m_begin    = other.m_bus->m_begin;
        m_bus->m_end      = other.m_bus->m_end;
        m_bus->m_capacity = other.m_bus->m_capacity;
        m_bus->m_data     = other.m_bus->m_data;

        other.m_bus->m_begin    = 0;
        other.m_bus->m_end      = 0;
        other.m_bus->m_capacity = 0;
        other.m_bus->m_data     = nullptr;
      }
      else
      {
        // This assumes that the bus is empty (i.e., no messages stored)
        reserve(other.m_bus->m_capacity);
        other.reset();
      }
    }
  }

  void reserve(size_type bus_sz)
  {
    reset();
    m_bus->m_data     = get_allocator().allocate(bus_sz);
    m_bus->m_capacity = bus_sz;
  }

  void reset()
  {
    // Verify that queue is not in use
    if (m_bus->m_data != nullptr)
    {
      get_resource().wait();
      get_allocator().deallocate(m_bus->m_data, m_bus->m_capacity);
      m_bus->m_data = nullptr;
    }
    m_bus->m_capacity = 0;
    m_bus->m_end      = 0;
    m_bus->m_begin    = 0;
  }

  bool has_pending_messages()
  {
    get_resource().wait();
    return (m_bus->m_end != 0);
  }

  void clear_messages()
  {
    get_resource().wait();
    m_bus->m_end   = 0;
    m_bus->m_begin = 0;
  }

  template<typename Policy, typename... Args>
  auto get_queue(std::pair<std::size_t, std::size_t> id) noexcept
  {
    return RAJA::messages::Queue<Queue, Policy, RAJA::MsgArgs<Args...>> {
        id, m_bus.get()};
  }

  template<typename Policy, typename... Args>
  auto get_queue(std::pair<std::size_t, std::size_t> id) const noexcept
  {
    return RAJA::messages::Queue<Queue, Policy, RAJA::MsgArgs<Args...>> {
        id, m_bus.get()};
  }

  auto get_allocator() noexcept
  {
    return allocator_type {m_bus.get_deleter().get_allocator()};
  }

  resource_type& get_resource() noexcept
  {
    return m_bus.get_deleter().get_resource();
  }

  iterator begin() noexcept { return iterator {m_bus->m_data}; }

  iterator begin() const noexcept { return iterator {m_bus->m_data}; }

  iterator end() noexcept { return iterator {m_bus->m_data + m_bus->m_end}; }

  iterator end() const noexcept
  {
    return iterator {m_bus->m_data + m_bus->m_end};
  }

private:
  std::unique_ptr<Queue, ResourceDeleter> m_bus;
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
template<typename Allocator>
class MessageManager
{
public:
  using msg_fn_list_t = std::vector<std::unique_ptr<RAJA::IMsgCallback>>;
  using msg_id        = std::pair<std::size_t, std::size_t>;
  using msg_bus       = MessageBus<char, Allocator>;

public:
  template<typename Resource>
  MessageManager(const std::size_t bus_sz, Resource res, Allocator alloc)
      : m_bus {bus_sz, res, alloc}
  {}

  ~MessageManager() = default;

  // Doesn't support copying
  MessageManager(const MessageManager&)            = delete;
  MessageManager& operator=(const MessageManager&) = delete;

  // Move ctor/operator
  MessageManager(MessageManager&&)            = default;
  MessageManager& operator=(MessageManager&&) = default;

  template<typename Policy, typename Callable>
  auto subscribe(Callable&& c)
  {
    msg_id id =
        std::make_pair(m_callback_map.size(), typeid(Callable).hash_code());

    return get_queue_impl<Policy>(
        id, RAJA::MsgCallback {std::forward<Callable>(c)});
  }

  template<typename Callable>
  void subscribe(msg_id id, Callable&& c)
  {
    RAJA::MsgCallback callback {std::forward<Callable>(c)};
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

  void unsubscribe_all(msg_id id) { m_callback_map.erase(id); }

  void unsubscribe_all() { m_callback_map.clear(); }

  void clear() { m_bus.clear_messages(); }

  bool test_any() { return m_bus.has_pending_messages(); }

  auto get_messages()
  {
    std::vector<const MsgHeader*> messages;

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
        msg_id id = std::make_pair(msg->type, msg->hash);
        for (auto& callback : m_callback_map[id])
        {
          (*callback)(msg->args);
        }
        msg->~MsgHeader();
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
        msg_id id = std::make_pair(msg->type, msg->hash);
        for (auto& callback : m_callback_map[id])
        {
          (*callback)(msg->args);
        }
        msg->~MsgHeader();
      }
      messages.clear();
    }
    clear();
  }

private:
  template<typename Policy, typename Callable, typename R, typename... Args>
  auto get_queue_impl(msg_id id, MsgCallback<Callable, R(Args...)>&& c)
  {
    using msg_callback_t = MsgCallback<Callable, R(Args...)>;

    m_callback_map[id].emplace_back(
        std::make_unique<msg_callback_t>(std::move(c)));

    return m_bus.template get_queue<Policy, std::decay_t<Args>...>(id);
  }

  msg_bus m_bus;
  std::unordered_map<msg_id, msg_fn_list_t, RAJA::PairHash> m_callback_map;
};

template<typename Resource,
         typename Allocator = RAJA::ResourceAllocator<char, Resource>>
auto make_message_manager(std::size_t bus_sz,
                          Resource r,
                          Allocator alloc = Allocator {})
{
  return RAJA::MessageManager<Allocator>(bus_sz, r, alloc);
}

template<typename ExecPol,
         typename Allocator = RAJA::ResourceAllocator<
             char,
             decltype(RAJA::resources::get_default_resource<ExecPol>())>>
auto make_message_manager(std::size_t bus_sz, Allocator alloc = Allocator {})
{
  auto r = RAJA::resources::get_default_resource<ExecPol>();
  return RAJA::MessageManager<Allocator>(bus_sz, r, alloc);
}

}  // namespace RAJA

#endif /* RAJA_MSG_MANAGER_HPP */
