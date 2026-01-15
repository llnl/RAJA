/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA Dispatcher for workgroup.
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

#ifndef RAJA_PATTERN_WORKGROUP_Dispatcher_HPP
#define RAJA_PATTERN_WORKGROUP_Dispatcher_HPP


#include "RAJA/config.hpp"

#include "RAJA/policy/WorkGroup.hpp"

#include "camp/number.hpp"
#include "camp/list.hpp"
#include "camp/helpers.hpp"

#include <utility>

namespace RAJA
{

namespace detail
{

template<typename>
struct DispatcherVoidPtrWrapper
{
  void* ptr;
  DispatcherVoidPtrWrapper() = default;

  // implicit constructor from void*
  RAJA_HOST_DEVICE DispatcherVoidPtrWrapper(void* p) : ptr(p) {}
};

template<typename>
struct DispatcherVoidConstPtrWrapper
{
  const void* ptr;
  DispatcherVoidConstPtrWrapper() = default;

  // implicit constructor from const void*
  RAJA_HOST_DEVICE DispatcherVoidConstPtrWrapper(const void* p) : ptr(p) {}
};

constexpr bool dispatcher_use_host_invoke(Platform platform)
{
  return !(platform == Platform::cuda || platform == Platform::hip);
}

// Transforms one dispatch policy into another by creating a dispatch policy
// of holder_type objects. See usage in WorkRunner for more explanation.
template<typename dispatch_policy, typename holder_type>
struct dispatcher_transform_types;
///
template<typename dispatch_policy, typename holder_type>
using dispatcher_transform_types_t =
    typename dispatcher_transform_types<dispatch_policy, holder_type>::type;

/*!
 * A dispatcher abstraction that provides an interface to some basic
 * functionality that is implemented differently based on the dispatch_policy.
 *
 * DispatcherID is used to differentiate function pointers based on their
 * function signature.
 */
template<Platform platform,
         typename dispatch_policy,
         typename DispatcherID,
         typename... CallArgs>
struct Dispatcher;

template<typename holder_type>
struct dispatcher_transform_types<::RAJA::indirect_function_call_dispatch,
                                  holder_type>
{
  using type = ::RAJA::indirect_function_call_dispatch;
};

/*!
 * Version of Dispatcher that acts essentially like a vtable. It implements
 * the interface with function pointers.
 *
 * DispatcherID can be helpful to avoid function signature collisions
 * with functions that will not be used through this class. This is useful
 * during device linking when functions with high register counts may cause
 * device linking to fail.
 */
template<Platform platform, typename DispatcherID, typename... CallArgs>
struct Dispatcher<platform,
                  ::RAJA::indirect_function_call_dispatch,
                  DispatcherID,
                  CallArgs...>
{
  static constexpr bool use_host_invoke = dispatcher_use_host_invoke(platform);
  using dispatch_policy   = ::RAJA::indirect_function_call_dispatch;
  using void_ptr_wrapper  = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  template<typename T>
  static void s_move_construct_destroy(void_ptr_wrapper dest,
                                       void_ptr_wrapper src)
  {
    T* dest_as_T = static_cast<T*>(dest.ptr);
    T* src_as_T  = static_cast<T*>(src.ptr);
    new (dest_as_T) T(std::move(*src_as_T));
    (*src_as_T).~T();
  }

  ///
  /// invoke the call operator of the object of type T in obj with args
  ///
  template<typename T>
  static void s_host_invoke(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }

  ///
  template<typename T>
  static RAJA_DEVICE void s_device_invoke(void_cptr_wrapper obj,
                                          CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }

  ///
  /// destroy the object of type T in obj
  ///
  template<typename T>
  static void s_destroy(void_ptr_wrapper obj)
  {
    T* obj_as_T = static_cast<T*>(obj.ptr);
    (*obj_as_T).~T();
  }

  using mover_type     = void (*)(void_ptr_wrapper /*dest*/,
                              void_ptr_wrapper /*src*/);
  using invoker_type   = void (*)(void_cptr_wrapper /*obj*/,
                                CallArgs... /*args*/);
  using destroyer_type = void (*)(void_ptr_wrapper /*obj*/);

  // This can't be a cuda device lambda due to compiler limitations
  template<typename T>
  struct DeviceInvokerFactory
  {
    using value_type = invoker_type;

    RAJA_DEVICE value_type operator()()
    {
#if defined(RAJA_ENABLE_HIP) && !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)
      return nullptr;
#else
      return &s_device_invoke<T>;
#endif
    }
  };

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename T,
           bool uhi               = use_host_invoke,
           std::enable_if_t<uhi>* = nullptr>
  static inline Dispatcher makeDispatcher()
  {
    return {mover_type {&s_move_construct_destroy<T>},
            invoker_type {&s_host_invoke<T>}, destroyer_type {&s_destroy<T>},
            sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// To do this the invoker_type must be created on the device to get the
  /// device function pointer. The createOnDevice parameter is responsible for
  /// providing the device context and returning the invoker object created.
  /// The createOnDevice object uses an invoker factory provided as an argument
  /// to create the invoker object. This allows for a separation between
  /// object creation and the device context (cuda, hip, etc) and copying.
  ///
  template<typename T,
           typename CreateOnDevice,
           bool uhi                = use_host_invoke,
           std::enable_if_t<!uhi>* = nullptr>
  static inline Dispatcher makeDispatcher(CreateOnDevice&& createOnDevice)
  {
    return {mover_type {&s_move_construct_destroy<T>},
            invoker_type {std::forward<CreateOnDevice>(createOnDevice)(
                DeviceInvokerFactory<T> {})},
            destroyer_type {&s_destroy<T>}, sizeof(T)};
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

template<typename holder_type>
struct dispatcher_transform_types<::RAJA::indirect_virtual_function_dispatch,
                                  holder_type>
{
  using type = ::RAJA::indirect_virtual_function_dispatch;
};

/*!
 * Version of Dispatcher that uses a class hierarchy and virtual functions to
 * implement the interface.
 *
 * DispatcherID can be helpful to avoid function signature collisions
 * with functions that will not be used through this class. This is useful
 * during device linking when functions with high register counts may cause
 * device linking to fail.
 */
template<Platform platform, typename DispatcherID, typename... CallArgs>
struct Dispatcher<platform,
                  ::RAJA::indirect_virtual_function_dispatch,
                  DispatcherID,
                  CallArgs...>
{
  static constexpr bool use_host_invoke = dispatcher_use_host_invoke(platform);
  using dispatch_policy   = ::RAJA::indirect_virtual_function_dispatch;
  using void_ptr_wrapper  = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  struct impl_base
  {
    virtual void move_destroy(void_ptr_wrapper dest,
                              void_ptr_wrapper src) const = 0;
    virtual void destroy(void_ptr_wrapper obj) const      = 0;
  };

  struct host_impl_base
  {
    virtual void invoke(void_cptr_wrapper obj, CallArgs... args) const = 0;
  };

  struct device_impl_base
  {
    virtual RAJA_DEVICE void invoke(void_cptr_wrapper obj,
                                    CallArgs... args) const = 0;
  };

  template<typename T>
  struct base_impl_type : impl_base
  {
    ///
    /// move construct an object of type T in dest as a copy of a T from src and
    /// destroy the T obj in src
    ///
    virtual void move_destroy(void_ptr_wrapper dest,
                              void_ptr_wrapper src) const override
    {
      T* dest_as_T = static_cast<T*>(dest.ptr);
      T* src_as_T  = static_cast<T*>(src.ptr);
      new (dest_as_T) T(std::move(*src_as_T));
      (*src_as_T).~T();
    }

    ///
    /// destroy the object of type T in obj
    ///
    virtual void destroy(void_ptr_wrapper obj) const override
    {
      T* obj_as_T = static_cast<T*>(obj.ptr);
      (*obj_as_T).~T();
    }
  };

  template<typename T>
  struct host_impl_type : host_impl_base
  {
    ///
    /// invoke the call operator of the object of type T in obj with args
    ///
    virtual void invoke(void_cptr_wrapper obj, CallArgs... args) const override
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  template<typename T>
  struct device_impl_type : device_impl_base
  {
    ///
    /// invoke the call operator of the object of type T in obj with args
    ///
    virtual RAJA_DEVICE void invoke(void_cptr_wrapper obj,
                                    CallArgs... args) const override
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  struct mover_type
  {
    impl_base* m_impl;

    void operator()(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      m_impl->move_destroy(dest, src);
    }
  };

  struct host_invoker_type
  {
    host_impl_base* m_impl;

    void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      m_impl->invoke(obj, std::forward<CallArgs>(args)...);
    }
  };

  ///
  struct device_invoker_type
  {
    device_impl_base* m_impl;

    RAJA_DEVICE void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      m_impl->invoke(obj, std::forward<CallArgs>(args)...);
    }
  };

  using invoker_type = std::
      conditional_t<use_host_invoke, host_invoker_type, device_invoker_type>;

  struct destroyer_type
  {
    impl_base* m_impl;

    void operator()(void_ptr_wrapper obj) const { m_impl->destroy(obj); }
  };

  // This can't be a cuda device lambda due to compiler limitations
  template<typename T>
  struct DeviceImplTypeFactory
  {
    using value_type = device_impl_type<T>*;

    RAJA_DEVICE value_type operator()()
    {
#if defined(RAJA_ENABLE_HIP) && !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)
      return nullptr;
#else
      static device_impl_type<T> s_device_impl;
      return &s_device_impl;
#endif
    }
  };

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename T,
           bool uhi               = use_host_invoke,
           std::enable_if_t<uhi>* = nullptr>
  static inline Dispatcher makeDispatcher()
  {
    static base_impl_type<T> s_base_impl;
    static host_impl_type<T> s_host_impl;
    return {mover_type {&s_base_impl}, host_invoker_type {&s_host_impl},
            destroyer_type {&s_base_impl}, sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// To do this the invoker_type must be created on the device to get the
  /// device function pointer. The createOnDevice parameter is responsible for
  /// providing the device context and returning the invoker object created.
  /// The createOnDevice object uses an invoker factory provided as an argument
  /// to create the invoker object. This allows for a separation between
  /// object creation and the device context (cuda, hip, etc) and copying.
  ///
  template<typename T,
           typename CreateOnDevice,
           bool uhi                = use_host_invoke,
           std::enable_if_t<!uhi>* = nullptr>
  static inline Dispatcher makeDispatcher(CreateOnDevice&& createOnDevice)
  {
    static base_impl_type<T> s_base_impl;
    static device_impl_type<T>* s_device_impl_ptr {std::forward<CreateOnDevice>(
        createOnDevice)(DeviceImplTypeFactory<T> {})};
    return {mover_type {&s_base_impl}, device_invoker_type {s_device_impl_ptr},
            destroyer_type {&s_base_impl}, sizeof(T)};
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

// direct_dispatch expects a list of types
template<typename... Ts, typename holder_type>
struct dispatcher_transform_types<::RAJA::direct_dispatch<Ts...>, holder_type>
{
  using type =
      ::RAJA::direct_dispatch<typename holder_type::template type<Ts>...>;
};

/*!
 * Version of Dispatcher that does direct dispatch to zero callable types.
 * It implements the interface with callable objects.
 */
template<Platform platform, typename DispatcherID, typename... CallArgs>
struct Dispatcher<platform,
                  ::RAJA::direct_dispatch<>,
                  DispatcherID,
                  CallArgs...>
{
  static constexpr bool use_host_invoke = dispatcher_use_host_invoke(platform);
  using dispatch_policy                 = ::RAJA::direct_dispatch<>;
  using void_ptr_wrapper  = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  struct mover_type
  {
    void operator()(void_ptr_wrapper, void_ptr_wrapper) const {}
  };

  ///
  /// invoke the call operator of the object of type T in obj with args
  ///
  struct host_invoker_type
  {
    void operator()(void_cptr_wrapper, CallArgs...) const {}
  };

  struct device_invoker_type
  {
    RAJA_DEVICE void operator()(void_cptr_wrapper, CallArgs...) const {}
  };

  using invoker_type = std::
      conditional_t<use_host_invoke, host_invoker_type, device_invoker_type>;

  ///
  /// destroy the object of type T in obj
  ///
  struct destroyer_type
  {
    void operator()(void_ptr_wrapper) const {}
  };

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename T,
           bool uhi               = use_host_invoke,
           std::enable_if_t<uhi>* = nullptr>
  static inline Dispatcher makeDispatcher()
  {
    return {mover_type {}, host_invoker_type {}, destroyer_type {}, sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// Ignore the CreateOnDevice object as the same invoker object can be used
  /// on the host and device.
  ///
  template<typename T,
           typename CreateOnDevice,
           bool uhi                = use_host_invoke,
           std::enable_if_t<!uhi>* = nullptr>
  static inline Dispatcher makeDispatcher(CreateOnDevice&&)
  {
    return {mover_type {}, device_invoker_type {}, destroyer_type {},
            sizeof(T)};
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

/*!
 * Version of Dispatcher that does direct dispatch to a single callable type.
 * It implements the interface with callable objects.
 */
template<Platform platform,
         typename T,
         typename DispatcherID,
         typename... CallArgs>
struct Dispatcher<platform,
                  ::RAJA::direct_dispatch<T>,
                  DispatcherID,
                  CallArgs...>
{
  static constexpr bool use_host_invoke = dispatcher_use_host_invoke(platform);
  using dispatch_policy                 = ::RAJA::direct_dispatch<T>;
  using void_ptr_wrapper  = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  struct mover_type
  {
    void operator()(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      T* dest_as_T = static_cast<T*>(dest.ptr);
      T* src_as_T  = static_cast<T*>(src.ptr);
      new (dest_as_T) T(std::move(*src_as_T));
      (*src_as_T).~T();
    }
  };

  ///
  /// invoke the call operator of the object of type T in obj with args
  ///
  struct host_invoker_type
  {
    void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  struct device_invoker_type
  {
    RAJA_DEVICE void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  using invoker_type = std::
      conditional_t<use_host_invoke, host_invoker_type, device_invoker_type>;

  ///
  /// destroy the object of type T in obj
  ///
  struct destroyer_type
  {
    void operator()(void_ptr_wrapper obj) const
    {
      T* obj_as_T = static_cast<T*>(obj.ptr);
      (*obj_as_T).~T();
    }
  };

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename U,
           bool uhi               = use_host_invoke,
           std::enable_if_t<uhi>* = nullptr>
  static inline Dispatcher makeDispatcher()
  {
    static_assert(std::is_same<T, U>::value,
                  "U must be in direct_dispatch types");
    return {mover_type {}, host_invoker_type {}, destroyer_type {}, sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// Ignore the CreateOnDevice object as the same invoker object can be used
  /// on the host and device.
  ///
  template<typename U,
           typename CreateOnDevice,
           bool uhi                = use_host_invoke,
           std::enable_if_t<!uhi>* = nullptr>
  static inline Dispatcher makeDispatcher(CreateOnDevice&&)
  {
    static_assert(std::is_same<T, U>::value,
                  "U must be in direct_dispatch types");
    return {mover_type {}, device_invoker_type {}, destroyer_type {},
            sizeof(T)};
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

/*!
 * Version of Dispatcher that does direct dispatch to multiple callable types.
 * It implements the interface with callable objects.
 */
template<typename T0,
         typename T1,
         typename... TNs,
         Platform platform,
         typename DispatcherID,
         typename... CallArgs>
struct Dispatcher<platform,
                  ::RAJA::direct_dispatch<T0, T1, TNs...>,
                  DispatcherID,
                  CallArgs...>
{
  static constexpr bool use_host_invoke = dispatcher_use_host_invoke(platform);
  using dispatch_policy   = ::RAJA::direct_dispatch<T0, T1, TNs...>;
  using void_ptr_wrapper  = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  using id_type          = int;
  using callable_indices = camp::make_int_seq_t<id_type, 2 + sizeof...(TNs)>;
  using callable_types   = camp::list<T0, T1, TNs...>;

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  struct mover_type
  {
    id_type id;

    void operator()(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      impl_helper(callable_indices {}, callable_types {}, dest, src);
    }

  private:
    template<int... id_types, typename... Ts>
    void impl_helper(camp::int_seq<int, id_types...>,
                     camp::list<Ts...>,
                     void_ptr_wrapper dest,
                     void_ptr_wrapper src) const
    {
      camp::sink(((id_types == id) ? (impl<Ts>(dest, src), 0) : 0)...);
    }

    template<typename T>
    void impl(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      T* dest_as_T = static_cast<T*>(dest.ptr);
      T* src_as_T  = static_cast<T*>(src.ptr);
      new (dest_as_T) T(std::move(*src_as_T));
      (*src_as_T).~T();
    }
  };

  ///
  /// invoke the call operator of the object of type T in obj with args
  ///
  struct host_invoker_type
  {
    id_type id;

    void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      impl_helper(callable_indices {}, callable_types {}, obj,
                  std::forward<CallArgs>(args)...);
    }

  private:
    template<int... id_types, typename... Ts>
    void impl_helper(camp::int_seq<int, id_types...>,
                     camp::list<Ts...>,
                     void_cptr_wrapper obj,
                     CallArgs... args) const
    {
      camp::sink(((id_types == id)
                      ? (impl<Ts>(obj, std::forward<CallArgs>(args)...), 0)
                      : 0)...);
    }

    template<typename T>
    void impl(void_cptr_wrapper obj, CallArgs... args) const
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  struct device_invoker_type
  {
    id_type id;

    RAJA_DEVICE void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      impl_helper(callable_indices {}, callable_types {}, obj,
                  std::forward<CallArgs>(args)...);
    }

  private:
    template<int... id_types, typename... Ts>
    RAJA_DEVICE void impl_helper(camp::int_seq<int, id_types...>,
                                 camp::list<Ts...>,
                                 void_cptr_wrapper obj,
                                 CallArgs... args) const
    {
      camp::sink(((id_types == id)
                      ? (impl<Ts>(obj, std::forward<CallArgs>(args)...), 0)
                      : 0)...);
    }

    template<typename T>
    RAJA_DEVICE void impl(void_cptr_wrapper obj, CallArgs... args) const
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  using invoker_type = std::
      conditional_t<use_host_invoke, host_invoker_type, device_invoker_type>;

  ///
  /// destroy the object of type T in obj
  ///
  struct destroyer_type
  {
    id_type id;

    void operator()(void_ptr_wrapper obj) const
    {
      impl_helper(callable_indices {}, callable_types {}, obj);
    }

  private:
    template<int... id_types, typename... Ts>
    void impl_helper(camp::int_seq<int, id_types...>,
                     camp::list<Ts...>,
                     void_ptr_wrapper obj) const
    {
      camp::sink(((id_types == id) ? (impl<Ts>(obj), 0) : 0)...);
    }

    template<typename T>
    void impl(void_ptr_wrapper obj) const
    {
      T* obj_as_T = static_cast<T*>(obj.ptr);
      (*obj_as_T).~T();
    }
  };

  ///
  /// get the id of type T
  ///
  /// The id is just the index of T in the list of callable_types.
  /// If T is not in Ts return -1.
  ///
  template<typename T, int... id_types, typename... Ts>
  static constexpr id_type get_id(camp::int_seq<int, id_types...>,
                                  camp::list<Ts...>)
  {
    id_type id {-1};
    // quiet UB warning by sequencing assignment to id with list initialization
    int unused[] {0,
                  (std::is_same<T, Ts>::value ? ((id = id_types), 0) : 0)...};
    camp::sink(unused);  // quiet unused var warning
    return id;
  }

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename T,
           bool uhi               = use_host_invoke,
           std::enable_if_t<uhi>* = nullptr>
  static inline Dispatcher makeDispatcher()
  {
    static constexpr id_type id =
        get_id<T>(callable_indices {}, callable_types {});
    static_assert(id != id_type(-1), "T must be in direct_dispatch types");
    return {mover_type {id}, host_invoker_type {id}, destroyer_type {id},
            sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// Ignore the CreateOnDevice object as the same invoker object can be used
  /// on the host and device.
  ///
  template<typename T,
           typename CreateOnDevice,
           bool uhi                = use_host_invoke,
           std::enable_if_t<!uhi>* = nullptr>
  static inline Dispatcher makeDispatcher(CreateOnDevice&&)
  {
    static constexpr id_type id =
        get_id<T>(callable_indices {}, callable_types {});
    static_assert(id != id_type(-1), "T must be in direct_dispatch types");
    return {mover_type {id}, device_invoker_type {id}, destroyer_type {id},
            sizeof(T)};
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

/*!
 * Populate and return a pointer to a Dispatcher object for the given policy.
 * NOTE: there is a function overload is in each policy/WorkGroup/Dispatcher.hpp
 */
// template < typename T, typename Dispatcher_T >
// inline const Dispatcher_T* get_Dispatcher(work_policy const&);

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
