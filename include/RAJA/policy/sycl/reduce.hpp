/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for SYCL reduction stucts/classes.
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

#ifndef RAJA_sycl_reduce_HPP
#define RAJA_sycl_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>
#include <stdexcept>
#include <string>

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/sycl/policy.hpp"

#include "camp/resource/sycl.hpp"

namespace RAJA
{

namespace sycl
{

constexpr bool uses_runtime_offload(Policy p)
{
  return policy_supported(PolicyList<Policy::sycl> {}, p);
}

template<typename T, typename I>
struct minloc
{
  RAJA_HOST_DEVICE static constexpr T identity()
  {
    return ::RAJA::operators::limits<T>::max();
  }

  RAJA_HOST_DEVICE RAJA_INLINE void operator()(T& val,
                                               I& loc,
                                               const T v,
                                               const I l)
  {
    if (v < val)
    {
      loc = l;
      val = v;
    }
  }
};

template<typename T, typename I>
struct maxloc
{
  RAJA_HOST_DEVICE static constexpr T identity()
  {
    return ::RAJA::operators::limits<T>::min();
  }

  RAJA_HOST_DEVICE RAJA_INLINE void operator()(T& val,
                                               I& loc,
                                               const T v,
                                               const I l)
  {
    if (v > val)
    {
      loc = l;
      val = v;
    }
  }
};

// Alias for clarity. Reduction size operates on number of  teams.
// Ideally, MaxNumTeams = ThreadsPerTeam in omp_target_parallel_for_exec.
static int MaxNumTeams = 1;

//! Information necessary for SYCL offload to be considered
struct Offload_Info
{
  int hostID {1};
  int deviceID {2};
  bool isMapped {false};

  Offload_Info() = default;

  Offload_Info(const Offload_Info& other)
      : hostID {other.hostID},
        deviceID {other.deviceID},
        isMapped {other.isMapped}
  {}
};

//! Reduction data for SYCL Offload -- stores value, host pointer, and device
//! pointer
template<typename T>
struct Reduce_Data
{
  mutable T value;
  T identity;
  T* device;
  T* host;

  //! disallow default constructor
  Reduce_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  Reduce_Data(T initValue,
              T identityValue,
              Offload_Info& info,
              bool use_offload = true)
      : value(initValue),
        identity(identityValue),
        device(nullptr),
        host(nullptr)
  {
    if (!use_offload)
    {
      return;
    }

    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();

    device = reinterpret_cast<T*>(
        ::sycl::malloc_device(sycl::MaxNumTeams * sizeof(T), q));
    host = reinterpret_cast<T*>(
        ::sycl::malloc_host(sycl::MaxNumTeams * sizeof(T), q));

    if (!host)
    {
      printf("Unable to allocate space on host\n");
      exit(1);
    }
    if (!device)
    {
      printf("Unable to allocate space on device\n");
      exit(1);
    }
    std::fill_n(host, sycl::MaxNumTeams, identityValue);
    hostToDevice(info);
  }

  void reset(T initValue) { value = initValue; }

  bool uses_offload() const { return device != nullptr; }

  //! default copy constructor for POD
  Reduce_Data(const Reduce_Data&) = default;

  //! default copy operator for POD
  Reduce_Data& operator=(const Reduce_Data&) = default;

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE void hostToDevice(Offload_Info& RAJA_UNUSED_ARG(info))
  {
    if (!device || !host)
    {
      return;
    }

    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();

    // precondition: host and device are valid pointers
    auto e =
        q.memcpy(reinterpret_cast<void*>(device), reinterpret_cast<void*>(host),
                 sycl::MaxNumTeams * sizeof(T));

    e.wait();
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE void deviceToHost(Offload_Info& RAJA_UNUSED_ARG(info))
  {
    if (!device || !host)
    {
      return;
    }

    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();

    // precondition: host and device are valid pointers
    auto e =
        q.memcpy(reinterpret_cast<void*>(host), reinterpret_cast<void*>(device),
                 sycl::MaxNumTeams * sizeof(T));

    e.wait();
  }

  //! frees all data from the offload information passed
  RAJA_INLINE void cleanup(Offload_Info& RAJA_UNUSED_ARG(info))
  {
    if (!device && !host)
    {
      return;
    }

    auto resource    = ::camp::resources::Sycl::get_default();
    ::sycl::queue& q = resource.get_queue();

    if (device)
    {
      ::sycl::free(reinterpret_cast<void*>(device), q);
      device = nullptr;
    }
    if (host)
    {
      ::sycl::free(reinterpret_cast<void*>(host), q);
      // delete[] host;
      host = nullptr;
    }
  }

  RAJA_INLINE void reset_device(T identityValue, Offload_Info& info)
  {
    if (!uses_offload())
    {
      return;
    }

    std::fill_n(host, sycl::MaxNumTeams, identityValue);
    hostToDevice(info);
  }
};

template<typename T>
struct Shared_Host_Data
{
  T hostVal;
  T identity;
  const void* rootToken;
};

template<typename T, typename IndexType>
struct Shared_Host_Loc_Data
{
  T hostVal;
  IndexType hostLoc;
  T identity;
  IndexType identityLoc;
  const void* rootToken;
};

}  // end namespace sycl

//! SYCL Target Reduction entity -- generalize on # of teams, reduction, and
//! type
// This is the last layer of the user facing Reduction API
template<typename Reducer, typename T>
struct TargetReduce
{
  TargetReduce()
      : TargetReduce(Policy::undefined,
                     Reducer::identity(),
                     Reducer::identity())
  {}

  explicit TargetReduce(Policy p)
      : TargetReduce(p, Reducer::identity(), Reducer::identity())
  {}

  explicit TargetReduce(T init_val, T identity_ = Reducer::identity())
      : TargetReduce(Policy::undefined, init_val, identity_)
  {}

  TargetReduce(Policy p, T init_val, T identity_ = Reducer::identity())
      : info(),
        val(identity_, identity_, info, checked_uses_offload(p)),
        hostData(new sycl::Shared_Host_Data<T> {init_val, identity_, this})
  {}

  TargetReduce(const TargetReduce&) = default;

  void reset(T init_val_, T identity_ = Reducer::identity())
  {
    hostData->hostVal  = init_val_;
    hostData->identity = identity_;
    val.reset(identity_);
    if (val.uses_offload())
    {
      val.reset_device(identity_, info);
    }
  }

  void reset(Policy p, T init_val_, T identity_ = Reducer::identity())
  {
    policy_supported_or_throw("SyclReduce::reset",
                              reduction_supported_policies_t<Policy::sycl> {},
                              p);
    const bool use_offload = sycl::uses_runtime_offload(p);
    if (val.uses_offload() && use_offload)
    {
      val.reset(identity_);
      val.reset_device(identity_, info);
    }
    else if (val.uses_offload() && !use_offload)
    {
      val.cleanup(info);
      val = sycl::Reduce_Data<T>(identity_, identity_, info, false);
    }
    else if (!val.uses_offload() && use_offload)
    {
      val = sycl::Reduce_Data<T>(identity_, identity_, info, true);
    }
    else
    {
      val.reset(identity_);
    }
    hostData->hostVal  = init_val_;
    hostData->identity = identity_;
  }

  //! apply reduction on device upon destruction
  ~TargetReduce()
  {
#ifndef __SYCL_DEVICE_ONLY__
    if (is_root())
    {
      val.cleanup(info);
      delete hostData;
      hostData = nullptr;
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    T result = hostData->hostVal;
    if (val.uses_offload())
    {
      val.deviceToHost(info);
      for (int i = 0; i < sycl::MaxNumTeams; ++i)
      {
        Reducer {}(result, val.host[i]);
      }
    }
    return result;
  }

  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction
  const TargetReduce& reduce(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i   = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    auto atm = ::sycl::atomic_ref<T, ::sycl::memory_order_acq_rel,
                                  ::sycl::memory_scope::device,
                                  ::sycl::access::address_space::global_space>(
        val.device[i]);
    Reducer {}(atm, rhsVal);
#else
    Reducer {}(hostData->hostVal, rhsVal);
#endif
    return *this;
  }

private:
  static bool checked_uses_offload(Policy p)
  {
    policy_supported_or_throw(
        "SyclReduce", reduction_supported_policies_t<Policy::sycl> {}, p);
    return sycl::uses_runtime_offload(p);
  }

  RAJA_INLINE bool is_root() const
  {
    return hostData && hostData->rootToken == this;
  }

  //! storage for offload information (host ID, device ID)
  sycl::Offload_Info info;

public:
  //! storage for reduction data (host ptr, device ptr, value)
  sycl::Reduce_Data<T> val;

private:
  //! shared host semantic state owned by the original reducer
  sycl::Shared_Host_Data<T>* hostData;
};

//! SYCL Target Reduction Location entity -- generalize on # of teams,
//! reduction, and type
// This is the last layer of the user facing Reduction API
template<typename Reducer, typename T, typename IndexType>
struct TargetReduceLoc
{
  TargetReduceLoc()
      : TargetReduceLoc(Policy::undefined,
                        Reducer::identity(),
                        RAJA::reduce::detail::DefaultLoc<IndexType>().value(),
                        Reducer::identity(),
                        RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {}

  explicit TargetReduceLoc(Policy p)
      : TargetReduceLoc(p,
                        Reducer::identity(),
                        RAJA::reduce::detail::DefaultLoc<IndexType>().value(),
                        Reducer::identity(),
                        RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {}

  explicit TargetReduceLoc(
      T init_val,
      IndexType init_loc,
      T identity_val_ = Reducer::identity(),
      IndexType identity_loc_ =
          RAJA::reduce::detail::DefaultLoc<IndexType>().value())
      : TargetReduceLoc(Policy::undefined,
                        init_val,
                        init_loc,
                        identity_val_,
                        identity_loc_)
  {}

  explicit TargetReduceLoc(
      Policy p,
      T init_val,
      IndexType init_loc,
      T identity_val_ = Reducer::identity(),
      IndexType identity_loc_ =
          RAJA::reduce::detail::DefaultLoc<IndexType>().value())
      : info(),
        val(identity_val_, identity_val_, info, checked_uses_offload(p)),
        loc(identity_loc_, identity_loc_, info, checked_uses_offload(p)),
        hostData(new sycl::Shared_Host_Loc_Data<T, IndexType> {
            init_val, init_loc, identity_val_, identity_loc_, this})
  {}

  TargetReduceLoc(const TargetReduceLoc&) = default;

  void reset(T init_val_,
             IndexType init_loc_,
             T identity_val_ = Reducer::identity(),
             IndexType identity_loc_ =
                 RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {
    hostData->hostVal     = init_val_;
    hostData->hostLoc     = init_loc_;
    hostData->identity    = identity_val_;
    hostData->identityLoc = identity_loc_;
    val.reset(identity_val_);
    loc.reset(identity_loc_);
    if (val.uses_offload())
    {
      val.reset_device(identity_val_, info);
      loc.reset_device(identity_loc_, info);
    }
  }

  void reset(Policy p,
             T init_val_,
             IndexType init_loc_,
             T identity_val_ = Reducer::identity(),
             IndexType identity_loc_ =
                 RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {
    policy_supported_or_throw("SyclReduceLoc::reset",
                              reduction_supported_policies_t<Policy::sycl> {},
                              p);
    const bool use_offload = sycl::uses_runtime_offload(p);
    if (val.uses_offload() && use_offload)
    {
      val.reset(identity_val_);
      loc.reset(identity_loc_);
      val.reset_device(identity_val_, info);
      loc.reset_device(identity_loc_, info);
    }
    else if (val.uses_offload() && !use_offload)
    {
      val.cleanup(info);
      loc.cleanup(info);
      val = sycl::Reduce_Data<T>(identity_val_, identity_val_, info, false);
      loc = sycl::Reduce_Data<IndexType>(identity_loc_, identity_loc_, info,
                                         false);
    }
    else if (!val.uses_offload() && use_offload)
    {
      val = sycl::Reduce_Data<T>(identity_val_, identity_val_, info, true);
      loc = sycl::Reduce_Data<IndexType>(identity_loc_, identity_loc_, info,
                                         true);
    }
    else
    {
      val.reset(identity_val_);
      loc.reset(identity_loc_);
    }
    hostData->hostVal     = init_val_;
    hostData->hostLoc     = init_loc_;
    hostData->identity    = identity_val_;
    hostData->identityLoc = identity_loc_;
  }

  //! apply reduction on device upon destruction
  ~TargetReduceLoc()
  {
#ifndef __SYCL_DEVICE_ONLY__
    if (is_root())
    {
      val.cleanup(info);
      loc.cleanup(info);
      delete hostData;
      hostData = nullptr;
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    T result;
    IndexType resultLoc;
    compute_result(result, resultLoc);
    return result;
  }

  //! alias for operator T()
  T get() { return operator T(); }

  //! map result value back to host if not done already; return aggregate
  //! location
  IndexType getLoc()
  {
    T result;
    IndexType resultLoc;
    compute_result(result, resultLoc);
    return resultLoc;
  }

  //! apply reduction
  const TargetReduceLoc& reduce(T rhsVal, IndexType rhsLoc) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    ::sycl::atomic_fence(::sycl::memory_order_acquire,
                         ::sycl::memory_scope::device);
    Reducer {}(val.device[i], loc.device[i], rhsVal, rhsLoc);
    ::sycl::atomic_fence(::sycl::memory_order_release,
                         ::sycl::memory_scope::device);
#else
    Reducer {}(hostData->hostVal, hostData->hostLoc, rhsVal, rhsLoc);
#endif
    return *this;
  }

private:
  static bool checked_uses_offload(Policy p)
  {
    policy_supported_or_throw(
        "SyclReduceLoc", reduction_supported_policies_t<Policy::sycl> {}, p);
    return sycl::uses_runtime_offload(p);
  }

  RAJA_INLINE bool is_root() const
  {
    return hostData && hostData->rootToken == this;
  }

  //! storage for offload information
  sycl::Offload_Info info;

public:
  //! storage for reduction data for value
  sycl::Reduce_Data<T> val;
  sycl::Reduce_Data<IndexType> loc;

private:
  void compute_result(T& result, IndexType& resultLoc)
  {
    result    = hostData->hostVal;
    resultLoc = hostData->hostLoc;
    if (val.uses_offload())
    {
      val.deviceToHost(info);
      loc.deviceToHost(info);

      for (int i = 0; i < sycl::MaxNumTeams; ++i)
      {
        Reducer {}(result, resultLoc, val.host[i], loc.host[i]);
      }
    }
  }

  //! shared host semantic state owned by the original reducer
  sycl::Shared_Host_Loc_Data<T, IndexType>* hostData;
};

//! specialization of ReduceSum for omp_target_reduce
// This is the first layer of the user facing Reduction API
template<typename T>
class ReduceSum<sycl_reduce, T> : public TargetReduce<RAJA::reduce::sum<T>, T>
{
public:
  using self   = ReduceSum<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::sum<T>, T>;
  using parent::parent;

  //! enable operator+= for ReduceSum -- alias for reduce()
  const self& operator+=(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i   = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    auto atm = ::sycl::atomic_ref<T, ::sycl::memory_order_acq_rel,
                                  ::sycl::memory_scope::device,
                                  ::sycl::access::address_space::global_space>(
        parent::val.device[i]);
    atm.fetch_add(rhsVal);
#else
    parent::reduce(rhsVal);
#endif
    return *this;
  }
};

//! specialization of ReduceBitOr for sycl_reduce
// This is the first layer of the user facing Reduction API
template<typename T>
class ReduceBitOr<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::or_bit<T>, T>
{
public:
  using self   = ReduceBitOr<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::or_bit<T>, T>;
  using parent::parent;

  //! enable operator|= for ReduceBitOr -- alias for reduce()
  const self& operator|=(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i   = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    auto atm = ::sycl::atomic_ref<T, ::sycl::memory_order_acq_rel,
                                  ::sycl::memory_scope::device,
                                  ::sycl::access::address_space::global_space>(
        parent::val.device[i]);
    atm |= rhsVal;
#else
    parent::reduce(rhsVal);
#endif
    return *this;
  }
};

//! specialization of ReduceBitAnd for sycl_reduce
// This is the first layer of the user facing Reduction API
template<typename T>
class ReduceBitAnd<sycl_reduce, T>
    : public TargetReduce<RAJA::reduce::and_bit<T>, T>
{
public:
  using self   = ReduceBitAnd<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::and_bit<T>, T>;
  using parent::parent;

  //! enable operator&= for ReduceBitAnd -- alias for reduce()
  const self& operator&=(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i   = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    auto atm = ::sycl::atomic_ref<T, ::sycl::memory_order_acq_rel,
                                  ::sycl::memory_scope::device,
                                  ::sycl::access::address_space::global_space>(
        parent::val.device[i]);
    atm &= rhsVal;
#else
    parent::reduce(rhsVal);
#endif
    return *this;
  }
};

//! specialization of ReduceMin for omp_target_reduce
// This is the first layer of the user facing Reduction API
template<typename T>
class ReduceMin<sycl_reduce, T> : public TargetReduce<RAJA::reduce::min<T>, T>
{
public:
  using self   = ReduceMin<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::min<T>, T>;
  using parent::parent;

  //! enable min() for ReduceMin -- alias for reduce()
  const self& min(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i   = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    auto atm = ::sycl::atomic_ref<T, ::sycl::memory_order_acq_rel,
                                  ::sycl::memory_scope::device,
                                  ::sycl::access::address_space::global_space>(
        parent::val.device[i]);
    atm.fetch_min(rhsVal);
#else
    parent::reduce(rhsVal);
#endif
    return *this;
  }
};

//! specialization of ReduceMax for omp_target_reduce
// This is the first layer of the user facing Reduction API
template<typename T>
class ReduceMax<sycl_reduce, T> : public TargetReduce<RAJA::reduce::max<T>, T>
{
public:
  using self   = ReduceMax<sycl_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::max<T>, T>;
  using parent::parent;

  //! enable max() for ReduceMax -- alias for reduce()
  const self& max(T rhsVal) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    auto i   = 0;  //__spirv::initLocalInvocationId<1, ::sycl::id<1>>()[0];
    auto atm = ::sycl::atomic_ref<T, ::sycl::memory_order_acq_rel,
                                  ::sycl::memory_scope::device,
                                  ::sycl::access::address_space::global_space>(
        parent::val.device[i]);
    atm.fetch_max(rhsVal);
#else
    parent::reduce(rhsVal);
#endif
    return *this;
  }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_SYCL guard

#endif  // closing endif for header file include guard
