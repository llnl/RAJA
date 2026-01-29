//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_omp_target_reduce_HPP
#define RAJA_omp_target_reduce_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)

//#include <cassert>  // Leaving out until XL is fixed 2/25/2019.

#include <algorithm>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/openmp/policy.hpp"

namespace RAJA
{

namespace omp
{
#pragma omp declare target

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

#pragma omp end declare target

// Alias for clarity. Reduction size operates on number of omp teams.
// Ideally, MaxNumTeams = ThreadsPerTeam in omp_target_parallel_for_exec.
static constexpr int MaxNumTeams = policy::omp::MAXNUMTHREADS;

//! Information necessary for OpenMP offload to be considered
struct Offload_Info
{
  int hostID {omp_get_initial_device()};
  int deviceID {omp_get_default_device()};
  bool isMapped {false};

  Offload_Info() = default;

  Offload_Info(const Offload_Info& other)
      : hostID {other.hostID},
        deviceID {other.deviceID},
        isMapped {other.isMapped}
  {}
};

//! Reduction data for OpenMP Offload -- stores value, host pointer, and device
//! pointer
template<typename T>
struct Reduce_Data
{
  mutable T value;
  T* device;
  T* host;

  //! disallow default constructor
  Reduce_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  Reduce_Data(T initValue, T identityValue, Offload_Info& info)
      : value(initValue),
        device {reinterpret_cast<T*>(
            omp_target_alloc(omp::MaxNumTeams * sizeof(T), info.deviceID))},
        host {new T[omp::MaxNumTeams]}
  {
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
    std::fill_n(host, omp::MaxNumTeams, identityValue);
    hostToDevice(info);
  }

  void reset(T initValue) { value = initValue; }

  //! default copy constructor for POD
  Reduce_Data(const Reduce_Data&) = default;

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE void hostToDevice(Offload_Info& info)
  {
    // precondition: host and device are valid pointers
    if (omp_target_memcpy(reinterpret_cast<void*>(device),
                          reinterpret_cast<void*>(host),
                          omp::MaxNumTeams * sizeof(T), 0, 0, info.deviceID,
                          info.hostID) != 0)
    {
      printf("Unable to copy memory from host to device\n");
      exit(1);
    }
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE void deviceToHost(Offload_Info& info)
  {
    // precondition: host and device are valid pointers
    if (omp_target_memcpy(reinterpret_cast<void*>(host),
                          reinterpret_cast<void*>(device),
                          omp::MaxNumTeams * sizeof(T), 0, 0, info.hostID,
                          info.deviceID) != 0)
    {
      printf("Unable to copy memory from device to host\n");
      exit(1);
    }
  }

  //! frees all data from the offload information passed
  RAJA_INLINE void cleanup(Offload_Info& info)
  {
    if (device)
    {
      omp_target_free(reinterpret_cast<void*>(device), info.deviceID);
      device = nullptr;
    }
    if (host)
    {
      delete[] host;
      host = nullptr;
    }
  }
};

}  // end namespace omp

//! OpenMP Target Reduction entity -- generalize on # of teams, reduction, and
//! type
template<typename Reducer, typename T>
struct TargetReduce
{
  TargetReduce()                    = delete;
  TargetReduce(const TargetReduce&) = default;

  explicit TargetReduce(T init_val_, T identity_ = Reducer::identity())
      : info(),
        val(identity_, identity_, info),
        initVal(init_val_),
        finalVal(identity_)
  {}

  void reset(T init_val_, T identity_ = Reducer::identity())
  {
    operator T();
    val.reset(identity_);
    initVal  = init_val_;
    finalVal = identity_;
  }

#ifdef __ibmxl__  // TODO: implicit declare target doesn't pick this up
#pragma omp declare target
#endif
  //! apply reduction on device upon destruction
  ~TargetReduce()
  {
    // assert ( omp_get_num_teams() <= omp::MaxNumTeams );  // Leaving out until
    // XL is fixed 2/25/2019.
    if (!omp_is_initial_device())
    {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer {}(val.device[tid], val.value);
      }
    }
  }
#ifdef __ibmxl__  // TODO: implicit declare target doesn't pick this up
#pragma omp end declare target
#endif

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    if (!info.isMapped)
    {
      val.deviceToHost(info);

      for (int i = 0; i < omp::MaxNumTeams; ++i)
      {
        Reducer {}(val.value, val.host[i]);
      }
      val.cleanup(info);
      info.isMapped = true;
    }
    finalVal = Reducer::identity();
    Reducer {}(finalVal, initVal);
    Reducer {}(finalVal, val.value);
    return finalVal;
  }

  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction
  TargetReduce& reduce(T rhsVal)
  {
    Reducer {}(val.value, rhsVal);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  const TargetReduce& reduce(T rhsVal) const
  {
    Reducer {}(val.value, rhsVal);
    return *this;
  }

private:
  //! storage for offload information (host ID, device ID)
  omp::Offload_Info info;
  //! storage for reduction data (host ptr, device ptr, value)
  omp::Reduce_Data<T> val;
  T initVal;
  T finalVal;
};

//! OpenMP Target Reduction Location entity -- generalize on # of teams,
//! reduction, and type
template<typename Reducer, typename T, typename IndexType>
struct TargetReduceLoc
{
  TargetReduceLoc()                       = delete;
  TargetReduceLoc(const TargetReduceLoc&) = default;

  explicit TargetReduceLoc(
      T init_val_,
      IndexType init_loc,
      T identity_val_ = Reducer::identity(),
      IndexType identity_loc_ =
          RAJA::reduce::detail::DefaultLoc<IndexType>().value())
      : info(),
        val(identity_val_, identity_val_, info),
        loc(identity_loc_, identity_loc_, info),
        initVal(init_val_),
        finalVal(identity_val_),
        initLoc(init_loc),
        finalLoc(identity_loc_)
  {}

  void reset(T init_val_,
             IndexType init_loc_,
             T identity_val_ = Reducer::identity(),
             IndexType identity_loc_ =
                 RAJA::reduce::detail::DefaultLoc<IndexType>().value())
  {
    operator T();
    val.reset(identity_val_);
    loc.reset(identity_loc_);
    initVal  = init_val_;
    finalVal = identity_val_;
    initLoc  = init_loc_;
    finalLoc = identity_loc_;
  }

  //! apply reduction on device upon destruction
  ~TargetReduceLoc()
  {
    // assert ( omp_get_num_teams() <= omp::MaxNumTeams );  // Leaving out until
    // XL is fixed 2/25/2019.
    if (!omp_is_initial_device())
    {
#pragma omp critical
      {
        int tid = omp_get_team_num();
        Reducer {}(val.device[tid], loc.device[tid], val.value, loc.value);
      }
    }
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    if (!info.isMapped)
    {
      val.deviceToHost(info);
      loc.deviceToHost(info);
      for (int i = 0; i < omp::MaxNumTeams; ++i)
      {
        Reducer {}(val.value, loc.value, val.host[i], loc.host[i]);
      }
      val.cleanup(info);
      loc.cleanup(info);
      info.isMapped = true;
    }
    finalVal = Reducer::identity();
    finalLoc = IndexType(RAJA::reduce::detail::DefaultLoc<IndexType>().value());
    Reducer {}(finalVal, finalLoc, initVal, initLoc);
    Reducer {}(finalVal, finalLoc, val.value, loc.value);
    return finalVal;
  }

  //! alias for operator T()
  T get() { return operator T(); }

  //! map result value back to host if not done already; return aggregate
  //! location
  IndexType getLoc()
  {
    if (!info.isMapped) get();
    // return loc.value;
    return (finalLoc);
  }

  //! apply reduction
  TargetReduceLoc& reduce(T rhsVal, IndexType rhsLoc)
  {
    Reducer {}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  const TargetReduceLoc& reduce(T rhsVal, IndexType rhsLoc) const
  {
    Reducer {}(val.value, loc.value, rhsVal, rhsLoc);
    return *this;
  }

private:
  //! storage for offload information
  omp::Offload_Info info;
  //! storage for reduction data for value
  omp::Reduce_Data<T> val;
  //! storage for redcution data for location
  omp::Reduce_Data<IndexType> loc;
  T initVal;
  T finalVal;
  IndexType initLoc;
  IndexType finalLoc;
};

//! specialization of ReduceSum for omp_target_reduce
template<typename T>
class ReduceSum<omp_target_reduce, T>
    : public TargetReduce<RAJA::reduce::sum<T>, T>
{
public:
  using self   = ReduceSum<omp_target_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::sum<T>, T>;
  using parent::parent;

  //! enable operator+= for ReduceSum -- alias for reduce()
  self& operator+=(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }

  //! enable operator+= for ReduceSum -- alias for reduce()
  const self& operator+=(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceBitOr for omp_target_reduce
template<typename T>
class ReduceBitOr<omp_target_reduce, T>
    : public TargetReduce<RAJA::reduce::or_bit<T>, T>
{
public:
  using self   = ReduceBitOr<omp_target_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::or_bit<T>, T>;
  using parent::parent;

  //! enable operator|= for ReduceBitOr -- alias for reduce()
  self& operator|=(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }

  //! enable operator|= for ReduceBitOr -- alias for reduce()
  const self& operator|=(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceBitAnd for omp_target_reduce
template<typename T>
class ReduceBitAnd<omp_target_reduce, T>
    : public TargetReduce<RAJA::reduce::and_bit<T>, T>
{
public:
  using self   = ReduceBitAnd<omp_target_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::and_bit<T>, T>;
  using parent::parent;

  //! enable operator&= for ReduceBitAnd -- alias for reduce()
  self& operator&=(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }

  //! enable operator&= for ReduceBitAnd -- alias for reduce()
  const self& operator&=(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMin for omp_target_reduce
template<typename T>
class ReduceMin<omp_target_reduce, T>
    : public TargetReduce<RAJA::reduce::min<T>, T>
{
public:
  using self   = ReduceMin<omp_target_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::min<T>, T>;
  using parent::parent;

  //! enable min() for ReduceMin -- alias for reduce()
  self& min(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }

  //! enable min() for ReduceMin -- alias for reduce()
  const self& min(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMax for omp_target_reduce
template<typename T>
class ReduceMax<omp_target_reduce, T>
    : public TargetReduce<RAJA::reduce::max<T>, T>
{
public:
  using self   = ReduceMax<omp_target_reduce, T>;
  using parent = TargetReduce<RAJA::reduce::max<T>, T>;
  using parent::parent;

  //! enable max() for ReduceMax -- alias for reduce()
  self& max(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }

  //! enable max() for ReduceMax -- alias for reduce()
  const self& max(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMinLoc for omp_target_reduce
template<typename T, typename IndexType>
class ReduceMinLoc<omp_target_reduce, T, IndexType>
    : public TargetReduceLoc<omp::minloc<T, IndexType>, T, IndexType>
{
public:
  using self   = ReduceMinLoc<omp_target_reduce, T, IndexType>;
  using parent = TargetReduceLoc<omp::minloc<T, IndexType>, T, IndexType>;
  using parent::parent;

  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  self& minloc(T rhsVal, IndexType rhsLoc)
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }

  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  const self& minloc(T rhsVal, IndexType rhsLoc) const
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
};

//! specialization of ReduceMaxLoc for omp_target_reduce
template<typename T, typename IndexType>
class ReduceMaxLoc<omp_target_reduce, T, IndexType>
    : public TargetReduceLoc<omp::maxloc<T, IndexType>, T, IndexType>
{
public:
  using self   = ReduceMaxLoc<omp_target_reduce, T, IndexType>;
  using parent = TargetReduceLoc<omp::maxloc<T, IndexType>, T, IndexType>;
  using parent::parent;

  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  self& maxloc(T rhsVal, IndexType rhsLoc)
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }

  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  const self& maxloc(T rhsVal, IndexType rhsLoc) const
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
};


}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_TARGET_OPENMP guard

#endif  // closing endif for header file include guard
