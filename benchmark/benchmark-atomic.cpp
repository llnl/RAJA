//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// This file is intended to provide an interface for comparing the performance
// of RAJA's atomic implementations with Desul's atomic implementations.  In order
// to accomplish this without requiring two separate build system configurations
// this file directly includes "desul/atomics.hpp" and directly calls desul namespace
// atomics.  This is different from how a typical RAJA user would call a desul atomic.

#include "RAJA/RAJA.hpp"
#include "RAJA/util/for_each.hpp"
#include "RAJA/util/Timer.hpp"
#if defined (RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp/atomic.hpp"
#endif
#include "desul/atomics.hpp"

#include <type_traits>
#include <iostream>
#include <sstream>
#include <string>


/// Conditional compilation for CUDA benchmarks.
#if defined (RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/atomic.hpp"

template<int BLOCK_SZ>
struct ExecPolicyGPU {
    using policy = RAJA::cuda_exec<BLOCK_SZ, false>;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "CUDA execution with block size " << BLOCK_SZ;
        return ss.str();
    }
};

struct GPUAtomic {
    using policy = RAJA::policy::cuda::cuda_atomic;
};

#elif defined (RAJA_ENABLE_HIP)
#include "RAJA/policy/hip/atomic.hpp"

template<int BLOCK_SZ>
struct ExecPolicyGPU {
    using policy = RAJA::hip_exec<BLOCK_SZ, false>;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "HIP execution with block size " << BLOCK_SZ;
        return ss.str();
    }
};

struct GPUAtomic {
    using policy = RAJA::policy::hip::hip_atomic;
};

#endif

#define BLOCK_SZ 256
#define INDENT "  "
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

// Desul atomics have a different signature than RAJA's built in ops. The following code provides some
// helper function templates so that they can be called using the same signature in timing code.

// Struct holding Desul atomic signature typedef
template<typename ReturnType, typename... Args>
struct DesulAtomicSignature {
    using type = ReturnType(*)(Args..., raja_default_desul_order, raja_default_desul_scope);
};

// Struct holding RAJA atomic signature typedef
template<typename AtomicType>
struct RajaAtomicSignature {
    using type = AtomicType(*)(AtomicType*, const AtomicType);
};

/// RAJA::atomicAdd and other RAJA namespace atomic calls are overloaded and have an ambiguous type
/// so they can't be passed as a template parameter.
/// The following macro disambiguates the call and provide a signature comaptible with the DESUL
/// wrapper. AtomicOperation must be a valid RAJA namespace atomic operation, like atomicAdd,
/// atomicMax, etc.
#define OPERATOR_CALL_BINARY(AtomicOperation)                                               \
    template<typename ArgType>                                                              \
    RAJA_HOST_DEVICE ArgType operator()(ArgType* acc, const ArgType val) const {            \
        return RAJA::AtomicOperation(Policy {}, acc, val);                                  \
    }                                                                                       \

#define OPERATOR_CALL_UNARY(AtomicOperation)                                                \
    template<typename ArgType>                                                              \
    RAJA_HOST_DEVICE ArgType operator()(ArgType* acc, const ArgType) const {                \
        return RAJA::AtomicOperation(Policy {}, acc);                                       \
    }                                                                                       \

#define DECLARE_ATOMIC_WRAPPER(AtomicFunctorName, AtomicOperatorDeclaration)                \
template<typename Policy>                                                                   \
struct AtomicFunctorName {                                                                  \
    const char* name = #AtomicFunctorName ;                                                 \
    AtomicOperatorDeclaration                                                               \
};                                                                                          \

DECLARE_ATOMIC_WRAPPER(AtomicAdd, OPERATOR_CALL_BINARY(atomicAdd))
DECLARE_ATOMIC_WRAPPER(AtomicSub, OPERATOR_CALL_BINARY(atomicSub))
DECLARE_ATOMIC_WRAPPER(AtomicMax, OPERATOR_CALL_BINARY(atomicMax))
DECLARE_ATOMIC_WRAPPER(AtomicMin, OPERATOR_CALL_BINARY(atomicMin))
DECLARE_ATOMIC_WRAPPER(AtomicIncBinary, OPERATOR_CALL_BINARY(atomicInc))
DECLARE_ATOMIC_WRAPPER(AtomicDecBinary, OPERATOR_CALL_BINARY(atomicDec))
DECLARE_ATOMIC_WRAPPER(AtomicIncUnary, OPERATOR_CALL_UNARY(atomicInc))
DECLARE_ATOMIC_WRAPPER(AtomicDecUnary, OPERATOR_CALL_UNARY(atomicDec))
DECLARE_ATOMIC_WRAPPER(AtomicAnd, OPERATOR_CALL_BINARY(atomicAnd))
DECLARE_ATOMIC_WRAPPER(AtomicOr, OPERATOR_CALL_BINARY(atomicOr))
DECLARE_ATOMIC_WRAPPER(AtomicXor, OPERATOR_CALL_BINARY(atomicXor))
DECLARE_ATOMIC_WRAPPER(AtomicExchange, OPERATOR_CALL_BINARY(atomicExchange))
DECLARE_ATOMIC_WRAPPER(AtomicLoad, OPERATOR_CALL_UNARY(atomicLoad))

/// Instead of complicating the above macro to handle these two atomics, do the declarations
/// manually below.
template<typename Policy>
struct AtomicStore {
    const char* name = "AtomicStore";
    template<typename ArgType>
    RAJA_HOST_DEVICE void operator()(ArgType* acc, const ArgType val) const {
        return RAJA::atomicStore(Policy {}, acc, val);
    }
};

template<typename Policy>
struct AtomicCAS {
    const char* name = "AtomicCAS";
    template<typename ArgType>
    RAJA_HOST_DEVICE ArgType operator()(ArgType* acc, ArgType compare) const {
        return RAJA::atomicCAS(Policy {}, acc, compare, ArgType(1));
    }
};

/// ExecPolicy wrapper for OpenMP
struct ExecPolicyOMP {
    using policy = RAJA::omp_parallel_for_exec;;
    static std::string PolicyName() {
        std::stringstream ss;
        ss << "OpenMP execution";
        return ss.str();
    }
};

/// Functor wrapping the desul implementation.  Wrapping the desul call ensure an identical signature with
/// RAJA's implementations.  Wrapping the call in an functor allows simple type deduction for printing
/// from within the benchmark.
template<typename T, typename ReturnType, typename DesulAtomicSignature<ReturnType, T*, const T, T>::type atomic_impl>
struct atomicWrapperDesulTernary {
    /// Call operator overload template that allows invoking DESUL atomic with a (int*)(T*, T) signature
    RAJA_HOST_DEVICE ReturnType operator()(T * acc, T value) const {
        return atomic_impl(acc, value, T(1), raja_default_desul_order{},
                        raja_default_desul_scope{});
    }
};

template<typename T, typename ReturnType, typename DesulAtomicSignature<ReturnType, T*, const T>::type atomic_impl>
struct atomicWrapperDesulBinary {
    /// Call operator overload template that allows invoking DESUL atomic with a (int*)(T*, T) signature
    RAJA_HOST_DEVICE ReturnType operator()(T * acc, T value) const {
        return atomic_impl(acc, value, raja_default_desul_order{},
                        raja_default_desul_scope{});
    }
};

/// Unary wrapper variant for increment and decrement benchmarks.
template<typename T, typename ReturnType, typename DesulAtomicSignature<ReturnType, T*>::type atomic_impl>
struct atomicWrapperDesulUnary {
    RAJA_HOST_DEVICE ReturnType operator()(T* acc, T) const {
        return atomic_impl(acc, raja_default_desul_order{},
                        raja_default_desul_scope{});
    }
};

template<typename T>
class IsDesul : public std::false_type {};

template<typename T, typename ReturnType, typename DesulAtomicSignature<ReturnType, T*, T, T>::type atomic_impl>
class IsDesul<atomicWrapperDesulTernary<T, ReturnType, atomic_impl>> : public std::true_type {};

template<typename T, typename ReturnType, typename DesulAtomicSignature<ReturnType, T*, T>::type atomic_impl>
class IsDesul<atomicWrapperDesulBinary<T, ReturnType, atomic_impl>> : public std::true_type {};

template<typename T, typename ReturnType, typename DesulAtomicSignature<ReturnType, T*>::type atomic_impl>
class IsDesul<atomicWrapperDesulUnary<T, ReturnType, atomic_impl>> : public std::true_type {};

template<typename AtomicImplType>
std::string GetImplName (const AtomicImplType& impl) {
    if (IsDesul<AtomicImplType>::value) {
        return "Desul atomic";
    } else {
        return "RAJA atomic";
    }
}

template <class ExecPolicy, typename AtomicType, bool test_array, typename AtomicImplType>
void TimeAtomicOp(const AtomicImplType& atomic_impl, uint64_t N, uint64_t num_iterations = 4, int array_size = 100, bool print_to_output = true) {
    RAJA::Timer timer;

    // Allocate memory
    AtomicType* device_value = nullptr;
    int len_array = test_array ? array_size : 1;
    camp::resources::Resource resource {RAJA::resources::get_resource<typename ExecPolicy::policy>::type::get_default()};
    device_value = resource.allocate<AtomicType>(len_array);

    timer.start();
    if (test_array) {
        for (uint64_t i = 0; i < num_iterations; ++i) {
            RAJA::forall<typename ExecPolicy::policy>(RAJA::TypedRangeSegment<uint64_t>(0, N),
            [=] RAJA_HOST_DEVICE(uint64_t tid)  {
                atomic_impl(&(device_value[tid % array_size]), AtomicType(1));
            });
        }
    } else {
        for (uint64_t i = 0; i < num_iterations; ++i) {
            RAJA::forall<typename ExecPolicy::policy>(RAJA::TypedRangeSegment<uint64_t>(0, N),
            [=] RAJA_HOST_DEVICE(uint64_t tid)  {
                atomic_impl(device_value, AtomicType(1));
            });
        }
    }

    resource.wait();
    timer.stop();
    resource.deallocate(device_value);

    double t = timer.elapsed();
    if (print_to_output) {
        std::cout << INDENT << INDENT << t << "s" << INDENT;
        std::cout << GetImplName(atomic_impl) << ", ";
        if (test_array) {
            std::cout << "Number of atomics under contention " << array_size << ", ";
        }
        std::cout << num_iterations * N << " many atomic operations" << ", ";
        std::cout << ExecPolicy::PolicyName();
        std::cout << std::endl;
    }
}

template<typename ...T>
struct list_concat;

template<typename ...T, typename...S>
struct list_concat<camp::list<T...>, camp::list<S...>> {
    using type = camp::list<T..., S...>;
};

/// Holder for atomic operations that work with arbitrary atomic type, e.g. double, float
/// and ints etc.
template<typename AtomicDataType, typename Policy>
struct universal_atomic_ops {
    using type = camp::list<std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_add>, AtomicAdd<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_sub>, AtomicSub<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_min>, AtomicMin<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_max>, AtomicMax<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_inc_mod>, AtomicIncBinary<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_dec_mod>, AtomicDecBinary<Policy>>,
                                                std::pair<atomicWrapperDesulUnary<AtomicDataType, AtomicDataType, desul::atomic_fetch_inc>, AtomicIncUnary<Policy>>,
                                                std::pair<atomicWrapperDesulUnary<AtomicDataType, AtomicDataType, desul::atomic_fetch_dec>, AtomicDecUnary<Policy>>,
                                                std::pair<atomicWrapperDesulUnary<const AtomicDataType, AtomicDataType, desul::atomic_load>, AtomicLoad<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, void, desul::atomic_store>, AtomicStore<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_exchange>, AtomicExchange<Policy>>,
                                                std::pair<atomicWrapperDesulTernary<AtomicDataType, AtomicDataType, desul::atomic_compare_exchange>, AtomicCAS<Policy>>>;
};

template<typename AtomicDataType, typename Policy>
struct integral_atomic_ops {
    using type = camp::list<std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_and>, AtomicAnd<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_or>, AtomicOr<Policy>>,
                                                std::pair<atomicWrapperDesulBinary<AtomicDataType, AtomicDataType, desul::atomic_fetch_xor>, AtomicXor<Policy>>>;
};

template<typename AtomicDataType, typename Policy, class t = void>
struct atomic_ops;

/// Include all atomic ops if the underlying atomic to benchmark is integral.
template<typename AtomicDataType, typename Policy>
struct atomic_ops<AtomicDataType, Policy, typename std::enable_if<std::is_integral<AtomicDataType>::value>::type> {
    using type = typename list_concat<typename universal_atomic_ops<AtomicDataType, Policy>::type, typename integral_atomic_ops<AtomicDataType, Policy>::type>::type;
};

/// Omit bitwise ops and, or, and xor for floating point types
template<typename AtomicDataType, typename Policy>
struct atomic_ops<AtomicDataType, Policy, typename std::enable_if<std::is_floating_point<AtomicDataType>::value>::type> {
    using type = typename universal_atomic_ops< AtomicDataType, Policy >::type;
};


template<typename AtomicDataType, typename AtomicPolicy, typename ExecPolicy>
void ExecuteBenchmark(uint64_t N) {
    using ops = atomic_ops<AtomicDataType, AtomicPolicy>;
    using iter_t = typename ops::type;
    auto iter = iter_t{};
    RAJA::for_each_type(iter, [&](auto type_pair) {
        auto desul_functor = type_pair.first;
        auto raja_functor = type_pair.second;
        std::cout << INDENT << "Executing " << raja_functor.name << " integer benchmarks" << std::endl;
        TimeAtomicOp<ExecPolicy, AtomicDataType, true>(desul_functor, N, 100, 10000);
        TimeAtomicOp<ExecPolicy, AtomicDataType, true>(raja_functor, N, 100, 10000);
        TimeAtomicOp<ExecPolicy, AtomicDataType, true>(desul_functor, N, 10, 1000);
        TimeAtomicOp<ExecPolicy, AtomicDataType, true>(raja_functor, N, 10, 1000);
        TimeAtomicOp<ExecPolicy, AtomicDataType, true>(desul_functor, N, 4, 10);
        TimeAtomicOp<ExecPolicy, AtomicDataType, true>(raja_functor, N, 4, 10);
        // Test contention over a single atomic
        TimeAtomicOp<ExecPolicy, AtomicDataType, false>(desul_functor, N);
        TimeAtomicOp<ExecPolicy, AtomicDataType, false>(raja_functor, N);
    });
}

int main (int argc, char* argv[]) {
    if (argc > 2) {
        RAJA_ABORT_OR_THROW("Usage: ./benchmark-atomic.exe <N> where N is the optional size of the benchmark loop");
    }
    uint64_t N = 1000000000;
    if (argc == 2) {
        N = std::stoll(argv[1]);
    }

    #if defined (RAJA_ENABLE_CUDA) || defined (RAJA_ENABLE_HIP)
    // Perform an untimed initialization of both desul and RAJA atomics.
    TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(atomicWrapperDesulBinary<int, int, desul::atomic_fetch_add>{}, N, 10, 1000, false);
    TimeAtomicOp<ExecPolicyGPU<BLOCK_SZ>, int, true>(AtomicAdd<typename GPUAtomic::policy>{}, N, 10, 1000, false);
    // GPU benchmarks
    std::cout << "Executing GPU benchmarks" << std::endl;
    ExecuteBenchmark<double, typename GPUAtomic::policy, ExecPolicyGPU<BLOCK_SZ>>(N);
    #endif

    #if defined (RAJA_ENABLE_OPENMP)
    // Perform an untimed initialization of both desul and RAJA atomics.
    TimeAtomicOp<ExecPolicyOMP, int, true>(AtomicAdd<typename RAJA::policy::omp::omp_atomic>{}, N, 10, 1000, false);
    TimeAtomicOp<ExecPolicyOMP, int, true>(atomicWrapperDesulBinary<int, int, desul::atomic_fetch_add>{}, N, 10, 1000, false);

    // OpenMP benchmarks
    std::cout << "Executing OpenMP benchmarks" << std::endl;
    ExecuteBenchmark<double, typename RAJA::policy::omp::omp_atomic, ExecPolicyOMP>(N);
    #endif

    return 0;
}
