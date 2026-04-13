#ifndef RAJA_launch_concepts_HPP
#define RAJA_launch_concepts_HPP

#include <cstddef>

#include "RAJA/config.hpp"

#include "RAJA/pattern/forall-concepts.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

template<typename LaunchContextPolicy>
class LaunchContextT;

template<typename T>
concept HostPolicyList = requires {
  typename camp::decay<T>::host_policy_t;
  typename resources::get_resource<typename camp::decay<T>::host_policy_t>::type;
};

#if defined(RAJA_GPU_ACTIVE)
template<typename T>
concept HostDevicePolicyList = HostPolicyList<T> && requires {
  typename camp::decay<T>::device_policy_t;
  typename resources::get_resource<typename camp::decay<T>::device_policy_t>::type;
};
#else
template<typename T>
concept HostDevicePolicyList = HostPolicyList<T>;
#endif

template<typename T>
concept LaunchPolicyList = HostDevicePolicyList<T>;

template<typename T>
concept LoopPolicyList = HostDevicePolicyList<T>;

template<typename T>
concept LaunchContextConcept = requires(camp::decay<T> ctx,
                                        std::size_t bytes,
                                        void* mem) {
  ctx.shared_mem_offset;
  ctx.shared_mem_ptr;
  ctx.shared_mem_offset = bytes;
  ctx.shared_mem_ptr    = mem;
  ctx.template getSharedMemory<int>(bytes);
  ctx.releaseSharedMemory();
  ctx.teamSync();
};

template<typename T>
concept LaunchContextPolicyConcept =
    LaunchContextConcept<LaunchContextT<camp::decay<T>>>;

template<typename Mapper, typename IndicesAndDims, typename IdxT = std::ptrdiff_t>
concept LaunchIndexMapperFor = requires(IndicesAndDims const& idxNDims) {
  camp::decay<Mapper>::template index<IdxT>(idxNDims);
  camp::decay<Mapper>::template size<IdxT>(idxNDims);
};

#if defined(RAJA_CUDA_ACTIVE)
template<typename Mapper, typename IdxT = std::ptrdiff_t>
concept CudaLaunchIndexMapper =
    LaunchIndexMapperFor<Mapper, cuda::NonCachedIndicesAndDims, IdxT>;
#endif

#if defined(RAJA_HIP_ACTIVE)
template<typename Mapper, typename IdxT = std::ptrdiff_t>
concept HipLaunchIndexMapper =
    LaunchIndexMapperFor<Mapper, hip::NonCachedIndicesAndDims, IdxT>;
#endif

}  // namespace RAJA

#endif
