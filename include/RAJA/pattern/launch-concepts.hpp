#ifndef RAJA_launch_concepts_HPP
#define RAJA_launch_concepts_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/forall-concepts.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

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

}  // namespace RAJA

#endif
