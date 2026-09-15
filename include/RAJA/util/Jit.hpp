#include "RAJA/config.hpp"
#include "RAJA/pattern/launch/launch_context_policy.hpp"
#if defined(RAJA_ENABLE_JIT)
#include "proteus/JitInterface.h"
#endif

#ifndef RAJA_jit_HPP
#define RAJA_jit_HPP

namespace RAJA
{
namespace jit
{

template<typename Lambda>
inline auto register_lambda(Lambda&& lambda)
{
#if defined RAJA_ENABLE_JIT
  return proteus::register_lambda(std::forward<Lambda>(lambda));
#else
  return std::forward<Lambda>(lambda);
#endif
}

}  // namespace jit
}  // namespace RAJA

#endif
