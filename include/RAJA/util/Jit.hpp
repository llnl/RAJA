#include "RAJA/config.hpp"
#if defined(RAJA_ENABLE_JIT)
#include "proteus/JitInterface.hpp"
#endif

#ifndef RAJA_jit_HPP
#define RAJA_jit_HPP

namespace RAJA
{
namespace internal
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

template<typename T>
inline auto jit_variable(T arg)
{
#if defined RAJA_ENABLE_JIT
  return proteus::jit_variable(std::forward<T>(arg));
#else
  return std::forward<T>(arg);
#endif
}

}  // namespace jit
}  // namespace internal
}  // namespace RAJA

#endif
