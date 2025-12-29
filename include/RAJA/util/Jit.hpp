#include "RAJA/config.hpp"
#if defined(RAJA_ENABLE_JIT)
#include "proteus/JitInterface.hpp"
#endif

#ifndef RAJA_plugins_HPP
#define RAJA_plugins_HPP

namespace RAJA {
  template<typename Lambda>
  inline auto register_lambda(Lambda&& lambda) {
    #if defined RAJA_ENABLE_JIT
      return proteus::register_lambda(std::forward<Lambda>(lambda));
    #endif
    return std::forward<Lambda>(lambda);
  }
}

#endif
