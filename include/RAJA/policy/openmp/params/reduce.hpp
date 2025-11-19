#ifndef NEW_REDUCE_OMP_REDUCE_HPP
#define NEW_REDUCE_OMP_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

#if defined(RAJA_ENABLE_OPENMP)

// Init
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_openmp_policy<EXEC_POL>>
param_init(EXEC_POL const&, Reducer<OP, T, VOp>& red)
{
  red.m_valop.val = OP::identity();
}

// Combine
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_openmp_policy<EXEC_POL>>
param_combine(EXEC_POL const&,
              Reducer<OP, T, VOp>& out,
              const Reducer<OP, T, VOp>& in)
{
  out.m_valop.val = OP {}(out.m_valop.val, in.m_valop.val);
}

// Resolve
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<RAJA::type_traits::is_openmp_policy<EXEC_POL>>
param_resolve(EXEC_POL const&, Reducer<OP, T, VOp>& red)
{
  red.combineTarget(red.m_valop.val);
}

#endif

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif  //  NEW_REDUCE_OMP_REDUCE_HPP
