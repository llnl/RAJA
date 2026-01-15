//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef NEW_REDUCE_SEQ_REDUCE_HPP
#define NEW_REDUCE_SEQ_REDUCE_HPP

#include "RAJA/pattern/params/reducer.hpp"

namespace RAJA
{
namespace expt
{
namespace detail
{

// Init
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::seq_exec>> param_init(
    EXEC_POL const&,
    Reducer<OP, T, VOp>& red)
{
  red.m_valop.val = OP::identity();
}

// Combine
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::seq_exec>> param_combine(
    EXEC_POL const&,
    Reducer<OP, T, VOp>& out,
    const Reducer<OP, T, VOp>& in)
{
  out.m_valop.val = OP {}(out.m_valop.val, in.m_valop.val);
}

template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::seq_exec>> param_combine(
    EXEC_POL const&,
    Reducer<OP, T, VOp>&)
{
  // noopt
}

// Resolve
template<typename EXEC_POL, typename OP, typename T, typename VOp>
camp::concepts::enable_if<std::is_same<EXEC_POL, RAJA::seq_exec>> param_resolve(
    EXEC_POL const&,
    Reducer<OP, T, VOp>& red)
{
  red.combineTarget(red.m_valop.val);
}

}  //  namespace detail
}  //  namespace expt
}  //  namespace RAJA

#endif  //  NEW_REDUCE_SEQ_REDUCE_HPP
