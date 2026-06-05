/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
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

#ifndef RAJA_pattern_tensor_TensorRegister_HPP
#define RAJA_pattern_tensor_TensorRegister_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorLayout.hpp"
#include "RAJA/pattern/tensor/internal/TensorRef.hpp"

namespace RAJA
{
namespace internal
{
namespace expt
{
class TensorRegisterConcreteBase;
}
}  // namespace internal

namespace expt
{


template<typename REGISTER_POLICY, typename T, typename LAYOUT, typename SIZES>
class TensorRegister;

/*
 * Overload for:    arithmetic + TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<
        std::is_base_of<RAJA::internal::expt::TensorRegisterConcreteBase,
                        RIGHT>::value,
        bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator+(LEFT const& lhs, RIGHT const& rhs)
{
  return RIGHT(lhs).add(rhs);
}

/*
 * Overload for:    arithmetic - TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<
        std::is_base_of<RAJA::internal::expt::TensorRegisterConcreteBase,
                        RIGHT>::value,
        bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator-(LEFT const& lhs, RIGHT const& rhs)
{
  return RIGHT(lhs).subtract(rhs);
}

/*
 * Overload for:    arithmetic * TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<
        std::is_base_of<RAJA::internal::expt::TensorRegisterConcreteBase,
                        RIGHT>::value,
        bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator*(LEFT const& lhs, RIGHT const& rhs)
{
  return rhs.scale(lhs);
}

/*
 * Overload for:    arithmetic / TensorRegister

 */
template<
    typename LEFT,
    typename RIGHT,
    typename std::enable_if<std::is_arithmetic<LEFT>::value, bool>::type = true,
    typename std::enable_if<
        std::is_base_of<RAJA::internal::expt::TensorRegisterConcreteBase,
                        RIGHT>::value,
        bool>::type = true>
RAJA_INLINE RAJA_HOST_DEVICE RIGHT operator/(LEFT const& lhs, RIGHT const& rhs)
{
  return RIGHT(lhs).divide(rhs);
}

}  // namespace expt
}  // namespace RAJA

#include "RAJA/pattern/tensor/internal/TensorRegisterBase.hpp"

// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/tensor/arch.hpp"


#endif
