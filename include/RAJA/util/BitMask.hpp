/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a bit masking operator
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

#ifndef RAJA_util_BitMask_HPP
#define RAJA_util_BitMask_HPP

#include "RAJA/config.hpp"

namespace RAJA
{

template<camp::idx_t N>
struct LogBase2
{
  static constexpr camp::idx_t value = LogBase2<(N >> 1)>::value + 1;
  static constexpr bool is_exact     = ((1 << value) == N);
};

template<>
struct LogBase2<0>
{
  static constexpr camp::idx_t value = -1;
  static constexpr bool is_exact     = true;
};

/*!
 * A bit-masking operator
 *
 * Provides an operator that shifts and masks in input value to extract
 * a contiguous set of bits.
 *
 * result = (input >> Shift) & (Mask)
 *
 * Where mask is (1<<Width)-1, or the number of bits defined by Width.
 *
 *
 */
template<int Width, int Shift>
struct BitMask
{
  static constexpr int shift            = Shift;
  static constexpr int width            = Width;
  static constexpr int max_input_size   = 1 << (Shift + Width);
  static constexpr int max_masked_size  = 1 << Width;
  static constexpr int max_shifted_size = 1 << Shift;

  template<typename T>
  RAJA_HOST_DEVICE static constexpr T maskValue(T input)
  {
    return ((input >> (static_cast<T>(Shift))) &
            static_cast<T>((1 << (Width)) - 1));
  }

  template<typename T>
  RAJA_HOST_DEVICE static constexpr T getOuter(T input)
  {
    return ((input >> (static_cast<T>(Shift))) >> Width);
  }

  template<typename T>
  RAJA_HOST_DEVICE static constexpr T maskOuter(T input)
  {
    return (input & (static_cast<T>(-1) << (Width + Shift)));
  }
};

}  // namespace RAJA

#endif  // RAJA_util_BitMask_HPP
