/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a SIMD register abstraction.
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

#ifdef __AVX__

#ifndef RAJA_policy_vector_register_avx_float_HPP
#define RAJA_policy_vector_register_avx_float_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/builtin_compat.hpp"
#include "RAJA/pattern/tensor/internal/RegisterBase.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>

namespace RAJA
{
namespace expt
{

template<>
class Register<float, avx_register>
    : public internal::expt::RegisterBase<Register<float, avx_register>>
{
public:
  using base_type = internal::expt::RegisterBase<Register<float, avx_register>>;

  using register_policy = avx_register;
  using self_type       = Register<float, avx_register>;
  using element_type    = float;
  using register_type   = __m256;

  using int_vector_type = Register<int32_t, avx_register>;


private:
  register_type m_value;

  RAJA_INLINE
  static register_type make_register(element_type x0,
                                     element_type x1,
                                     element_type x2,
                                     element_type x3,
                                     element_type x4,
                                     element_type x5,
                                     element_type x6,
                                     element_type x7)
  {
    return _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
  }

  RAJA_INLINE
  __m256i createMask(camp::idx_t N) const
  {
    // Generate a mask
    return _mm256_set_epi32(N >= 8 ? -1 : 0, N >= 7 ? -1 : 0, N >= 6 ? -1 : 0,
                            N >= 5 ? -1 : 0, N >= 4 ? -1 : 0, N >= 3 ? -1 : 0,
                            N >= 2 ? -1 : 0, N >= 1 ? -1 : 0);
  }

  static RAJA_NOINLINE element_type load_scalar(
      element_type const* ptr)
  {
    element_type value;
    RAJA_BUILTIN_MEMCPY(&value, ptr, sizeof(value));
    return value;
  }

  RAJA_INLINE
  static register_type load_register(element_type const* ptr)
  {
    return make_register(load_scalar(ptr + 0),
                         load_scalar(ptr + 1),
                         load_scalar(ptr + 2),
                         load_scalar(ptr + 3),
                         load_scalar(ptr + 4),
                         load_scalar(ptr + 5),
                         load_scalar(ptr + 6),
                         load_scalar(ptr + 7));
  }

  RAJA_INLINE
  static void store_register(element_type* ptr, register_type value)
  {
    _mm256_storeu_ps(ptr, value);
  }

  RAJA_INLINE
  self_type& assign_register(register_type value)
  {
    RAJA_BUILTIN_MEMCPY(&m_value, &value, sizeof(m_value));
    return *this;
  }

  RAJA_INLINE
  static element_type lane(register_type value, camp::idx_t i)
  {
    element_type lanes[s_num_elem];
    store_register(lanes, value);
    return lanes[i];
  }

public:
  static constexpr camp::idx_t s_num_elem = 8;

  /*!
   * @brief Default constructor, zeros register contents
   */
  RAJA_INLINE
  Register() : base_type(), m_value(_mm256_setzero_ps()) {}

  /*!
   * @brief Copy constructor from underlying simd register
   */
  RAJA_INLINE
  explicit Register(register_type const& c) : base_type(), m_value(c) {}

  /*!
   * @brief Construct register with explicit values
   */
  RAJA_INLINE
  Register(element_type x0,
           element_type x1,
           element_type x2,
           element_type x3,
           element_type x4,
           element_type x5,
           element_type x6,
           element_type x7)
      : m_value(_mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0))
  {}

  /*!
   * @brief Copy constructor
   */
  RAJA_INLINE
  Register(self_type const& c) : base_type(), m_value(_mm256_setzero_ps())
  {
    assign_register(c.m_value);
  }

  /*!
   * @brief Copy assignment constructor
   */
  RAJA_INLINE
  self_type& operator=(self_type const& c)
  {
    return assign_register(c.m_value);
  }

  /*!
   * @brief Construct from scalar.
   * Sets all elements to same value (broadcast).
   */
  RAJA_INLINE
  Register(element_type const& c) : m_value(_mm256_set1_ps(c)) {}

  /*!
   * @brief Load a full register from a stride-one memory location
   *
   */
  RAJA_INLINE
  self_type& load_packed(element_type const* ptr)
  {
    return assign_register(load_register(ptr));
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE
  self_type& load_packed_n(element_type const* ptr, camp::idx_t N)
  {
    m_value = _mm256_maskload_ps(ptr, createMask(N));
    return *this;
  }

  /*!
   * @brief Gather a full register from a strided memory location
   *
   */
  RAJA_INLINE
  self_type& load_strided(element_type const* ptr, camp::idx_t stride)
  {
    return assign_register(make_register(load_scalar(ptr + 0 * stride),
                                         load_scalar(ptr + 1 * stride),
                                         load_scalar(ptr + 2 * stride),
                                         load_scalar(ptr + 3 * stride),
                                         load_scalar(ptr + 4 * stride),
                                         load_scalar(ptr + 5 * stride),
                                         load_scalar(ptr + 6 * stride),
                                         load_scalar(ptr + 7 * stride)));
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE
  self_type& load_strided_n(element_type const* ptr,
                            camp::idx_t stride,
                            camp::idx_t N)
  {
    camp::idx_t count = N < s_num_elem ? N : s_num_elem;

    element_type x0 = count > 0 ? load_scalar(ptr + 0 * stride) : 0.0f;
    element_type x1 = count > 1 ? load_scalar(ptr + 1 * stride) : 0.0f;
    element_type x2 = count > 2 ? load_scalar(ptr + 2 * stride) : 0.0f;
    element_type x3 = count > 3 ? load_scalar(ptr + 3 * stride) : 0.0f;
    element_type x4 = count > 4 ? load_scalar(ptr + 4 * stride) : 0.0f;
    element_type x5 = count > 5 ? load_scalar(ptr + 5 * stride) : 0.0f;
    element_type x6 = count > 6 ? load_scalar(ptr + 6 * stride) : 0.0f;
    element_type x7 = count > 7 ? load_scalar(ptr + 7 * stride) : 0.0f;

    return assign_register(
        make_register(x0, x1, x2, x3, x4, x5, x6, x7));
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed(element_type* ptr) const
  {
    store_register(ptr, m_value);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed_n(element_type* ptr, camp::idx_t N) const
  {
    _mm256_maskstore_ps(ptr, createMask(N), m_value);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided(element_type* ptr, camp::idx_t stride) const
  {
    element_type lanes[s_num_elem];
    store_register(lanes, m_value);
    for (camp::idx_t i = 0; i < 8; ++i)
    {
      ptr[i * stride] = lanes[i];
    }
    return *this;
  }

  /*!
   * @brief Store partial register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided_n(element_type* ptr,
                                   camp::idx_t stride,
                                   camp::idx_t N) const
  {
    element_type lanes[s_num_elem];
    store_register(lanes, m_value);
    for (camp::idx_t i = 0; i < N; ++i)
    {
      ptr[i * stride] = lanes[i];
    }
    return *this;
  }

  /*!
   * @brief Get scalar value from vector register
   * @param i Offset of scalar to get
   * @return Returns scalar value at i
   */
  RAJA_INLINE
  element_type get(camp::idx_t i) const { return lane(m_value, i); }

  /*!
   * @brief Set scalar value in vector register
   * @param i Offset of scalar to set
   * @param value Value of scalar to set
   */
  RAJA_INLINE
  self_type& set(element_type value, camp::idx_t i)
  {
    element_type lanes[s_num_elem];
    store_register(lanes, m_value);
    lanes[i] = value;
    m_value = load_register(lanes);
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type const& value)
  {
    m_value = _mm256_set1_ps(value);
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& copy(self_type const& src)
  {
    return assign_register(src.m_value);
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type add(self_type const& b) const
  {
    return self_type(_mm256_add_ps(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    return self_type(_mm256_sub_ps(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {
    return self_type(_mm256_mul_ps(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& b) const
  {
    return self_type(_mm256_div_ps(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, camp::idx_t N) const
  {
    // AVX2 does not supply a masked divide
    return self_type(_mm256_set_ps(
        N >= 8 ? get(7) / b.get(7) : 0, N >= 7 ? get(6) / b.get(6) : 0,
        N >= 6 ? get(5) / b.get(5) : 0, N >= 5 ? get(4) / b.get(4) : 0,
        N >= 4 ? get(3) / b.get(3) : 0, N >= 3 ? get(2) / b.get(2) : 0,
        N >= 2 ? get(1) / b.get(1) : 0, N >= 1 ? get(0) / b.get(0) : 0));
  }

  /*!
   * @brief Sum the elements of this vector
   * @return Sum of the values of the vectors scalar elements
   */
  RAJA_INLINE
  element_type sum() const
  {
    // swap odd-even pairs and add
    auto sh1  = _mm256_permute_ps(m_value, 0xB1);
    auto red1 = _mm256_add_ps(m_value, sh1);

    // swap odd-even quads and add
    auto sh2  = _mm256_permute_ps(red1, 0x4E);
    auto red2 = _mm256_add_ps(red1, sh2);

    return lane(red2, 0) + lane(red2, 4);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max() const
  {
    // swap odd-even pairs and combine
    auto sh1  = _mm256_permute_ps(m_value, 0xB1);
    auto red1 = _mm256_max_ps(m_value, sh1);

    // swap odd-even quads and combine
    auto sh2  = _mm256_permute_ps(red1, 0x4E);
    auto red2 = _mm256_max_ps(red1, sh2);

    // combine quads
    return RAJA::max<element_type>(lane(red2, 0), lane(red2, 4));
  }

  /*!
   * @brief Returns the largest element of first N lanes
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max_n(camp::idx_t N) const
  {
    // Some simple cases
    if (N <= 0 || N > 8)
    {
      return RAJA::operators::limits<float>::min();
    }
    if (N == 1)
    {
      return get(0);
    }
    if (N == 2)
    {
      return RAJA::max<element_type>(get(0), get(1));
    }

    // swap odd-even pairs and add
    auto sh1 = _mm256_permute_ps(m_value, 0xB1);

    if (N == 7)
    {
      // blend out the 8th lane of the permute
      sh1 = _mm256_blend_ps(sh1, m_value, 0x40);
    }

    auto red1 = _mm256_max_ps(m_value, sh1);

    // Some more simple shortcuts
    if (N == 3)
    {
      return RAJA::max<element_type>(lane(red1, 0), get(2));
    }


    // swap odd-even quads and add
    auto sh2  = _mm256_permute_ps(red1, 0x4E);
    auto red2 = _mm256_max_ps(red1, sh2);

    if (N == 4)
    {
      return lane(red2, 0);
    }
    if (N == 5)
    {
      return RAJA::max<element_type>(lane(red2, 0), get(4));
    }
    if (N == 6)
    {
      return RAJA::max<element_type>(lane(red2, 0), lane(red1, 4));
    }

    // 7 or 8 lanes
    return RAJA::max<element_type>(lane(red2, 0), lane(red2, 4));
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmax(self_type a) const
  {
    return self_type(_mm256_max_ps(m_value, a.m_value));
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min() const
  {
    // swap odd-even pairs and combine
    auto sh1  = _mm256_permute_ps(m_value, 0xB1);
    auto red1 = _mm256_min_ps(m_value, sh1);

    // swap odd-even quads and combine
    auto sh2  = _mm256_permute_ps(red1, 0x4E);
    auto red2 = _mm256_min_ps(red1, sh2);

    // combine quads
    return RAJA::min<element_type>(lane(red2, 0), lane(red2, 4));
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min_n(camp::idx_t N) const
  {
    // Some simple cases
    if (N <= 0 || N > 8)
    {
      return RAJA::operators::limits<float>::max();
    }
    if (N == 1)
    {
      return get(0);
    }
    if (N == 2)
    {
      return RAJA::min<element_type>(get(0), get(1));
    }

    // swap odd-even pairs and add
    auto sh1 = _mm256_permute_ps(m_value, 0xB1);

    if (N == 7)
    {
      // blend out the 8th lane of the permute
      sh1 = _mm256_blend_ps(sh1, m_value, 0x40);
    }

    auto red1 = _mm256_min_ps(m_value, sh1);

    // Some more simple shortcuts
    if (N == 3)
    {
      return RAJA::min<element_type>(lane(red1, 0), get(2));
    }


    // swap odd-even quads and add
    auto sh2  = _mm256_permute_ps(red1, 0x4E);
    auto red2 = _mm256_min_ps(red1, sh2);

    if (N == 4)
    {
      return lane(red2, 0);
    }
    if (N == 5)
    {
      return RAJA::min<element_type>(lane(red2, 0), get(4));
    }
    if (N == 6)
    {
      return RAJA::min<element_type>(lane(red2, 0), lane(red1, 4));
    }

    // 7 or 8 lanes
    return RAJA::min<element_type>(lane(red2, 0), lane(red2, 4));
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmin(self_type a) const
  {
    return self_type(_mm256_min_ps(m_value, a.m_value));
  }
};


}  // namespace expt

}  // namespace RAJA


#endif

#endif  //__AVX__
