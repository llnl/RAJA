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

#ifndef RAJA_policy_vector_register_avx_int32_HPP
#define RAJA_policy_vector_register_avx_int32_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/internal/RegisterBase.hpp"

// Include SIMD intrinsics header file
#include <immintrin.h>
#include <cmath>

namespace RAJA
{
namespace expt
{
template<>
class Register<int32_t, avx_register>
    : public internal::expt::RegisterBase<Register<int32_t, avx_register>>
{
public:
  using base_type =
      internal::expt::RegisterBase<Register<int32_t, avx_register>>;

  using register_policy = avx_register;
  using self_type       = Register<int32_t, avx_register>;
  using element_type    = int32_t;
  using register_type   = __m256i;

  using int_vector_type = Register<int32_t, avx_register>;


private:
  register_type m_value;

  RAJA_INLINE
  __m256i createMask(camp::idx_t N) const
  {
    // Generate a mask
    return _mm256_set_epi32(N >= 8 ? -1 : 0, N >= 7 ? -1 : 0, N >= 6 ? -1 : 0,
                            N >= 5 ? -1 : 0, N >= 4 ? -1 : 0, N >= 3 ? -1 : 0,
                            N >= 2 ? -1 : 0, N >= 1 ? -1 : 0);
  }

  RAJA_INLINE
  __m256i createStridedOffsets(camp::idx_t stride) const
  {
    // Generate a strided offset list
    return _mm256_set_epi32(7 * stride, 6 * stride, 5 * stride, 4 * stride,
                            3 * stride, 2 * stride, stride, 0);
  }

  RAJA_INLINE
  __m256i createPermute1(camp::idx_t N) const
  {
    // Generate a permutation for first round of min/max routines
    return _mm256_set_epi32(N >= 7 ? 6 : 0, N >= 8 ? 7 : 0, N >= 5 ? 4 : 0,
                            N >= 6 ? 5 : 0, N >= 3 ? 2 : 0, N >= 4 ? 3 : 0,
                            N >= 1 ? 0 : 0, N >= 2 ? 1 : 0);
  }

  RAJA_INLINE
  __m256i createPermute2(camp::idx_t N) const
  {
    // Generate a permutation for second round of min/max routines
    return _mm256_set_epi32(N >= 6 ? 5 : 0, N >= 5 ? 4 : 0, N >= 8 ? 7 : 0,
                            N >= 7 ? 6 : 0, N >= 2 ? 1 : 0, N >= 1 ? 0 : 0,
                            N >= 4 ? 3 : 0, N >= 2 ? 2 : 0);
  }

public:
  static constexpr camp::idx_t s_num_elem = 8;

  /*!
   * @brief Default constructor, zeros register contents
   */
  RAJA_INLINE
  Register() : base_type(), m_value(_mm256_setzero_si256()) {}

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
      : m_value(_mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0))
  {}

  /*!
   * @brief Copy constructor
   */
  RAJA_INLINE
  Register(self_type const& c) : base_type(), m_value(c.m_value) {}

  /*!
   * @brief Copy assignment constructor
   */
  RAJA_INLINE
  self_type& operator=(self_type const& c)
  {
    m_value = c.m_value;
    return *this;
  }

  /*!
   * @brief Construct from scalar.
   * Sets all elements to same value (broadcast).
   */
  RAJA_INLINE
  Register(element_type const& c) : m_value(_mm256_set1_epi32(c)) {}

  /*!
   * @brief Load a full register from a stride-one memory location
   *
   */
  RAJA_INLINE
  self_type& load_packed(element_type const* ptr)
  {
    m_value = _mm256_loadu_si256((__m256i const*)ptr);
    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE
  self_type& load_packed_n(element_type const* ptr, camp::idx_t N)
  {
    m_value = _mm256_setzero_si256();
    for (camp::idx_t i = 0; i < N; ++i)
    {
      set(ptr[i], i);
    }
    return *this;
  }

  /*!
   * @brief Gather a full register from a strided memory location
   *
   */
  RAJA_INLINE
  self_type& load_strided(element_type const* ptr, camp::idx_t stride)
  {
    for (camp::idx_t i = 0; i < 8; ++i)
    {
      set(ptr[i * stride], i);
    }
    return *this;
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
    m_value = _mm256_setzero_si256();
    for (camp::idx_t i = 0; i < N; ++i)
    {
      set(ptr[i * stride], i);
    }
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed(element_type* ptr) const
  {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), m_value);
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_packed_n(element_type* ptr, camp::idx_t N) const
  {
    _mm256_maskstore_ps(reinterpret_cast<float*>(ptr), createMask(N),
                        reinterpret_cast<__m256>(m_value));
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE
  self_type const& store_strided(element_type* ptr, camp::idx_t stride) const
  {
    for (camp::idx_t i = 0; i < 8; ++i)
    {
      ptr[i * stride] = get(i);
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
    for (camp::idx_t i = 0; i < N; ++i)
    {
      ptr[i * stride] = get(i);
    }
    return *this;
  }

  /*!
   * @brief Get scalar value from vector register
   * @param i Offset of scalar to get
   * @return Returns scalar value at i
   */
  RAJA_INLINE
  element_type get(camp::idx_t i) const
  {
    // got to be a nicer way to do this!?!?
    switch (i)
    {
      case 0:
        return _mm256_extract_epi32(m_value, 0);
      case 1:
        return _mm256_extract_epi32(m_value, 1);
      case 2:
        return _mm256_extract_epi32(m_value, 2);
      case 3:
        return _mm256_extract_epi32(m_value, 3);
      case 4:
        return _mm256_extract_epi32(m_value, 4);
      case 5:
        return _mm256_extract_epi32(m_value, 5);
      case 6:
        return _mm256_extract_epi32(m_value, 6);
      case 7:
        return _mm256_extract_epi32(m_value, 7);
    }
    return 0;
  }

  /*!
   * @brief Set scalar value in vector register
   * @param i Offset of scalar to set
   * @param value Value of scalar to set
   */
  RAJA_INLINE
  self_type& set(element_type value, camp::idx_t i)
  {
    // got to be a nicer way to do this!?!?
    switch (i)
    {
      case 0:
        m_value = _mm256_insert_epi32(m_value, value, 0);
        break;
      case 1:
        m_value = _mm256_insert_epi32(m_value, value, 1);
        break;
      case 2:
        m_value = _mm256_insert_epi32(m_value, value, 2);
        break;
      case 3:
        m_value = _mm256_insert_epi32(m_value, value, 3);
        break;
      case 4:
        m_value = _mm256_insert_epi32(m_value, value, 4);
        break;
      case 5:
        m_value = _mm256_insert_epi32(m_value, value, 5);
        break;
      case 6:
        m_value = _mm256_insert_epi32(m_value, value, 6);
        break;
      case 7:
        m_value = _mm256_insert_epi32(m_value, value, 7);
        break;
    }

    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type const& value)
  {
    m_value = _mm256_set1_epi32(value);
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& copy(self_type const& src)
  {
    m_value = src.m_value;
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type add(self_type const& b) const
  {
    // no 8-way 32-bit add, but there is a 4-way... split and conquer

    // Low 128-bits  - use _mm256_castsi256_si128???
    auto low_a   = _mm256_castsi256_si128(m_value);
    auto low_b   = _mm256_castsi256_si128(b.m_value);
    auto res_low = _mm256_castsi128_si256(_mm_add_epi32(low_a, low_b));

    // Hi 128-bits
    auto hi_a   = _mm256_extractf128_si256(m_value, 1);
    auto hi_b   = _mm256_extractf128_si256(b.m_value, 1);
    auto res_hi = _mm_add_epi32(hi_a, hi_b);

    // Stitch back together
    return self_type(_mm256_insertf128_si256(res_low, res_hi, 1));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    // no 8-way 32-bit subtract, but there is a 4-way... split and conquer

    // Low 128-bits
    auto low_a   = _mm256_castsi256_si128(m_value);
    auto low_b   = _mm256_castsi256_si128(b.m_value);
    auto res_low = _mm256_castsi128_si256(_mm_sub_epi32(low_a, low_b));

    // Hi 128-bits
    auto hi_a   = _mm256_extractf128_si256(m_value, 1);
    auto hi_b   = _mm256_extractf128_si256(b.m_value, 1);
    auto res_hi = _mm_sub_epi32(hi_a, hi_b);

    // Stitch back together
    return self_type(_mm256_insertf128_si256(res_low, res_hi, 1));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {
    // no 8-way 32-bit multiply, but there is a 32x32 -> 64
    // This gets ugly :)

    // Low 128-bits
    auto low_a = _mm256_castsi256_si128(m_value);
    auto low_b = _mm256_castsi256_si128(b.m_value);
    // multiply even lanes 0, 2
    auto res_low_even = _mm_mul_epi32(low_a, low_b);

    // multiply odd lanes 1, 3
    auto low_a_sh    = _mm_shuffle_epi32(low_a, 0xB1);
    auto low_b_sh    = _mm_shuffle_epi32(low_b, 0xB1);
    auto res_low_odd = _mm_mul_epi32(low_a_sh, low_b_sh);

    // recombine to get all 4 lanes
    // note: AVX doesn't have a int32 blend, so we use the float32 blend
    res_low_odd  = _mm_shuffle_epi32(res_low_odd, 0xB1);
    auto res_low = _mm256_castsi128_si256(_mm_castps_si128(_mm_blend_ps(
        _mm_castsi128_ps(res_low_odd), _mm_castsi128_ps(res_low_even), 0x05)));


    // High 128-bits
    auto hi_a = _mm256_extractf128_si256(m_value, 1);
    auto hi_b = _mm256_extractf128_si256(b.m_value, 1);
    // multiply even lanes 0, 2
    auto res_hi_even = _mm_mul_epi32(hi_a, hi_b);

    // multiply odd lanes 1, 3
    auto hi_a_sh    = _mm_shuffle_epi32(hi_a, 0xB1);
    auto hi_b_sh    = _mm_shuffle_epi32(hi_b, 0xB1);
    auto res_hi_odd = _mm_mul_epi32(hi_a_sh, hi_b_sh);

    // recombine to get all 4 lanes
    // note: AVX doesn't have a int32 blend, so we use the float32 blend
    res_hi_odd  = _mm_shuffle_epi32(res_hi_odd, 0xB1);
    auto res_hi = _mm_castps_si128(_mm_blend_ps(
        _mm_castsi128_ps(res_hi_odd), _mm_castsi128_ps(res_hi_even), 0x05));

    // Stitch back together
    return self_type(_mm256_insertf128_si256(res_low, res_hi, 1));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& b) const
  {
    // AVX2 does not supply an integer divide, so do it manually
    return self_type(_mm256_set_epi32(get(7) / b.get(7), get(6) / b.get(6),
                                      get(5) / b.get(5), get(4) / b.get(4),
                                      get(3) / b.get(3), get(2) / b.get(2),
                                      get(1) / b.get(1), get(0) / b.get(0)));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, camp::idx_t N) const
  {
    // AVX2 does not supply an integer divide, so do it manually
    return self_type(_mm256_set_epi32(
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
    // Low 128-bits
    auto low = _mm256_castsi256_si128(m_value);

    auto low_sh1  = _mm_shuffle_epi32(low, 0xB1);
    auto low_red1 = _mm_add_epi32(low, low_sh1);

    auto low_sh2  = _mm_shuffle_epi32(low_red1, 0x1B);
    auto low_red2 = _mm_add_epi32(low_red1, low_sh2);


    // High 128-bits
    auto hi = _mm256_extractf128_si256(m_value, 1);

    auto hi_sh1  = _mm_shuffle_epi32(hi, 0xB1);
    auto hi_red1 = _mm_add_epi32(hi, hi_sh1);

    auto hi_sh2  = _mm_shuffle_epi32(hi_red1, 0x1B);
    auto hi_red2 = _mm_add_epi32(hi_red1, hi_sh2);


    // Sum halves, extract total sum
    auto hi_low = _mm_add_epi32(hi_red2, low_red2);
    return _mm_extract_epi32(hi_low, 0);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max() const
  {
    // this is just painful, since we don't have a proper masked permute
    // in AVX.  Lots of special cases to make sure we compare just the
    // right lanes


    // Low 128-bits
    auto low = _mm256_castsi256_si128(m_value);

    auto low_sh1  = _mm_shuffle_epi32(low, 0xB1);
    auto low_red1 = _mm_max_epi32(low, low_sh1);

    auto low_sh2 = _mm_shuffle_epi32(low_red1, 0x1B);

    // lane 0 of low_red2 now has reduction of 0,1,2,3
    auto low_red2 = _mm_max_epi32(low_red1, low_sh2);


    // High 128-bits
    auto hi = _mm256_extractf128_si256(m_value, 1);


    auto hi_sh1  = _mm_shuffle_epi32(hi, 0xB1);
    auto hi_red1 = _mm_max_epi32(hi, hi_sh1);

    auto hi_sh2  = _mm_shuffle_epi32(hi_red1, 0x1B);
    auto hi_red2 = _mm_max_epi32(hi_red1, hi_sh2);


    // Sum halves, extract final reduction
    auto hi_low = _mm_max_epi32(hi_red2, low_red2);
    return _mm_extract_epi32(hi_low, 0);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max_n(camp::idx_t N) const
  {
    // Some simple cases
    if (N <= 0 || N > 8)
    {
      return RAJA::operators::limits<int32_t>::min();
    }

    // this is just painful, since we don't have a proper masked permute
    // in AVX.  Lots of special cases to make sure we compare just the
    // right lanes
    if (N == 1)
    {
      return _mm256_extract_epi32(m_value, 0);
    }

    // Low 128-bits
    auto low = _mm256_castsi256_si128(m_value);

    auto low_sh1  = _mm_shuffle_epi32(low, 0xB1);
    auto low_red1 = _mm_max_epi32(low, low_sh1);

    if (N == 2)
    {
      return _mm_extract_epi32(low_red1, 0);
    }

    if (N == 3)
    {
      // get lane 2 into lane 0
      auto low_sh1a  = _mm_shuffle_epi32(low, 0x2);
      auto low_red1a = _mm_max_epi32(low_red1, low_sh1a);
      return _mm_extract_epi32(low_red1a, 0);
    }

    auto low_sh2 = _mm_shuffle_epi32(low_red1, 0x1B);

    // lane 0 of low_red2 now has reduction of 0,1,2,3
    auto low_red2 = _mm_max_epi32(low_red1, low_sh2);

    if (N == 4)
    {
      return _mm_extract_epi32(low_red2, 0);
    }

    // High 128-bits
    auto hi = _mm256_extractf128_si256(m_value, 1);

    if (N == 5)
    {
      auto red_5 = _mm_max_epi32(low_red2, hi);
      return _mm_extract_epi32(red_5, 0);
    }

    auto hi_sh1  = _mm_shuffle_epi32(hi, 0xB1);
    auto hi_red1 = _mm_max_epi32(hi, hi_sh1);

    if (N == 6)
    {
      auto red_6 = _mm_max_epi32(low_red2, hi_red1);
      return _mm_extract_epi32(red_6, 0);
    }
    if (N == 7)
    {
      // get lane 6 (lane 2 of hi) into lane 0
      auto hi_sh7   = _mm_shuffle_epi32(hi, 0x2);
      auto hi_red_6 = _mm_max_epi32(hi_sh7, hi_red1);
      auto red_7    = _mm_max_epi32(low_red2, hi_red_6);
      return _mm_extract_epi32(red_7, 0);
    }

    auto hi_sh2  = _mm_shuffle_epi32(hi_red1, 0x1B);
    auto hi_red2 = _mm_max_epi32(hi_red1, hi_sh2);


    // Sum halves, extract total sum
    auto hi_low = _mm_max_epi32(hi_red2, low_red2);
    return _mm_extract_epi32(hi_low, 0);
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmax(self_type b) const
  {
    // no 8-way 32-bit min, but there is a 4-way... split and conquer

    // Low 128-bits  - use _mm256_castsi256_si128???
    auto low_a   = _mm256_castsi256_si128(m_value);
    auto low_b   = _mm256_castsi256_si128(b.m_value);
    auto res_low = _mm256_castsi128_si256(_mm_max_epi32(low_a, low_b));

    // Hi 128-bits
    auto hi_a   = _mm256_extractf128_si256(m_value, 1);
    auto hi_b   = _mm256_extractf128_si256(b.m_value, 1);
    auto res_hi = _mm_max_epi32(hi_a, hi_b);

    // Stitch back together
    return self_type(_mm256_insertf128_si256(res_low, res_hi, 1));
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min() const
  {
    // this is just painful, since we don't have a proper masked permute
    // in AVX.  Lots of special cases to make sure we compare just the
    // right lanes

    // Low 128-bits
    auto low = _mm256_castsi256_si128(m_value);

    auto low_sh1  = _mm_shuffle_epi32(low, 0xB1);
    auto low_red1 = _mm_min_epi32(low, low_sh1);

    auto low_sh2 = _mm_shuffle_epi32(low_red1, 0x1B);

    // lane 0 of low_red2 now has reduction of 0,1,2,3
    auto low_red2 = _mm_min_epi32(low_red1, low_sh2);


    // High 128-bits
    auto hi = _mm256_extractf128_si256(m_value, 1);

    auto hi_sh1  = _mm_shuffle_epi32(hi, 0xB1);
    auto hi_red1 = _mm_min_epi32(hi, hi_sh1);


    auto hi_sh2  = _mm_shuffle_epi32(hi_red1, 0x1B);
    auto hi_red2 = _mm_min_epi32(hi_red1, hi_sh2);


    // Sum halves, extract total sum
    auto hi_low = _mm_min_epi32(hi_red2, low_red2);
    return _mm_extract_epi32(hi_low, 0);
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
      return RAJA::operators::limits<int32_t>::max();
    }
    // this is just painful, since we don't have a proper masked permute
    // in AVX.  Lots of special cases to make sure we compare just the
    // right lanes
    if (N == 1)
    {
      return _mm256_extract_epi32(m_value, 0);
    }

    // Low 128-bits
    auto low = _mm256_castsi256_si128(m_value);

    auto low_sh1  = _mm_shuffle_epi32(low, 0xB1);
    auto low_red1 = _mm_min_epi32(low, low_sh1);

    if (N == 2)
    {
      return _mm_extract_epi32(low_red1, 0);
    }

    if (N == 3)
    {
      // get lane 2 into lane 0
      auto low_sh1a  = _mm_shuffle_epi32(low, 0x2);
      auto low_red1a = _mm_min_epi32(low_red1, low_sh1a);
      return _mm_extract_epi32(low_red1a, 0);
    }

    auto low_sh2 = _mm_shuffle_epi32(low_red1, 0x1B);

    // lane 0 of low_red2 now has reduction of 0,1,2,3
    auto low_red2 = _mm_min_epi32(low_red1, low_sh2);

    if (N == 4)
    {
      return _mm_extract_epi32(low_red2, 0);
    }

    // High 128-bits
    auto hi = _mm256_extractf128_si256(m_value, 1);

    if (N == 5)
    {
      auto red_5 = _mm_min_epi32(low_red2, hi);
      return _mm_extract_epi32(red_5, 0);
    }

    auto hi_sh1  = _mm_shuffle_epi32(hi, 0xB1);
    auto hi_red1 = _mm_min_epi32(hi, hi_sh1);

    if (N == 6)
    {
      auto red_6 = _mm_min_epi32(low_red2, hi_red1);
      return _mm_extract_epi32(red_6, 0);
    }
    if (N == 7)
    {
      // get lane 6 (lane 2 of hi) into lane 0
      auto hi_sh7   = _mm_shuffle_epi32(hi, 0x2);
      auto hi_red_6 = _mm_min_epi32(hi_sh7, hi_red1);
      auto red_7    = _mm_min_epi32(low_red2, hi_red_6);
      return _mm_extract_epi32(red_7, 0);
    }

    auto hi_sh2  = _mm_shuffle_epi32(hi_red1, 0x1B);
    auto hi_red2 = _mm_min_epi32(hi_red1, hi_sh2);


    // Sum halves, extract total sum
    auto hi_low = _mm_min_epi32(hi_red2, low_red2);
    return _mm_extract_epi32(hi_low, 0);
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmin(self_type b) const
  {
    // no 8-way 32-bit min, but there is a 4-way... split and conquer

    // Low 128-bits  - use _mm256_castsi256_si128???
    auto low_a   = _mm256_castsi256_si128(m_value);
    auto low_b   = _mm256_castsi256_si128(b.m_value);
    auto res_low = _mm256_castsi128_si256(_mm_min_epi32(low_a, low_b));

    // Hi 128-bits
    auto hi_a   = _mm256_extractf128_si256(m_value, 1);
    auto hi_b   = _mm256_extractf128_si256(b.m_value, 1);
    auto res_hi = _mm_min_epi32(hi_a, hi_b);

    // Stitch back together
    return self_type(_mm256_insertf128_si256(res_low, res_hi, 1));
  }
};


}  // namespace expt

}  // namespace RAJA


#endif

#endif  //__AVX2__
