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

#ifdef __AVX2__

#ifndef RAJA_policy_vector_register_avx2_int32_HPP
#define RAJA_policy_vector_register_avx2_int32_HPP

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
class Register<int32_t, avx2_register>
    : public internal::expt::RegisterBase<Register<int32_t, avx2_register>>
{
public:
  using base_type =
      internal::expt::RegisterBase<Register<int32_t, avx2_register>>;

  using register_policy = avx2_register;
  using self_type       = Register<int32_t, avx2_register>;
  using element_type    = int32_t;
  using register_type   = __m256i;

  using int_vector_type = Register<int32_t, avx2_register>;


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
  Register() : m_value(_mm256_setzero_si256()) {}

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
   * @brief Copy constructor from underlying simd register
   */
  RAJA_INLINE
  explicit Register(register_type const& c) : m_value(c) {}

  /*!
   * @brief Copy constructor
   */
  RAJA_INLINE
  Register(self_type const& c) : base_type(c), m_value(c.m_value) {}

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
   * @brief Returns underlying SIMD register.
   */
  RAJA_INLINE
  constexpr register_type get_register() const { return m_value; }

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
    m_value = _mm256_maskload_epi32(ptr, createMask(N));
    return *this;
  }

  /*!
   * @brief Gather a full register from a strided memory location
   *
   */
  RAJA_INLINE
  self_type& load_strided(element_type const* ptr, camp::idx_t stride)
  {
    m_value = _mm256_i32gather_epi32(ptr, createStridedOffsets(stride),
                                     sizeof(element_type));
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
    m_value = _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), ptr,
                                          createStridedOffsets(stride),
                                          createMask(N), sizeof(element_type));
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
    _mm256_maskstore_epi32(ptr, createMask(N), m_value);
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
    return self_type(_mm256_add_epi32(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    return self_type(_mm256_sub_epi32(m_value, b.m_value));
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {

    // the AVX2 epi32 multiply only multiplies the even elements
    // and provides 64-bit results
    // need to do some repacking to get this to work

    // multiply 0, 2, 4, 6
    auto prod_even = _mm256_mul_epi32(m_value, b.m_value);

    // Swap 32-bit words
    auto sh_a = _mm256_castps_si256(
        _mm256_permute_ps(_mm256_castsi256_ps(m_value), 0xB1));

    auto sh_b = _mm256_castps_si256(
        _mm256_permute_ps(_mm256_castsi256_ps(b.m_value), 0xB1));

    // multiply 1, 3, 5, 7
    auto prod_odd = _mm256_mul_epi32(sh_a, sh_b);

    // Stitch prod_odd and prod_even back together
    auto sh_odd = _mm256_castps_si256(
        _mm256_permute_ps(_mm256_castsi256_ps(prod_odd), 0xB1));

    return self_type(_mm256_blend_epi32(prod_even, sh_odd, 0xAA));
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
    // swap odd-even pairs and add
    auto sh1 = _mm256_castps_si256(
        _mm256_permute_ps(_mm256_castsi256_ps(m_value), 0xB1));
    auto red1 = _mm256_add_epi32(m_value, sh1);


    // swap odd-even quads and add
    auto sh2 =
        _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(red1), 0x4E));
    auto red2 = _mm256_add_epi32(red1, sh2);

    return _mm256_extract_epi32(red2, 0) + _mm256_extract_epi32(red2, 4);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type max() const
  {

    // swap odd-even pairs and add
    auto sh1  = _mm256_permutevar8x32_epi32(m_value, createPermute1(8));
    auto red1 = _mm256_max_epi32(m_value, sh1);

    // swap odd-even quads and add
    auto sh2  = _mm256_permutevar8x32_epi32(red1, createPermute2(8));
    auto red2 = _mm256_max_epi32(red1, sh2);

    return std::max<element_type>(_mm256_extract_epi32(red2, 0),
                                  _mm256_extract_epi32(red2, 4));
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
    if (N == 1)
    {
      return get(0);
    }

    if (N == 2)
    {
      return std::max<element_type>(get(0), get(1));
    }

    // swap odd-even pairs and add
    auto sh1  = _mm256_permutevar8x32_epi32(m_value, createPermute1(N));
    auto red1 = _mm256_max_epi32(m_value, sh1);

    if (N == 3)
    {
      return std::max<element_type>(_mm256_extract_epi32(red1, 0), get(2));
    }
    if (N == 4)
    {
      return std::max<element_type>(_mm256_extract_epi32(red1, 0),
                                    _mm256_extract_epi32(red1, 2));
    }

    // swap odd-even quads and add
    auto sh2  = _mm256_permutevar8x32_epi32(red1, createPermute2(N));
    auto red2 = _mm256_max_epi32(red1, sh2);

    return std::max<element_type>(_mm256_extract_epi32(red2, 0),
                                  _mm256_extract_epi32(red2, 4));
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmax(self_type a) const
  {
    return self_type(_mm256_max_epi32(m_value, a.m_value));
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE
  element_type min() const
  {

    // swap odd-even pairs and add
    auto sh1  = _mm256_permutevar8x32_epi32(m_value, createPermute1(8));
    auto red1 = _mm256_min_epi32(m_value, sh1);


    // swap odd-even quads and add
    auto sh2  = _mm256_permutevar8x32_epi32(red1, createPermute2(8));
    auto red2 = _mm256_min_epi32(red1, sh2);

    return std::min<element_type>(_mm256_extract_epi32(red2, 0),
                                  _mm256_extract_epi32(red2, 4));
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
    if (N == 1)
    {
      return get(0);
    }

    if (N == 2)
    {
      return std::min<element_type>(get(0), get(1));
    }

    // swap odd-even pairs and add
    auto sh1  = _mm256_permutevar8x32_epi32(m_value, createPermute1(N));
    auto red1 = _mm256_min_epi32(m_value, sh1);

    if (N == 3)
    {
      return std::min<element_type>(_mm256_extract_epi32(red1, 0), get(2));
    }
    if (N == 4)
    {
      return std::min<element_type>(_mm256_extract_epi32(red1, 0),
                                    _mm256_extract_epi32(red1, 2));
    }

    // swap odd-even quads and add
    auto sh2  = _mm256_permutevar8x32_epi32(red1, createPermute2(N));
    auto red2 = _mm256_min_epi32(red1, sh2);

    return std::min<element_type>(_mm256_extract_epi32(red2, 0),
                                  _mm256_extract_epi32(red2, 4));
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE
  self_type vmin(self_type a) const
  {
    return self_type(_mm256_min_epi32(m_value, a.m_value));
  }
};


}  // namespace expt

}  // namespace RAJA


#endif

#endif  //__AVX2__
