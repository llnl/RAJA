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

#ifndef RAJA_policy_vector_register_scalar_HPP
#define RAJA_policy_vector_register_scalar_HPP

#include "RAJA/pattern/tensor/internal/RegisterBase.hpp"

namespace RAJA
{
namespace expt
{

/**
 * A specialization for a single element register.
 * We will implement this as a scalar value, and let the compiler use
 * whatever registers it deems appropriate.
 */
template<typename T>
class Register<T, scalar_register>
    : public internal::expt::RegisterBase<Register<T, scalar_register>>
{
public:
  using base_type = internal::expt::RegisterBase<Register<T, scalar_register>>;

  using register_policy = scalar_register;
  using self_type       = Register<T, scalar_register>;
  using element_type    = T;
  using register_type   = T;

  using int_vector_type =
      Register<typename internal::expt::RegisterTraits<scalar_register,
                                                       T>::int_element_type,
               scalar_register>;


private:
  T m_value;

public:
  static constexpr camp::idx_t s_num_elem = 1;

  /*!
   * @brief Default constructor, zeros register contents
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr Register() : base_type(), m_value(0) {}

  /*!
   * @brief Copy constructor from underlying simd register
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr Register(element_type const& c) : base_type(), m_value(c) {}

  /*!
   * @brief Copy constructor
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr Register(self_type const& c) : base_type(), m_value(c.m_value) {}

  /*!
   * @brief Copy assignment constructor
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& operator=(self_type const& c)
  {
    m_value = c.m_value;
    return *this;
  }

  /*!
   * @brief Load a full register from a stride-one memory location
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_packed(element_type const* ptr)
  {
    m_value = ptr[0];
    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_packed_n(element_type const* ptr, camp::idx_t N)
  {
    if (N > 0)
    {
      m_value = ptr[0];
    }
    else
    {
      m_value = element_type(0);
    }
    return *this;
  }

  /*!
   * @brief Gather a full register from a strided memory location
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_strided(element_type const* ptr, camp::idx_t)
  {
    m_value = ptr[0];
    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& load_strided_n(element_type const* ptr, camp::idx_t, camp::idx_t N)
  {
    if (N > 0)
    {
      m_value = ptr[0];
    }
    else
    {
      m_value = element_type(0);
    }
    return *this;
  }

  /*!
   * @brief Generic gather operation for full vector.
   *
   * Must provide another register containing offsets of all values
   * to be loaded relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_INLINE
  self_type& gather(element_type const* ptr, int_vector_type offsets)
  {

    m_value = ptr[offsets.get(0)];

    return *this;
  }

  /*!
   * @brief Generic gather operation for n-length subvector.
   *
   * Must provide another register containing offsets of all values
   * to be loaded relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_INLINE
  self_type& gather_n(element_type const* ptr,
                      int_vector_type offsets,
                      camp::idx_t N)
  {
    if (N > 0)
    {
      m_value = ptr[offsets.get(0)];
    }
    else
    {
      m_value = element_type(0);
    }
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_packed(element_type* ptr) const
  {
    ptr[0] = m_value;
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_packed_n(element_type* ptr, camp::idx_t N) const
  {
    if (N > 0)
    {
      ptr[0] = m_value;
    }
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_strided(element_type* ptr, camp::idx_t) const
  {
    ptr[0] = m_value;
    return *this;
  }

  /*!
   * @brief Store partial register to consecutive memory locations
   *
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type const& store_strided_n(element_type* ptr,
                                   camp::idx_t,
                                   camp::idx_t N) const
  {
    if (N > 0)
    {
      ptr[0] = m_value;
    }
    return *this;
  }

  /*!
   * @brief Generic scatter operation for full vector.
   *
   * Must provide another register containing offsets of all values
   * to be stored relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_INLINE
  self_type const& scatter(element_type* ptr, int_vector_type offsets) const
  {

    ptr[offsets.get(0)] = m_value;

    return *this;
  }

  /*!
   * @brief Generic scatter operation for n-length subvector.
   *
   * Must provide another register containing offsets of all values
   * to be stored relative to supplied pointer.
   *
   * Offsets are element-wise, not byte-wise.
   *
   */
  RAJA_INLINE
  self_type const& scatter_n(element_type* ptr,
                             int_vector_type offsets,
                             camp::idx_t N) const
  {
    if (N > 0)
    {
      ptr[offsets.get(0)] = m_value;
    }
    return *this;
  }

  /*!
   * @brief Get scalar value from vector register
   * @param i Offset of scalar to get
   * @return Returns scalar value at i
   */
  constexpr RAJA_INLINE RAJA_HOST_DEVICE element_type get(camp::idx_t) const
  {
    return m_value;
  }

  /*!
   * @brief Set scalar value in vector register
   * @param i Offset of scalar to set
   * @param value Value of scalar to set
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type& set(element_type value, camp::idx_t)
  {
    m_value = value;
    return *this;
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type const& a)
  {
    m_value = a;
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
    return self_type(m_value + b.m_value);
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    return self_type(m_value - b.m_value);
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {
    return self_type(m_value * b.m_value);
  }

  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& b) const
  {
    return self_type(m_value / b.m_value);
  }

  /*!
   * @brief Fused multiply add: fma(b, c) = (*this)*b+c
   *
   * Derived types can override this to implement intrinsic FMA's
   *
   * @param b Second product operand
   * @param c Sum operand
   * @return Value of (*this)*b+c
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type multiply_add(self_type const& b, self_type const& c) const
  {
    return m_value * b.m_value + c.m_value;
  }

  /*!
   * @brief Fused multiply subtract: fms(b, c) = (*this)*b-c
   *
   * Derived types can override this to implement intrinsic FMS's
   *
   * @param b Second product operand
   * @param c Subtraction operand
   * @return Value of (*this)*b-c
   */
  RAJA_INLINE

  RAJA_HOST_DEVICE
  self_type multiply_subtract(self_type const& b, self_type const& c) const
  {
    return m_value * b.m_value - c.m_value;
  }

  /*!
   * @brief Sum the elements of this vector
   * @return Sum of the values of the vectors scalar elements
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr element_type sum() const { return m_value; }

  /*!
   * @brief Sum the elements of this vector
   * @return Sum of the values of the vectors scalar elements
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr element_type dot(self_type const& b) const
  {
    return m_value * b.m_value;
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  constexpr element_type max() const { return m_value; }

  /*!
   * @brief Returns the largest element from first N lanes
   * @return The largest scalar element in the register
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  element_type max_n(camp::idx_t N) const
  {
    return N ? m_value : RAJA::operators::limits<element_type>::min();
    ;
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type vmax(self_type a) const
  {
    return self_type(RAJA::max<element_type>(m_value, a.m_value));
  }

  /*!
   * @brief Returns the smallest element
   * @return The smallest scalar element in the register
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  element_type min() const { return m_value; }

  /*!
   * @brief Returns the smallest element from first N lanes
   * @return The smallest scalar element in the register
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  element_type min_n(camp::idx_t N) const
  {
    return N ? m_value : RAJA::operators::limits<element_type>::max();
    ;
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_HOST_DEVICE

  RAJA_INLINE
  self_type vmin(self_type a) const
  {
    return self_type(RAJA::min<element_type>(m_value, a.m_value));
  }
};
}  // namespace expt
}  // namespace RAJA


#endif
