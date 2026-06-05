/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing SIMT distrubuted register abstractions for
 *          CUDA
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

#ifndef RAJA_policy_tensor_arch_cuda_cuda_warp_register_HPP
#define RAJA_policy_tensor_arch_cuda_cuda_warp_register_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_CUDA_ACTIVE)

#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/tensor/internal/RegisterBase.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Operators.hpp"

#include "RAJA/policy/cuda/intrinsics.hpp"

namespace RAJA
{
namespace expt
{

template<typename ELEMENT_TYPE>
class Register<ELEMENT_TYPE, cuda_warp_register>
    : public internal::expt::RegisterBase<
          Register<ELEMENT_TYPE, cuda_warp_register>>
{
public:
  using base_type =
      internal::expt::RegisterBase<Register<ELEMENT_TYPE, cuda_warp_register>>;

  using register_policy = cuda_warp_register;
  using self_type       = Register<ELEMENT_TYPE, cuda_warp_register>;
  using element_type    = ELEMENT_TYPE;
  using register_type   = ELEMENT_TYPE;

  using int_vector_type = Register<int64_t, cuda_warp_register>;


private:
  element_type m_value;

public:
  static constexpr int s_num_elem = RAJA_CUDA_WARPSIZE;

  /*!
   * @brief Default constructor, zeros register contents
   */
  RAJA_INLINE

  RAJA_DEVICE
  constexpr Register() : base_type(), m_value(0) {}

  /*!
   * @brief Copy constructor from raw value
   */
  RAJA_INLINE

  RAJA_DEVICE
  constexpr Register(element_type c) : base_type(), m_value(c) {}

  /*!
   * @brief Copy constructor
   */
  RAJA_INLINE

  RAJA_DEVICE
  constexpr Register(self_type const& c) : base_type(), m_value(c.m_value) {}

  /*!
   * @brief Copy assignment operator
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type& operator=(self_type const& c)
  {
    m_value = c.m_value;
    return *this;
  }

  RAJA_INLINE

  RAJA_DEVICE
  self_type& operator=(element_type c)
  {
    m_value = c;
    return *this;
  }

  /*!
   * @brief Gets our warp lane
   */
  RAJA_INLINE

  RAJA_DEVICE
  constexpr static int get_lane() { return threadIdx.x; }

  RAJA_DEVICE

  RAJA_INLINE
  constexpr element_type const& get_raw_value() const { return m_value; }

  RAJA_DEVICE

  RAJA_INLINE
  element_type& get_raw_value() { return m_value; }

  RAJA_DEVICE

  RAJA_INLINE
  static constexpr bool is_root() { return get_lane() == 0; }

  /*!
   * @brief Load a full register from a stride-one memory location
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type& load_packed(element_type const* ptr)
  {

    auto lane = get_lane();

    m_value = ptr[lane];

    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type& load_packed_n(element_type const* ptr, int N)
  {
    auto lane = get_lane();
    if (lane < N)
    {
      m_value = ptr[lane];
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
  RAJA_INLINE

  RAJA_DEVICE
  self_type& load_strided(element_type const* ptr, int stride)
  {

    auto lane = get_lane();

    m_value = ptr[stride * lane];

    return *this;
  }

  /*!
   * @brief Partially load a register from a stride-one memory location given
   *        a run-time number of elements.
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type& load_strided_n(element_type const* ptr, int stride, int N)
  {
    auto lane = get_lane();

    if (lane < N)
    {
      m_value = ptr[stride * lane];
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

  RAJA_DEVICE
  self_type& gather(element_type const* ptr, int_vector_type offsets)
  {

    m_value = ptr[offsets.get_raw_value()];

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

  RAJA_DEVICE
  self_type& gather_n(element_type const* ptr,
                      int_vector_type offsets,
                      camp::idx_t N)
  {
    if (get_lane() < N)
    {
      m_value = ptr[offsets.get_raw_value()];
    }
    else
    {
      m_value = element_type(0);
    }

    return *this;
  }

  /*!
   * @brief Generic segmented load operation used for loading sub-matrices
   * from larger arrays.
   *
   * The default operation combines the s_segmented_offsets and gather
   * operations.
   *
   *
   */
  RAJA_DEVICE

  RAJA_INLINE
  self_type& segmented_load(element_type const* ptr,
                            camp::idx_t segbits,
                            camp::idx_t stride_inner,
                            camp::idx_t stride_outer)
  {
    auto lane = get_lane();

    // compute segment and segment_size
    auto seg = lane >> segbits;
    auto i   = lane & ((1 << segbits) - 1);

    m_value = ptr[seg * stride_outer + i * stride_inner];

    return *this;
  }

  /*!
   * @brief Generic segmented load operation used for loading sub-matrices
   * from larger arrays where we load partial segments.
   *
   *
   *
   */
  RAJA_DEVICE

  RAJA_INLINE
  self_type& segmented_load_nm(element_type const* ptr,
                               camp::idx_t segbits,
                               camp::idx_t stride_inner,
                               camp::idx_t stride_outer,
                               camp::idx_t num_inner,
                               camp::idx_t num_outer)
  {
    auto lane = get_lane();

    // compute segment and segment_size
    auto seg = lane >> segbits;
    auto i   = lane & ((1 << segbits) - 1);

    if (seg >= num_outer || i >= num_inner)
    {
      m_value = element_type(0);
    }
    else
    {
      m_value = ptr[seg * stride_outer + i * stride_inner];
    }

    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type const& store_packed(element_type* ptr) const
  {

    auto lane = get_lane();

    ptr[lane] = m_value;

    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type const& store_packed_n(element_type* ptr, int N) const
  {

    auto lane = get_lane();

    if (lane < N)
    {
      ptr[lane] = m_value;
    }
    return *this;
  }

  /*!
   * @brief Store entire register to consecutive memory locations
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type const& store_strided(element_type* ptr, int stride) const
  {

    auto lane = get_lane();

    ptr[lane * stride] = m_value;

    return *this;
  }

  /*!
   * @brief Store partial register to consecutive memory locations
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type const& store_strided_n(element_type* ptr, int stride, int N) const
  {

    auto lane = get_lane();

    if (lane < N)
    {
      ptr[lane * stride] = m_value;
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
  template<typename T2>
  RAJA_DEVICE RAJA_INLINE self_type const& scatter(element_type* ptr,
                                                   T2 const& offsets) const
  {

    ptr[offsets.get_raw_value()] = m_value;

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
  template<typename T2>
  RAJA_DEVICE RAJA_INLINE self_type const& scatter_n(element_type* ptr,
                                                     T2 const& offsets,
                                                     camp::idx_t N) const
  {
    if (get_lane() < N)
    {
      ptr[offsets.get_raw_value()] = m_value;
    }

    return *this;
  }

  /*!
   * @brief Generic segmented store operation used for storing sub-matrices
   * to larger arrays.
   *
   */
  RAJA_DEVICE

  RAJA_INLINE
  self_type const& segmented_store(element_type* ptr,
                                   camp::idx_t segbits,
                                   camp::idx_t stride_inner,
                                   camp::idx_t stride_outer) const
  {
    auto lane = get_lane();

    // compute segment and segment_size
    auto seg = lane >> segbits;
    auto i   = lane & ((1 << segbits) - 1);

    ptr[seg * stride_outer + i * stride_inner] = m_value;

    return *this;
  }

  /*!
   * @brief Generic segmented store operation used for storing sub-matrices
   * to larger arrays where we store partial segments.
   *
   */
  RAJA_DEVICE

  RAJA_INLINE
  self_type const& segmented_store_nm(element_type* ptr,
                                      camp::idx_t segbits,
                                      camp::idx_t stride_inner,
                                      camp::idx_t stride_outer,
                                      camp::idx_t num_inner,
                                      camp::idx_t num_outer) const
  {
    auto lane = get_lane();

    // compute segment and segment_size
    auto seg = lane >> segbits;
    auto i   = lane & ((1 << segbits) - 1);

    if (seg >= num_outer || i >= num_inner)
    {
      // nop
    }
    else
    {
      ptr[seg * stride_outer + i * stride_inner] = m_value;
    }

    return *this;
  }

  /*!
   * @brief Get scalar value from vector register
   * @param i Offset of scalar to get
   * @return Returns scalar value at i
   */
  constexpr RAJA_INLINE RAJA_DEVICE element_type get(int i) const
  {
    return __shfl_sync(0xffffffff, m_value, i);
  }

  /*!
   * @brief Set scalar value in vector register
   * @param i Offset of scalar to set
   * @param value Value of scalar to set
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type& set(element_type value, int i)
  {
    auto lane = get_lane();
    if (lane == i)
    {
      m_value = value;
    }
    return *this;
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type& broadcast(element_type const& a)
  {
    m_value = a;
    return *this;
  }

  /*!
   * @brief Extracts a scalar value and broadcasts to a new register
   */
  RAJA_DEVICE

  RAJA_INLINE
  self_type get_and_broadcast(int i) const
  {
    self_type x;
    x.m_value = __shfl_sync(0xffffffff, m_value, i);
    return x;
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type& copy(self_type const& src)
  {
    m_value = src.m_value;
    return *this;
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type add(self_type const& b) const
  {
    return self_type(m_value + b.m_value);
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type subtract(self_type const& b) const
  {
    return self_type(m_value - b.m_value);
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type multiply(self_type const& b) const
  {
    return self_type(m_value * b.m_value);
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type divide(self_type const& b) const
  {
    return self_type(m_value / b.m_value);
  }

  RAJA_DEVICE

  RAJA_INLINE
  self_type divide_n(self_type const& b, int N) const
  {
    return get_lane() < N ? self_type(m_value / b.m_value)
                          : self_type(element_type(0));
  }

  /**
   * floats and doubles use the CUDA instrinsic FMA
   */
  template<typename RETURN_TYPE = self_type>
  RAJA_DEVICE RAJA_INLINE
      typename std::enable_if<!std::numeric_limits<element_type>::is_integer,
                              RETURN_TYPE>::type
      multiply_add(self_type const& b, self_type const& c) const
  {
    return self_type(fma(m_value, b.m_value, c.m_value));
  }

  /**
   * int32 and int64 don't have a CUDA intrinsic FMA, do unfused ops
   */
  template<typename RETURN_TYPE = self_type>
  RAJA_DEVICE RAJA_INLINE
      typename std::enable_if<std::numeric_limits<element_type>::is_integer,
                              RETURN_TYPE>::type
      multiply_add(self_type const& b, self_type const& c) const
  {
    return self_type(m_value * b.m_value + c.m_value);
  }

  /**
   * floats and doubles use the CUDA instrinsic FMS
   */
  template<typename RETURN_TYPE = self_type>
  RAJA_DEVICE RAJA_INLINE
      typename std::enable_if<!std::numeric_limits<element_type>::is_integer,
                              RETURN_TYPE>::type
      multiply_subtract(self_type const& b, self_type const& c) const
  {
    return self_type(fma(m_value, b.m_value, -c.m_value));
  }

  /**
   * int32 and int64 don't have a CUDA intrinsic FMS, do unfused ops
   */
  template<typename RETURN_TYPE = self_type>
  RAJA_DEVICE RAJA_INLINE
      typename std::enable_if<std::numeric_limits<element_type>::is_integer,
                              RETURN_TYPE>::type
      multiply_subtract(self_type const& b, self_type const& c) const
  {
    return self_type(m_value * b.m_value - c.m_value);
  }

  /*!
   * @brief Sum the elements of this vector
   * @return Sum of the values of the vectors scalar elements
   */
  RAJA_INLINE

  RAJA_DEVICE
  element_type sum() const
  {
    // Allreduce sum
    using combiner_t =
        RAJA::reduce::detail::op_adapter<element_type, RAJA::operators::plus>;

    return RAJA::cuda::impl::warp_allreduce<combiner_t, element_type>(m_value);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE

  RAJA_DEVICE
  element_type max() const
  {
    // Allreduce maximum
    using combiner_t =
        RAJA::reduce::detail::op_adapter<element_type,
                                         RAJA::operators::maximum>;

    return RAJA::cuda::impl::warp_allreduce<combiner_t, element_type>(m_value);
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE

  RAJA_DEVICE
  element_type max_n(int N) const
  {
    // Allreduce maximum
    using combiner_t =
        RAJA::reduce::detail::op_adapter<element_type,
                                         RAJA::operators::maximum>;

    auto ident = RAJA::operators::limits<element_type>::min();
    auto lane  = get_lane();
    auto value = lane < N ? m_value : ident;
    return RAJA::cuda::impl::warp_allreduce<combiner_t, element_type>(value);
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type vmax(self_type a) const
  {
    return self_type {RAJA::max<element_type>(m_value, a.m_value)};
  }

  /*!
   * @brief Returns the largest element
   * @return The largest scalar element in the register
   */
  RAJA_INLINE

  RAJA_DEVICE
  element_type min() const
  {
    // Allreduce minimum
    using combiner_t =
        RAJA::reduce::detail::op_adapter<element_type,
                                         RAJA::operators::minimum>;

    return RAJA::cuda::impl::warp_allreduce<combiner_t, element_type>(m_value);
  }

  /*!
   * @brief Returns the largest element from first N lanes
   * @return The largest scalar element in the register
   */
  RAJA_INLINE

  RAJA_DEVICE
  element_type min_n(int N) const
  {
    // Allreduce minimum
    using combiner_t =
        RAJA::reduce::detail::op_adapter<element_type,
                                         RAJA::operators::minimum>;

    auto ident = RAJA::operators::limits<element_type>::max();
    auto lane  = get_lane();
    auto value = lane < N ? m_value : ident;
    return RAJA::cuda::impl::warp_allreduce<combiner_t, element_type>(value);
  }

  /*!
   * @brief Returns element-wise largest values
   * @return Vector of the element-wise max values
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type vmin(self_type a) const
  {
    return self_type {RAJA::min<element_type>(m_value, a.m_value)};
  }

  /*!
   * Provides gather/scatter indices for segmented loads and stores
   *
   * THe number of segment bits (segbits) is specified, as well as the
   * stride between elements in a segment (stride_inner),
   * and the stride between segments (stride_outer)
   */
  RAJA_INLINE

  RAJA_DEVICE
  static int_vector_type s_segmented_offsets(camp::idx_t segbits,
                                             camp::idx_t stride_inner,
                                             camp::idx_t stride_outer)
  {
    int_vector_type result;

    auto lane = get_lane();

    // compute segment and segment_size
    auto seg = lane >> segbits;
    auto i   = lane & ((1 << segbits) - 1);

    result.get_raw_value() = seg * stride_outer + i * stride_inner;

    return result;
  }

  /*!
   * Sum elements within each segment, with segment size defined by segbits.
   * Stores each segments sum consecutively, but shifed to the
   * corresponding output_segment slot.
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 is equivalent to the input vector,  since there are 8
   *      outputs, there is only 1 output segment
   *
   *      Result= x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=1 sums neighboring pairs of values.  There are 4 output,
   *      so there are possible output segments.
   *
   *      output_segment=0:
   *      Result= x0+x1, x2+x3, x4+x5, x6+x7, 0, 0, 0, 0
   *
   *      output_segment=1:
   *      Result= 0, 0, 0, 0, x0+x1, x2+x3, x4+x5, x6+x7
   *
   *  and so on up to segbits=3, which is a full sum of x0..x7, and the
   *      output_segment denotes the vector position of the sum
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type segmented_sum_inner(camp::idx_t segbits,
                                camp::idx_t output_segment) const
  {

    // First: tree reduce values within each segment
    element_type x = m_value;
    RAJA_UNROLL
    for (int delta = 1; delta < 1 << segbits; delta = delta << 1)
    {

      // tree shuffle
      element_type y = __shfl_sync(0xffffffff, x, get_lane() + delta);

      // reduce
      x += y;
    }

    // Second: send result to output segment lanes
    self_type result;
    result.get_raw_value() = __shfl_sync(0xffffffff, x, get_lane() << segbits);

    // Third: mask off everything but output_segment
    //        this is because all output segments are valid at this point
    static constexpr int log2_warp_size = RAJA::log2(RAJA_CUDA_WARPSIZE);
    int our_output_segment = get_lane() >> (log2_warp_size - segbits);
    bool in_output_segment = our_output_segment == output_segment;
    if (!in_output_segment)
    {
      result.get_raw_value() = 0;
    }

    return result;
  }

  /*!
   * Sum across segments, with segment size defined by segbits
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 is equivalent to the input vector,  since there are 8
   *      outputs, there is only 1 output segment
   *
   *      Result= x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=1 sums strided pairs of values.  There are 4 output,
   *      so there are possible output segments.
   *
   *      output_segment=0:
   *      Result= x0+x4, x1+x5, x2+x6, x3+x7, 0, 0, 0, 0
   *
   *      output_segment=1:
   *      Result= 0, 0, 0, 0, x0+x4, x1+x5, x2+x6, x3+x7
   *
   *  and so on up to segbits=3, which is a full sum of x0..x7, and the
   *      output_segment denotes the vector position of the sum
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type segmented_sum_outer(camp::idx_t segbits,
                                camp::idx_t output_segment) const
  {

    // First: tree reduce values within each segment
    element_type x                      = m_value;
    static constexpr int log2_warp_size = RAJA::log2(RAJA_CUDA_WARPSIZE);
    RAJA_UNROLL
    for (int i = 0; i < log2_warp_size - segbits; ++i)
    {

      // tree shuffle
      int delta      = s_num_elem >> (i + 1);
      element_type y = __shfl_sync(0xffffffff, x, get_lane() + delta);

      // reduce
      x += y;
    }

    // Second: send result to output segment lanes
    self_type result;
    int get_from           = get_lane() & ((1 << segbits) - 1);
    result.get_raw_value() = __shfl_sync(0xffffffff, x, get_from);

    int mask = (get_lane() >> segbits) == output_segment;


    // Third: mask off everything but output_segment
    if (!mask)
    {
      result.get_raw_value() = 0;
    }

    return result;
  }

  RAJA_INLINE

  RAJA_DEVICE
  self_type segmented_divide_nm(self_type den,
                                camp::idx_t segbits,
                                camp::idx_t num_inner,
                                camp::idx_t num_outer) const
  {
    self_type result;

    auto lane = get_lane();

    // compute segment and segment_size
    auto seg = lane >> segbits;
    auto i   = lane & ((1 << segbits) - 1);

    if (seg >= num_outer || i >= num_inner)
    {
      // nop
    }
    else
    {
      result.get_raw_value() = m_value / den.get_raw_value();
    }

    return result;
  }

  /*!
   * Segmented broadcast copies a segment to all output segments of a vector
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 means the input segment size is 1, so this selects the
   *      value at x[input_segmnet] and broadcasts it to the rest of the
   *      vector
   *
   *      input segments allowed are from 0 to 7, inclusive
   *
   *      input_segment=0
   *      Result= x0, x0, x0, x0, x0, x0, x0, x0
   *
   *      input_segment=5
   *      Result= x5, x5, x5, x5, x5, x5, x5, x5
   *
   *  segbits=1 means that the input segments are each pair of x values:
   *
   *      input segments allowed are from 0 to 3, inclusive
   *
   *      output_segment=0:
   *      Result= x0, x1, x0, x1, x0, x1, x0, x1
   *
   *      output_segment=1:
   *      Result= x2, x3, x2, x3, x2, x3, x2, x3
   *
   *      output_segment=3:
   *      Result= x6, x7, x6, x7, x6, x7, x6, x7
   *
   *  and so on up to segbits=2, the input segments are 4 wide:
   *
   *      input segments allowed are from 0 or 1
   *
   *      output_segment=0:
   *      Result= x0, x1, x2, x3, x0, x1, x2, x3
   *
   *      output_segment=1:
   *      Result= x4, x5, x6, x7, x4, x5, x6, x7
   *
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type segmented_broadcast_inner(camp::idx_t segbits,
                                      camp::idx_t input_segment) const
  {
    self_type result;

    camp::idx_t mask   = (1 << segbits) - 1;
    camp::idx_t offset = input_segment << segbits;


    camp::idx_t i = (get_lane() & mask) + offset;

    result.get_raw_value() = __shfl_sync(0xffffffff, m_value, i);


    return result;
  }

  /*!
   * Segmented broadcast spreads a segment to all output segments of a vector
   *
   * Note: segment size is 1<<segbits elements
   *       number of segments is s_num_elem>>seg_bits
   *
   *
   *  Example:
   *
   *  Given input vector  X = x0, x1, x2, x3, x4, x5, x6, x7
   *
   *  segbits=0 means the input segment size is 1, so this selects the
   *      value at x[input_segmnet] and broadcasts it to the rest of the
   *      vector
   *
   *      input segments allowed are from 0 to 7, inclusive
   *
   *      input_segment=0
   *      Result= x0, x0, x0, x0, x0, x0, x0, x0
   *
   *      input_segment=5
   *      Result= x5, x5, x5, x5, x5, x5, x5, x5
   *
   *  segbits=1 means that the input segments are each pair of x values:
   *
   *      input segments allowed are from 0 to 3, inclusive
   *
   *      output_segment=0:
   *      Result= x0, x0, x0, x0, x1, x1, x1, x1
   *
   *      output_segment=1:
   *      Result= x2, x2, x2, x2, x3, x3, x3, x3
   *
   *      output_segment=3:
   *      Result= x6, x6, x6, x6, x7, x7, x7, x7
   */
  RAJA_INLINE

  RAJA_DEVICE
  self_type segmented_broadcast_outer(camp::idx_t segbits,
                                      camp::idx_t input_segment) const
  {
    self_type result;

    camp::idx_t offset = input_segment * (self_type::s_num_elem >> segbits);

    camp::idx_t i = (get_lane() >> segbits) + offset;

    result.get_raw_value() = __shfl_sync(0xffffffff, m_value, i);

    return result;
  }
};


}  // namespace expt

}  // namespace RAJA


#endif  // CUDA

#endif  // Guard
