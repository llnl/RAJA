//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

namespace
{

using GenericMemPool =
    RAJA::basic_mempool::MemPool<RAJA::basic_mempool::generic_allocator>;

constexpr size_t test_arena_size = 128;

}  // namespace

TEST(BasicMemPool, release_unused_empty_pool)
{
  GenericMemPool pool;

  ASSERT_EQ(pool.release_unused(), size_t {0});
}

TEST(BasicMemPool, release_unused_after_all_allocations_are_freed)
{
  GenericMemPool pool;
  pool.arena_size(test_arena_size);

  char* ptr = pool.malloc<char>(1);
  ASSERT_NE(ptr, nullptr);
  pool.free(ptr);

  ASSERT_GE(pool.release_unused(), test_arena_size);
  ASSERT_EQ(pool.release_unused(), size_t {0});

  ptr = pool.malloc<char>(1);
  ASSERT_NE(ptr, nullptr);
  pool.free(ptr);

  ASSERT_GE(pool.release_unused(), test_arena_size);
  ASSERT_EQ(pool.release_unused(), size_t {0});
}

TEST(BasicMemPool, release_unused_keeps_arena_with_live_allocation)
{
  GenericMemPool pool;
  pool.arena_size(test_arena_size);

  char* first = pool.malloc<char>(32);
  ASSERT_NE(first, nullptr);

  char* second = pool.malloc<char>(32);
  ASSERT_NE(second, nullptr);
  pool.free(second);

  ASSERT_EQ(pool.release_unused(), size_t {0});

  pool.free(first);

  ASSERT_GE(pool.release_unused(), test_arena_size);
}

TEST(BasicMemPool, release_unused_internal_memory_public_api)
{
  const size_t released = RAJA::release_unused_internal_memory();
  ASSERT_EQ(released, size_t {0});

#if defined(RAJA_CUDA_ACTIVE)
  ASSERT_EQ(RAJA::cuda::release_unused_internal_memory(), size_t {0});
#endif
#if defined(RAJA_HIP_ACTIVE)
  ASSERT_EQ(RAJA::hip::release_unused_internal_memory(), size_t {0});
#endif
#if defined(RAJA_SYCL_ACTIVE)
  ASSERT_EQ(RAJA::sycl::release_unused_internal_memory(), size_t {0});
#endif
}

#if defined(RAJA_CUDA_ACTIVE)
namespace
{
struct SetCudaPoolValues
{
  int* values;
  int iteration;

  RAJA_DEVICE void operator()(RAJA::Index_type i) const
  {
    values[i] = static_cast<int>(i) + iteration;
  }
};
}  // namespace

TEST(BasicMemPool, cuda_reduction_pool_host_access_between_async_launches)
{
  RAJA::cuda::host_accessible_reduction_mempool_type pool;
  constexpr int count = 65536;
  pool.arena_size(4 * count * sizeof(int));
  int* values = pool.malloc<int>(count);

  // Host bookkeeping and a disjoint allocation must remain accessible while
  // the GPU uses another allocation from the same reduction pool.
  for (int iteration = 0; iteration < 3; ++iteration)
  {
    RAJA::forall<RAJA::cuda_exec_async<256>>(
        RAJA::RangeSegment(0, count), SetCudaPoolValues {values, iteration});
    int* host_value = pool.malloc<int>(1);
    *host_value     = iteration;
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(*host_value, iteration);
    EXPECT_EQ(values[0], iteration);
    EXPECT_EQ(values[count - 1], count - 1 + iteration);
    pool.free(host_value);
  }

  pool.free(values);
  EXPECT_GE(pool.release_unused(), 4 * count * sizeof(int));
  EXPECT_EQ(pool.release_unused(), size_t {0});
}

TEST(BasicMemPool, cuda_reduction_pool_public_release)
{
  auto& pool =
      RAJA::cuda::host_accessible_reduction_mempool_type::getInstance();
  int* value = pool.malloc<int>(1);
  *value     = 42;
  EXPECT_EQ(*value, 42);
  pool.free(value);

  EXPECT_GT(RAJA::release_unused_internal_memory(), size_t {0});
  EXPECT_EQ(pool.release_unused(), size_t {0});
}
#endif
