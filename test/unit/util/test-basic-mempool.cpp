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

  char* first  = pool.malloc<char>(32);
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
