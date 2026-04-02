//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup unordered runs.
///

#ifndef __TEST_WORKGROUP_UNORDERED_SINGLE__
#define __TEST_WORKGROUP_UNORDERED_SINGLE__

#include "RAJA_test-workgroup.hpp"
#include "RAJA_test-forall-data.hpp"

#include <random>


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename DispatchTyper,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
struct testWorkGroupUnorderedSingle {
void operator()(IndexType begin, IndexType end) const
{

  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  WORKING_RES res = WORKING_RES::get_default();
  camp::resources::Resource working_res{res};

  IndexType* working_array;
  IndexType* check_array;
  IndexType* test_array;

  allocateForallTestData<IndexType>(N,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);

  IndexType const test_val(5);

  using range_segment = RAJA::TypedRangeSegment<IndexType>;

  auto callable = [=] RAJA_HOST_DEVICE (IndexType i) {
        working_array[i] += i + test_val;
      };

  using DispatchPolicy = typename DispatchTyper::template type<
      camp::list<range_segment, decltype(callable)> >;

  using WorkPool_type = RAJA::WorkPool<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkGroup_type = RAJA::WorkGroup<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkSite_type = RAJA::WorkSite<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using resource_type = typename WorkSite_type::resource_type;
  static_assert(std::is_same<WORKING_RES, resource_type>::value,
                "Expected same resource types");

  {
    for (IndexType i = IndexType(0); i < N; i++) {
      test_array[i] = IndexType(0);
    }

    res.memcpy(working_array, test_array, sizeof(IndexType) * N);

    for (IndexType i = begin; i < end; ++i) {
      test_array[ i ] = IndexType(i);
    }
  }

  WorkPool_type pool(Allocator{});

  {
    pool.enqueue(range_segment{ begin, end }, callable);
  }

  WorkGroup_type group = pool.instantiate();

  WorkSite_type site = group.run();

  auto e = site.get_resource().get_event();
  e.wait();

  {
    res.memcpy(check_array, working_array, sizeof(IndexType) * N);

    for (IndexType i = IndexType(0); i < begin; i++) {
      ASSERT_EQ(test_array[i], check_array[i]);
    }
    for (IndexType i = begin;        i < end;   i++) {
      ASSERT_EQ(test_array[i] + test_val, check_array[i]);
    }
    for (IndexType i = end;          i < N;     i++) {
      ASSERT_EQ(test_array[i], check_array[i]);
    }
  }


  deallocateForallTestData<IndexType>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}
};


#if defined(RAJA_ENABLE_HIP) && !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

/// leave unsupported types untested
template <size_t BLOCK_SIZE, bool Async,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
struct testWorkGroupUnorderedSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                    RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                    StoragePolicy,
                                    detail::indirect_function_call_dispatch_typer,
                                    IndexType,
                                    Allocator,
                                    WORKING_RES> {
void operator()(
    IndexType, IndexType) const
{ }
};
///
template <size_t BLOCK_SIZE, bool Async,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
struct testWorkGroupUnorderedSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                    RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                    StoragePolicy,
                                    detail::indirect_virtual_function_dispatch_typer,
                                    IndexType,
                                    Allocator,
                                    WORKING_RES> {
void operator()(
    IndexType, IndexType) const
{ }
};

#endif


template <typename T>
class WorkGroupBasicUnorderedSingleFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedSingleFunctionalTest);


TYPED_TEST_P(WorkGroupBasicUnorderedSingleFunctionalTest, BasicWorkGroupUnorderedSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<3>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<6>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType b1 = dist_type(IndexType(0), IndexType(15))(rng);
  IndexType e1 = dist_type(b1, IndexType(16))(rng);

  IndexType b2 = dist_type(e1, IndexType(127))(rng);
  IndexType e2 = dist_type(b2, IndexType(128))(rng);

  IndexType b3 = dist_type(e2, IndexType(1023))(rng);
  IndexType e3 = dist_type(b3, IndexType(1024))(rng);

  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator, WORKING_RESOURCE >{}(b1, e1);
  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator, WORKING_RESOURCE >{}(b2, e2);
  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator, WORKING_RESOURCE >{}(b3, e3);
}

#endif  //__TEST_WORKGROUP_UNORDERED_SINGLE__
