//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_MULTIREDUCE_HPP__
#define __TEST_KERNEL_NESTED_MULTIREDUCE_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>
#include <random>
#include <type_traits>

template <typename EXEC_POLICY, typename REDUCE_POLICY, typename ABSTRACTION,
          typename DATA_TYPE, typename IDX_TYPE,
          typename SEGMENTS_TYPE, typename Container,
          typename WORKING_RES, typename RandomGenerator>
// use enable_if in return type to appease nvcc 11.2
// add bool return type to disambiguate signatures of these functions for MSVC
std::enable_if_t<!ABSTRACTION::template supports<DATA_TYPE>(), bool>
KernelMultiReduceNestedTestImpl(const SEGMENTS_TYPE&,
                                const Container&,
                                WORKING_RES,
                                RandomGenerator&)
{ return false; }
///
template <typename EXEC_POLICY, typename REDUCE_POLICY, typename ABSTRACTION,
          typename DATA_TYPE, typename IDX_TYPE,
          typename SEGMENTS_TYPE, typename Container,
          typename WORKING_RES, typename RandomGenerator>
// use enable_if in return type to appease nvcc 11.2
std::enable_if_t<ABSTRACTION::template supports<DATA_TYPE>()>
KernelMultiReduceNestedTestImpl(const SEGMENTS_TYPE& segments,
                                const Container& multi_init,
                                WORKING_RES working_res,
                                RandomGenerator& rngen)
{
  using RAJA::get;
  using MULTIREDUCER = typename ABSTRACTION::template multi_reducer<REDUCE_POLICY, DATA_TYPE>;

  auto si = get<2>(segments);
  auto sj = get<1>(segments);
  auto sk = get<0>(segments);

  RAJA_EXTRACT_BED_SUFFIXED(si, _si);
  RAJA_EXTRACT_BED_SUFFIXED(sj, _sj);
  RAJA_EXTRACT_BED_SUFFIXED(sk, _sk);

  IDX_TYPE dimi = begin_si[distance_si-1] + 1;
  IDX_TYPE dimj = begin_sj[distance_sj-1] + 1;
  IDX_TYPE dimk = begin_sk[distance_sk-1] + 1;

  const IDX_TYPE idx_range = dimi * dimj * dimk;

  const int modval = 100;
  const size_t num_bins = multi_init.size();

  IDX_TYPE* working_range;
  IDX_TYPE* check_range;
  IDX_TYPE* test_range;

  DATA_TYPE* working_array;
  DATA_TYPE* check_array;
  DATA_TYPE* test_array;

  IDX_TYPE* working_bins;
  IDX_TYPE* check_bins;
  IDX_TYPE* test_bins;

  IDX_TYPE data_len = 0;

  allocateForallTestData(idx_range+1,
                         working_res,
                         &working_range,
                         &check_range,
                         &test_range);

  for (IDX_TYPE i = 0; i < idx_range+1; ++i) {
    test_range[i] = ~IDX_TYPE(0);
  }

  {
    std::uniform_int_distribution<IDX_TYPE> work_per_iterate_distribution(0, num_bins);

    for (IDX_TYPE k : sk) {
      for (IDX_TYPE j : sj) {
        for (IDX_TYPE i : si) {
          IDX_TYPE ii = (dimi * dimj * k) + (dimi * j) + i;
          test_range[ii] = data_len;
          data_len += work_per_iterate_distribution(rngen);
          test_range[ii+1] = data_len;
        }
      }
    }
  }

  allocateForallTestData(data_len,
                         working_res,
                         &working_array,
                         &check_array,
                         &test_array);

  allocateForallTestData(data_len,
                         working_res,
                         &working_bins,
                         &check_bins,
                         &test_bins);

  if (data_len > IDX_TYPE(0)) {

    // use ints to initialize array here to avoid floating point precision issues
    std::uniform_int_distribution<int> array_int_distribution(0, modval-1);
    std::uniform_int_distribution<IDX_TYPE> bin_distribution(0, num_bins-1);


    for (IDX_TYPE i = 0; i < data_len; ++i) {
      test_array[i] = DATA_TYPE(array_int_distribution(rngen));

      // this may use the same bin multiple times per iterate
      test_bins[i] = bin_distribution(rngen);
    }
  }

  working_res.memcpy(working_range, test_range, sizeof(IDX_TYPE) * (idx_range+1));
  working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);
  working_res.memcpy(working_bins, test_bins, sizeof(IDX_TYPE) * data_len);


  MULTIREDUCER red(num_bins);
  MULTIREDUCER red2(multi_init);

  // basic test with two multi reducers in the same loop
  {
    std::vector<DATA_TYPE> ref_vals(num_bins, ABSTRACTION::identity(red));

    for (IDX_TYPE i = 0; i < data_len; ++i) {
      ref_vals[test_bins[i]] = ABSTRACTION::combine(ref_vals[test_bins[i]], test_array[i]);
    }

    RAJA::kernel_resource<EXEC_POLICY>(segments, working_res,
        [=] RAJA_HOST_DEVICE (IDX_TYPE k, IDX_TYPE j, IDX_TYPE i) {
      IDX_TYPE ii = (dimi * dimj * k) + (dimi * j) + i;
      for (IDX_TYPE idx = working_range[ii]; idx < working_range[ii+1]; ++idx) {
        ABSTRACTION::reduce(red[working_bins[idx]],  working_array[idx]);
        ABSTRACTION::reduce(red2[working_bins[idx]], working_array[idx]);
      }
    });

    size_t bin = 0;
    for (auto init_val : multi_init) {
      ASSERT_EQ(DATA_TYPE(red[bin].get()), ref_vals[bin]);
      ASSERT_EQ(red2.get(bin), ABSTRACTION::combine(ref_vals[bin], init_val));
      ++bin;
    }
  }


  red.reset();

  // basic multiple use test, ensure same reducer can combine values from multiple loops
  {
    std::vector<DATA_TYPE> ref_vals(num_bins, ABSTRACTION::identity(red));

    const int nloops = 2;
    for (int j = 0; j < nloops; ++j) {

      for (IDX_TYPE i = 0; i < data_len; ++i) {
        ref_vals[test_bins[i]] = ABSTRACTION::combine(ref_vals[test_bins[i]], test_array[i]);
      }

      RAJA::kernel_resource<EXEC_POLICY>(segments, working_res,
          [=] RAJA_HOST_DEVICE (IDX_TYPE K, IDX_TYPE J, IDX_TYPE I) {
        IDX_TYPE ii = (dimi * dimj * K) + (dimi * J) + I;
        for (IDX_TYPE idx = working_range[ii]; idx < working_range[ii+1]; ++idx) {
          ABSTRACTION::reduce(red[working_bins[idx]],  working_array[idx]);
        }
      });
    }

    for (size_t bin = 0; bin < num_bins; ++bin) {
      ASSERT_EQ(static_cast<DATA_TYPE>(red[bin].get()), ref_vals[bin]);
    }
  }


  // test the consistency of answers, if we expect them to be consistent
  if (ABSTRACTION::consistent(red)) {

    if /* constexpr */ (std::is_floating_point<DATA_TYPE>::value) {

      // use floating point values to accentuate floating point precision issues
      std::conditional_t<!std::is_floating_point<DATA_TYPE>::value,
          std::uniform_int_distribution<DATA_TYPE>,
          std::uniform_real_distribution<DATA_TYPE>> array_flt_distribution(0, modval-1);

      for (IDX_TYPE i = 0; i < data_len; ++i) {
        test_array[i] = DATA_TYPE(array_flt_distribution(rngen));
      }
      working_res.memcpy(working_array, test_array, sizeof(DATA_TYPE) * data_len);
    }


    std::vector<DATA_TYPE> ref_vals;
    bool got_ref_vals = false;

    const int nloops = 2;
    for (int j = 0; j < nloops; ++j) {
      red.reset();

      RAJA::kernel_resource<EXEC_POLICY>(segments, working_res,
          [=] RAJA_HOST_DEVICE (IDX_TYPE K, IDX_TYPE J, IDX_TYPE I) {
        IDX_TYPE ii = (dimi * dimj * K) + (dimi * J) + I;
        for (IDX_TYPE idx = working_range[ii]; idx < working_range[ii+1]; ++idx) {
          ABSTRACTION::reduce(red[working_bins[idx]],  working_array[idx]);
        }
      });

      if (!got_ref_vals) {
        ref_vals.resize(num_bins);
        red.get_all(ref_vals);
        got_ref_vals = true;
      } else {
        for (size_t bin = 0; bin < num_bins; ++bin) {
          ASSERT_EQ(red.get(bin), ref_vals[bin]);
        }
      }
    }
  }


  deallocateForallTestData(working_res,
                           working_bins,
                           check_bins,
                           test_bins);
  deallocateForallTestData(working_res,
                           working_array,
                           check_array,
                           test_array);
  deallocateForallTestData(working_res,
                           working_range,
                           check_range,
                           test_range);
}


TYPED_TEST_SUITE_P(KernelMultiReduceNestedTest);
template <typename T>
class KernelMultiReduceNestedTest : public ::testing::Test
{
};

//
//
// Defining the Kernel Loop structure for MultiReduce Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename POLICY_DATA>
struct MultiReduceNestedLoopExec;

template<typename POLICY_DATA>
struct MultiReduceNestedLoopExec<DEPTH_3, POLICY_DATA> {
  using type =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
};

template<typename POLICY_DATA>
struct MultiReduceNestedLoopExec<DEPTH_3_COLLAPSE, POLICY_DATA> {
  using type =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<0,1,2>,
        RAJA::statement::Lambda<0>
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP) or defined(RAJA_ENABLE_SYCL)

template<typename POLICY_DATA>
struct MultiReduceNestedLoopExec<DEVICE_DEPTH_3, POLICY_DATA> {
  using type =
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::For<2, typename camp::at<POLICY_DATA, camp::num<2>>::type,
              RAJA::statement::Lambda<0>
            >
          >
        >
      > // end DEVICE_KERNEL
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP or RAJA_ENABLE_SYCL

TYPED_TEST_P(KernelMultiReduceNestedTest, MultiReduceNestedKernel)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using ABSTRACTION   = typename camp::at<TypeParam, camp::num<5>>::type;

  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;
  using LOOP_POLS = typename EXEC_POL_DATA::type;
  using EXEC_POLICY = typename MultiReduceNestedLoopExec<LOOP_TYPE, LOOP_POLS>::type;

  // for setting random values in arrays
  auto random_seed = std::random_device{}();
  std::mt19937 rngen(random_seed);

  WORKING_RES working_res{WORKING_RES::get_default()};

  std::vector<DATA_TYPE> container;

  std::vector<size_t> num_bins_max_container({0, 1, 100});
  size_t num_bins_min = 0;
  for (size_t num_bins_max : num_bins_max_container) {

    std::uniform_int_distribution<size_t> num_bins_dist(num_bins_min, num_bins_max);
    num_bins_min = num_bins_max+1;
    size_t num_bins = num_bins_dist(rngen);

    container.resize(num_bins, DATA_TYPE(2));

    // Range segment tests
    auto s1 = RAJA::make_tuple(RAJA::TypedRangeSegment<IDX_TYPE>( 0, 2 ),
                               RAJA::TypedRangeSegment<IDX_TYPE>( 0, 7 ),
                               RAJA::TypedRangeSegment<IDX_TYPE>( 0, 3 ));
    KernelMultiReduceNestedTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE, IDX_TYPE>(
                                   s1, container, working_res, rngen);

    auto s2 = RAJA::make_tuple(RAJA::TypedRangeSegment<IDX_TYPE>( 2, 35 ),
                               RAJA::TypedRangeSegment<IDX_TYPE>( 0, 19 ),
                               RAJA::TypedRangeSegment<IDX_TYPE>( 3, 13 ));
    KernelMultiReduceNestedTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE, IDX_TYPE>(
                                   s2, container, working_res, rngen);

    // Range-stride segment tests
    auto s3 = RAJA::make_tuple(RAJA::TypedRangeStrideSegment<IDX_TYPE>( 0, 6, 2 ),
                               RAJA::TypedRangeStrideSegment<IDX_TYPE>( 1, 38, 3 ),
                               RAJA::TypedRangeStrideSegment<IDX_TYPE>( 5, 17, 1 ));
    KernelMultiReduceNestedTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE, IDX_TYPE>(
                                   s3, container, working_res, rngen);

  }
}

REGISTER_TYPED_TEST_SUITE_P(KernelMultiReduceNestedTest,
                            MultiReduceNestedKernel);

#endif  // __TEST_KERNEL_NESTED_MULTIREDUCE_HPP__
