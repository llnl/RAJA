//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_BASIC_REDUCESUM_HPP__
#define __TEST_FORALL_BASIC_REDUCESUM_HPP__

#include <cstdlib>
#include <ctime>
#include <numeric>
#include <vector>
#include <random>
#include <type_traits>

template <typename EXEC_POLICY, typename REDUCE_POLICY, typename ABSTRACTION,
          typename DATA_TYPE, typename IDX_TYPE,
          typename SEG_TYPE, typename Container,
          typename RandomGenerator>
// use enable_if in return type to appease nvcc 11.2
// add bool return type to disambiguate signatures of these functions for MSVC
std::enable_if_t<!ABSTRACTION::template supports<DATA_TYPE>(), bool>
ForallMultiReduceBasicTestImpl(const SEG_TYPE&,
                               const Container&,
                               const std::vector<IDX_TYPE>&,
                               camp::resources::Resource,
                               RandomGenerator&)
{ return false; }
///
template <typename EXEC_POLICY, typename REDUCE_POLICY, typename ABSTRACTION,
          typename DATA_TYPE, typename IDX_TYPE,
          typename SEG_TYPE, typename Container,
          typename RandomGenerator>
// use enable_if in return type to appease nvcc 11.2
std::enable_if_t<ABSTRACTION::template supports<DATA_TYPE>()>
ForallMultiReduceBasicTestImpl(const SEG_TYPE& seg,
                               const Container& multi_init,
                               const std::vector<IDX_TYPE>& seg_idx,
                               camp::resources::Resource working_res,
                               RandomGenerator& rngen)
{
  using MULTIREDUCER = typename ABSTRACTION::template multi_reducer<REDUCE_POLICY, DATA_TYPE>;

  const IDX_TYPE idx_range = seg_idx[seg_idx.size() - 1] + 1;
  const IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );

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

    for (IDX_TYPE i = 0; i < idx_len; ++i) {
      IDX_TYPE idx = seg_idx[i];
      test_range[idx] = data_len;
      data_len += work_per_iterate_distribution(rngen);
      test_range[idx+1] = data_len;
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

    RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE ii) {
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

      RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE ii) {
        for (IDX_TYPE idx = working_range[ii]; idx < working_range[ii+1]; ++idx) {
          ABSTRACTION::reduce(red[working_bins[idx]], working_array[idx]);
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

      RAJA::forall<EXEC_POLICY>(seg, [=] RAJA_HOST_DEVICE(IDX_TYPE ii) {
        for (IDX_TYPE idx = working_range[ii]; idx < working_range[ii+1]; ++idx) {
          ABSTRACTION::reduce(red[working_bins[idx]], working_array[idx]);
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


TYPED_TEST_SUITE_P(ForallMultiReduceBasicTest);
template <typename T>
class ForallMultiReduceBasicTest : public ::testing::Test
{
};

TYPED_TEST_P(ForallMultiReduceBasicTest, MultiReduceBasicForall)
{
  using IDX_TYPE      = typename camp::at<TypeParam, camp::num<0>>::type;
  using DATA_TYPE     = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES   = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY   = typename camp::at<TypeParam, camp::num<3>>::type;
  using REDUCE_POLICY = typename camp::at<TypeParam, camp::num<4>>::type;
  using ABSTRACTION   = typename camp::at<TypeParam, camp::num<5>>::type;

  // for setting random values in arrays
  auto random_seed = std::random_device{}();
  std::mt19937 rngen(random_seed);

  camp::resources::Resource working_res{WORKING_RES::get_default()};

  std::vector<IDX_TYPE> seg_idx;

  std::vector<DATA_TYPE> container;

  std::vector<size_t> num_bins_max_container({0, 1, 100});
  size_t num_bins_min = 0;
  for (size_t num_bins_max : num_bins_max_container) {

    std::uniform_int_distribution<size_t> num_bins_dist(num_bins_min, num_bins_max);
    num_bins_min = num_bins_max+1;
    size_t num_bins = num_bins_dist(rngen);

    container.resize(num_bins, DATA_TYPE(2));

    // Range segment tests
    RAJA::TypedRangeSegment<IDX_TYPE> r1( 0, 28 );
    RAJA::getIndices(seg_idx, r1);
    ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                   r1, container, seg_idx, working_res, rngen);

    seg_idx.clear();
    RAJA::TypedRangeSegment<IDX_TYPE> r3( 3, 2060 );
    RAJA::getIndices(seg_idx, r3);
    ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                   r3, container, seg_idx, working_res, rngen);

    // Range-stride segment test
    seg_idx.clear();
    RAJA::TypedRangeStrideSegment<IDX_TYPE> r5( 3, 1029, 3 );
    RAJA::getIndices(seg_idx, r5);
    ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                   r5, container, seg_idx, working_res, rngen);

    // List segment test
    seg_idx.clear();
    IDX_TYPE last = 10567;
    std::uniform_int_distribution<IDX_TYPE> dist(0, last-1);
    for (IDX_TYPE i = 0; i < last; ++i) {
      IDX_TYPE randval = dist(rngen);
      if ( i < randval ) {
        seg_idx.push_back(i);
      }
    }
    RAJA::TypedListSegment<IDX_TYPE> l1( &seg_idx[0], seg_idx.size(),
                                         working_res );
    ForallMultiReduceBasicTestImpl<EXEC_POLICY, REDUCE_POLICY, ABSTRACTION, DATA_TYPE>(
                                   l1, container, seg_idx, working_res, rngen);
  }
}

REGISTER_TYPED_TEST_SUITE_P(ForallMultiReduceBasicTest,
                            MultiReduceBasicForall);

#endif  // __TEST_FORALL_BASIC_REDUCESUM_HPP__
