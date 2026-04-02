//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing test infrastructure for sort tests
///

#ifndef __TEST_ALGORITHM_SORT_UTILS_HPP__
#define __TEST_ALGORITHM_SORT_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-forall-data.hpp"
#include "type_helper.hpp"
#include "RAJA_unit-test-forone.hpp"

#include <string>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <algorithm>
#include <chrono>
#include <random>


// tag classes to differentiate sort by attributes and apply correct testing
struct unstable_sort_tag { };
struct stable_sort_tag { };

struct sort_interface_tag { };
struct sort_pairs_interface_tag { };

struct sort_default_interface_tag { };
struct sort_comp_interface_tag { };
struct sort_res_default_interface_tag { };
struct sort_res_comp_interface_tag { };


// synchronize based on a RAJA execution policy
template < typename policy >
struct PolicySynchronize
{
  void synchronize()
  {
    // no synchronization needed
  }
};

#if defined(RAJA_ENABLE_CUDA)
// partial specialization for cuda_exec
template < size_t BLOCK_SIZE, bool Async >
struct PolicySynchronize<RAJA::cuda_exec<BLOCK_SIZE, Async>>
{
  void synchronize()
  {
    if (Async) { RAJA::synchronize<RAJA::cuda_synchronize>(); }
  }
};
#endif

#if defined(RAJA_ENABLE_HIP)
// partial specialization for hip_exec
template < size_t BLOCK_SIZE, bool Async >
struct PolicySynchronize<RAJA::hip_exec<BLOCK_SIZE, Async>>
{
  void synchronize()
  {
    if (Async) { RAJA::synchronize<RAJA::hip_synchronize>(); }
  }
};
#endif


template <typename Res,
          typename pairs_category,
          typename K,
          typename V = RAJA::Index_type>
struct SortData;

template <typename Res, typename K, typename V>
struct SortData<Res, sort_interface_tag, K, V>
{
  K* orig_keys = nullptr;
  K* sorted_keys = nullptr;
  Res m_res;

  template < typename RandomGenerator >
  SortData(size_t N, Res res, RandomGenerator gen_random)
    : m_res(res)
  {
    if (N > 0) {
      orig_keys = m_res.template allocate<K>(N, camp::resources::MemoryAccess::Managed);
      sorted_keys = m_res.template allocate<K>(N, camp::resources::MemoryAccess::Managed);
    }

    for (size_t i = 0; i < N; i++) {
      orig_keys[i] = gen_random();
    }
  }

  void copy_data(size_t N)
  {
    if ( N == 0 ) return;
    m_res.memcpy(sorted_keys, orig_keys, N*sizeof(K));
  }

  Res resource()
  {
    return m_res;
  }

  SortData(SortData const&) = delete;
  SortData& operator=(SortData const&) = delete;

  ~SortData()
  {
    if (orig_keys != nullptr) {
      m_res.deallocate(orig_keys, camp::resources::MemoryAccess::Managed);
      m_res.deallocate(sorted_keys, camp::resources::MemoryAccess::Managed);
    }
  }
};


template <typename Res, typename K, typename V>
struct SortData<Res, sort_pairs_interface_tag, K, V> : SortData<Res, sort_interface_tag, K, V>
{
  using base = SortData<Res, sort_interface_tag, K, V>;

  V* orig_vals = nullptr;
  V* sorted_vals = nullptr;

  template < typename RandomGenerator >
  SortData(size_t N, Res res, RandomGenerator gen_random)
    : base(N, res, gen_random)
  {
    if (N > 0) {
      orig_vals = this->m_res.template allocate<V>(N, camp::resources::MemoryAccess::Managed);
      sorted_vals = this->m_res.template allocate<V>(N, camp::resources::MemoryAccess::Managed);
    }

    for (size_t i = 0; i < N; i++) {
      orig_vals[i] = gen_random();
    }
  }

  void copy_data(size_t N)
  {
    base::copy_data(N);
    if ( N == 0 ) return;
    this->m_res.memcpy(sorted_vals, orig_vals, N*sizeof(V));
  }

  SortData(SortData const&) = delete;
  SortData& operator=(SortData const&) = delete;

  ~SortData()
  {
    if (orig_vals != nullptr) {
      this->m_res.deallocate(orig_vals, camp::resources::MemoryAccess::Managed);
      this->m_res.deallocate(sorted_vals, camp::resources::MemoryAccess::Managed);
    }
  }
};


template <typename Res,
          typename T,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_interface_tag, T> & data,
            RAJA::Index_type N,
            Compare,
            Sorter sorter, sort_interface_tag, sort_default_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  sorter(RAJA::make_span(data.sorted_keys, N));
  sorter.synchronize();
}

template <typename Res,
          typename T,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_interface_tag, T> & data,
            RAJA::Index_type N,
            Compare comp,
            Sorter sorter, sort_interface_tag, sort_comp_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  sorter(RAJA::make_span(data.sorted_keys, N),
         comp);
  sorter.synchronize();
}

template <typename Res,
          typename T,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_interface_tag, T> & data,
            RAJA::Index_type N,
            Compare,
            Sorter sorter, sort_interface_tag, sort_res_default_interface_tag)
{
  data.copy_data(N);
  sorter(data.resource(),
         RAJA::make_span(data.sorted_keys, N));
  data.resource().wait();
}

template <typename Res,
          typename T,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_interface_tag, T> & data,
            RAJA::Index_type N,
            Compare comp,
            Sorter sorter, sort_interface_tag, sort_res_comp_interface_tag)
{
  data.copy_data(N);
  sorter(data.resource(),
         RAJA::make_span(data.sorted_keys, N),
         comp);
  data.resource().wait();
}

template <typename Res,
          typename K,
          typename V,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_pairs_interface_tag, K, V> & data,
            RAJA::Index_type N,
            Compare,
            Sorter sorter, sort_pairs_interface_tag, sort_default_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  sorter(RAJA::make_span(data.sorted_keys, N),
         RAJA::make_span(data.sorted_vals, N));
  sorter.synchronize();
}

template <typename Res,
          typename K,
          typename V,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_pairs_interface_tag, K, V> & data,
            RAJA::Index_type N,
            Compare comp,
            Sorter sorter, sort_pairs_interface_tag, sort_comp_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  sorter(RAJA::make_span(data.sorted_keys, N),
         RAJA::make_span(data.sorted_vals, N),
         comp);
  sorter.synchronize();
}

template <typename Res,
          typename K,
          typename V,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_pairs_interface_tag, K, V> & data,
            RAJA::Index_type N,
            Compare,
            Sorter sorter, sort_pairs_interface_tag, sort_res_default_interface_tag)
{
  data.copy_data(N);
  sorter(data.resource(),
         RAJA::make_span(data.sorted_keys, N),
         RAJA::make_span(data.sorted_vals, N));
  data.resource().wait();
}

template <typename Res,
          typename K,
          typename V,
          typename Compare,
          typename Sorter>
void doSort(SortData<Res, sort_pairs_interface_tag, K, V> & data,
            RAJA::Index_type N,
            Compare comp,
            Sorter sorter, sort_pairs_interface_tag, sort_res_comp_interface_tag)
{
  data.copy_data(N);
  sorter(data.resource(),
         RAJA::make_span(data.sorted_keys, N),
         RAJA::make_span(data.sorted_vals, N),
         comp);
  data.resource().wait();
}


template <typename Res,
          typename T,
          typename Compare,
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<Res, sort_interface_tag, T> & data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, unstable_sort_tag, sort_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to keys
  using val_map = std::unordered_multiset<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(data.orig_keys[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(data.orig_keys[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << data.sorted_keys[i-1] << ", " << data.sorted_keys[i]
             << " (at index " << i-1 << ")";
    // test there is an item with this
    auto key_iter = keys.find(data.sorted_keys[i]);
    if (key_iter == keys.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << data.sorted_keys[i]
             << " (at index " << i << ")";
    auto val_iter = key_iter->second.find(data.sorted_keys[i]);
    if (val_iter == key_iter->second.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate val "
             << data.sorted_keys[i]
             << " (at index " << i << ")";
    key_iter->second.erase(val_iter);
    if (key_iter->second.size() == 0) {
      keys.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename Res,
          typename T,
          typename Compare,
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<Res, sort_interface_tag, T> & data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, stable_sort_tag, sort_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to keys
  using val_map = std::list<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(data.orig_keys[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(data.orig_keys[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << data.sorted_keys[i-1] << ", " << data.sorted_keys[i]
             << " (at index " << i-1 << ")";
    // test there is an item with this
    auto key_iter = keys.find(data.sorted_keys[i]);
    if (key_iter == keys.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << data.sorted_keys[i]
             << " (at index " << i << ")";
    if (key_iter->second.front() != data.sorted_keys[i])
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of stable order or unknown val "
             << data.sorted_keys[i]
             << " (at index " << i << ")";
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}


template <typename Res,
          typename K,
          typename V,
          typename Compare,
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<Res, sort_pairs_interface_tag, K, V> & data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, unstable_sort_tag, sort_pairs_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to vals
  using val_map = std::unordered_multiset<V>;
  std::unordered_map<K, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(data.orig_keys[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(data.orig_vals[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " keys " << data.sorted_keys[i-1] << ", " << data.sorted_keys[i] << " out of order"
             << " vals " << data.sorted_vals[i-1] << ", " << data.sorted_vals[i]
             << " (at index " << i-1 << ")";
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(data.sorted_keys[i]);
    if (key_iter == keys_to_vals.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    auto val_iter = key_iter->second.find(data.sorted_vals[i]);
    if (val_iter == key_iter->second.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (unstable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate val "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    key_iter->second.erase(val_iter);
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename Res,
          typename K,
          typename V,
          typename Compare,
          typename TestSorter,
          typename CompareInterface>
::testing::AssertionResult testSort(
    const char* test_name,
    const unsigned seed,
    SortData<Res, sort_pairs_interface_tag, K, V> & data,
    RAJA::Index_type N,
    Compare comp,
    TestSorter test_sorter, stable_sort_tag, sort_pairs_interface_tag si, CompareInterface ci)
{
  doSort(data, N, comp, test_sorter, si, ci);

  // make map of keys to vals
  using val_map = std::list<V>;
  std::unordered_map<K, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(data.orig_keys[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(data.orig_keys[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(data.orig_vals[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(data.sorted_keys[i], data.sorted_keys[i-1]))
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of order "
             << " keys " << data.sorted_keys[i-1] << ", " << data.sorted_keys[i]
             << " vals " << data.sorted_vals[i-1] << ", " << data.sorted_vals[i]
             << " (at index " << i-1 << ")";
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(data.sorted_keys[i]);
    if (key_iter == keys_to_vals.end())
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " unknown or duplicate key "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    if (key_iter->second.front() != data.sorted_vals[i])
      return ::testing::AssertionFailure()
             << test_sorter.name() << " (stable sort pairs) " << test_name
             << " (with N " << N << " with seed " << seed << ")"
             << " out of stable order or unknown val "
             << " key " << data.sorted_keys[i]
             << " val " << data.sorted_vals[i]
             << " (at index " << i << ")";
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  return ::testing::AssertionSuccess();
}


template <typename Res,
          typename K,
          typename V,
          typename Sorter>
void testSorterResInterfaces(
    std::false_type,
    unsigned,
    SortData<Res, typename Sorter::sort_interface, K, V> &,
    RAJA::Index_type,
    Sorter)
{
  // Sorter does not support resource interface, no tests
}

template <typename Res,
          typename K,
          typename V,
          typename Sorter>
void testSorterResInterfaces(
    std::true_type,
    unsigned seed,
    SortData<Res, typename Sorter::sort_interface, K, V> & data,
    RAJA::Index_type N,
    Sorter sorter)
{
  // Sorter supports resource interface, res tests
  using stability_category = typename Sorter::sort_category ;
  using pairs_category     = typename Sorter::sort_interface ;
  using resource_no_comparator  = sort_res_default_interface_tag;
  using resource_use_comparator = sort_res_comp_interface_tag;

  ASSERT_TRUE(testSort("resource+default", seed, data, N, RAJA::operators::less<K>{},
      sorter, stability_category{}, pairs_category{}, resource_no_comparator{}));
  ASSERT_TRUE(testSort("resource+ascending", seed, data, N, RAJA::operators::less<K>{},
      sorter, stability_category{}, pairs_category{}, resource_use_comparator{}));
  ASSERT_TRUE(testSort("resource+descending", seed, data, N, RAJA::operators::greater<K>{},
      sorter, stability_category{}, pairs_category{}, resource_use_comparator{}));
}

template <typename K,
          typename Sorter,
          typename Res>
void testSorterInterfaces(unsigned seed, RAJA::Index_type MaxN, Sorter sorter, Res res)
{
  using stability_category = typename Sorter::sort_category ;
  using pairs_category     = typename Sorter::sort_interface ;
  using supports_resource  = typename Sorter::supports_resource ;
  using no_comparator      = sort_default_interface_tag;
  using use_comparator     = sort_comp_interface_tag;

  std::mt19937 rng(seed);
  RAJA::Index_type N = std::uniform_int_distribution<RAJA::Index_type>((MaxN+1)/2, MaxN)(rng);
  std::uniform_int_distribution<RAJA::Index_type> dist(-N, N);

  SortData<Res, pairs_category, K> data(N, res, [&](){ return dist(rng); });

  ASSERT_TRUE(testSort("default", seed, data, N, RAJA::operators::less<K>{},
      sorter, stability_category{}, pairs_category{}, no_comparator{}));
  ASSERT_TRUE(testSort("ascending", seed, data, N, RAJA::operators::less<K>{},
      sorter, stability_category{}, pairs_category{}, use_comparator{}));
  ASSERT_TRUE(testSort("descending", seed, data, N, RAJA::operators::greater<K>{},
      sorter, stability_category{}, pairs_category{}, use_comparator{}));

  testSorterResInterfaces(supports_resource(), seed, data, N, sorter);
}

template <typename K,
          typename Sorter,
          typename Res>
void testSorter(unsigned seed, RAJA::Index_type MaxN, Sorter sorter, Res res)
{
  testSorterInterfaces<K>(seed, 0, sorter, res);
  for (RAJA::Index_type n = 1; n <= MaxN; n *= 10) {
    testSorterInterfaces<K>(seed, n, sorter, res);
  }
}

inline unsigned get_random_seed()
{
  static unsigned seed = std::random_device{}();
  return seed;
}


TYPED_TEST_SUITE_P(SortUnitTest);

template < typename T >
class SortUnitTest : public ::testing::Test
{ };

TYPED_TEST_P(SortUnitTest, UnitSort)
{
  using Sorter   = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType  = typename camp::at<TypeParam, camp::num<1>>::type;
  using KeyType  = typename camp::at<TypeParam, camp::num<2>>::type;
  using MaxNType = typename camp::at<TypeParam, camp::num<3>>::type;

  unsigned seed = get_random_seed();
  RAJA::Index_type MaxN = MaxNType::value;
  Sorter sorter{};
  ResType res = ResType::get_default();

  testSorter<KeyType>(seed, MaxN, sorter, res);
}

REGISTER_TYPED_TEST_SUITE_P(SortUnitTest, UnitSort);


//
// Key types for sort tests
//
using SortKeyTypeList =
  camp::list<
              RAJA::Index_type,
              int,
#if defined(RAJA_TEST_EXHAUSTIVE)
              unsigned,
              long long,
              unsigned long long,
              float,
#endif
              double
            >;

// Max test lengths for sort tests
using SortMaxNListDefault =
  camp::list<
              camp::num<10000>
            >;

using SortMaxNListSmall =
  camp::list<
              camp::num<1000>
            >;

using SortMaxNListTiny =
  camp::list<
              camp::num<100>
            >;

#endif //__TEST_ALGORITHM_SORT_UTILS_HPP__

