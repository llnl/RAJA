//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#define OP_GREATER RAJA::operators::greater<int>
#define OP_LESS RAJA::operators::less<int>

#define CHECK_UNSTABLE_SORT_RESULT(X) checkUnstableSortResult<X>(in, out, N) 
#define CHECK_UNSTABLE_SORT_PAIR_RESULT(X) checkUnstableSortResult<X>(in, out, in_vals, out_vals, N) 
#define CHECK_STABLE_SORT_RESULT(X) checkStableSortResult<X>(in, out, N) 
#define CHECK_STABLE_SORT_PAIR_RESULT(X) checkStableSortResult<X>(in, out, in_vals, out_vals, N) 

#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"

/*
 *  Sort Exercise
 *
 *  Exercise demonstrates how to perform RAJA unstable and stable sort operations
 *  for integer arrays, including pairs variant, using different comparators.
 *  Other array data types, comparators, etc. are similar
 *
 *  RAJA features shown:
 *    - `RAJA::sort` and `RAJA::sort_pairs` methods
 *    - `RAJA::stable_sort` and `RAJA::stable_sort_pairs` methods
 *    -  RAJA operators
 *    -  Execution policies
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Specify the number of threads in a GPU thread block
*/
#if defined(RAJA_ENABLE_CUDA)
constexpr int CUDA_BLOCK_SIZE = 16;
#endif

#if defined(RAJA_ENABLE_HIP)
constexpr int HIP_BLOCK_SIZE = 16;
#endif

//
// Functions for checking results and printing vectors
//
template <typename Function, typename T>
void checkUnstableSortResult(const T* in, const T* out, int N);
template <typename Function, typename T, typename U>
void checkUnstableSortResult(const T* in, const T* out,
                             const U* in_vals, const U* out_vals, int N);
//
template <typename Function, typename T>
void checkStableSortResult(const T* in, const T* out, int N);
template <typename Function, typename T, typename U>
void checkStableSortResult(const T* in, const T* out,
                           const U* in_vals, const U* out_vals, int N);
//
template <typename T>
void printArray(const T* k, int N);
template <typename T, typename U>
void printArray(const T* k, const U* v, int N);


int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA sort example...\n";

  // _sort_array_init_start
//
// Define array length
//
  constexpr int N = 20;

//
// Allocate and initialize vector data
//
  int* in = memoryManager::allocate<int>(N);
  int* out = memoryManager::allocate<int>(N);

  unsigned* in_vals = memoryManager::allocate<unsigned>(N);
  unsigned* out_vals = memoryManager::allocate<unsigned>(N);

  std::iota(in      , in + N/2, 0);
  std::iota(in + N/2, in + N  , 0);
  std::shuffle(in      , in + N/2, std::mt19937{12345u});
  std::shuffle(in + N/2, in + N  , std::mt19937{67890u});

  std::fill(in_vals      , in_vals + N/2, 0);
  std::fill(in_vals + N/2, in_vals + N  , 1);

  std::cout << "\n in keys...\n";
  printArray(in, N);
  std::cout << "\n in (key, value) pairs...\n";
  printArray(in, in_vals, N);
  std::cout << "\n";

  // _sort_array_init_end


//----------------------------------------------------------------------------//
// Perform various sequential sorts to illustrate unstable/stable,
// pairs, default sorts with different comparators
//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential sort (default)...\n";

  // _sort_seq_start
  std::copy_n(in, N, out);

  RAJA::sort<RAJA::seq_exec>(RAJA::make_span(out, N));
  // _sort_seq_end

  //checkUnstableSortResult<RAJA::operators::less<int>>(in, out, N);
  CHECK_UNSTABLE_SORT_RESULT(OP_LESS);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential sort (non-decreasing)...\n";

  std::copy_n(in, N, out);

  // _sort_seq_less_start
  RAJA::sort<RAJA::seq_exec>(RAJA::make_span(out, N),
                             RAJA::operators::less<int>{});
  // _sort_seq_less_end

  //checkUnstableSortResult<RAJA::operators::less<int>>(in, out, N);
  CHECK_UNSTABLE_SORT_RESULT(OP_LESS);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential stable_sort (non-decreasing)...\n";

  std::copy_n(in, N, out);

  // _sort_stable_seq_less_start
  RAJA::stable_sort<RAJA::seq_exec>(RAJA::make_span(out, N),
                                    RAJA::operators::less<int>{});
  // _sort_stable_seq_less_end

  //checkStableSortResult<RAJA::operators::less<int>>(in, out, N);
  CHECK_STABLE_SORT_RESULT(OP_LESS);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential stable_sort (non-increasing)...\n";

  std::copy_n(in, N, out);

  // _sort_stable_seq_greater_start
  RAJA::stable_sort<RAJA::seq_exec>(RAJA::make_span(out, N),
                                    RAJA::operators::greater<int>{});
  // _sort_stable_seq_greater_end

  //checkStableSortResult<RAJA::operators::greater<int>>(in, out, N);
  CHECK_STABLE_SORT_RESULT(OP_GREATER);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential sort_pairs (non-decreasing)...\n";

  std::copy_n(in, N, out);
  std::copy_n(in_vals, N, out_vals);

  // _sort_pairs_seq_less_start
  RAJA::sort_pairs<RAJA::seq_exec>(RAJA::make_span(out, N),
                                   RAJA::make_span(out_vals, N),
                                   RAJA::operators::less<int>{});
  // _sort_pairs_seq_less_end

  //checkUnstableSortResult<RAJA::operators::less<int>>(in, out, in_vals, out_vals, N);
  CHECK_UNSTABLE_SORT_PAIR_RESULT(OP_LESS);
  printArray(out, out_vals, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential stable_sort_pairs (non-increasing)...\n";

  std::copy_n(in, N, out);
  std::copy_n(in_vals, N, out_vals);

  // _sort_stable_pairs_seq_greater_start
  RAJA::stable_sort_pairs<RAJA::seq_exec>(RAJA::make_span(out, N),
                                          RAJA::make_span(out_vals, N),
                                          RAJA::operators::greater<int>{});
  // _sort_stable_pairs_seq_greater_end

  //checkStableSortResult<RAJA::operators::greater<int>>(in, out, in_vals, out_vals, N);
  CHECK_STABLE_SORT_PAIR_RESULT(OP_GREATER);
  printArray(out, out_vals, N);
  std::cout << "\n";


#if defined(RAJA_ENABLE_OPENMP)

//----------------------------------------------------------------------------//
// Perform a couple of OpenMP sorts...
//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP sort (non-decreasing)...\n";

  std::copy_n(in, N, out);

  // _sort_omp_less_start
  RAJA::sort<RAJA::omp_parallel_for_exec>(RAJA::make_span(out, N),
                                          RAJA::operators::less<int>{});
  // _sort_omp_less_end

  //checkUnstableSortResult<RAJA::operators::less<int>>(in, out, N);
  CHECK_UNSTABLE_SORT_RESULT(OP_LESS);
  printArray(out, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP stable_sort_pairs (non-increasing)...\n";

  std::copy_n(in, N, out);
  std::copy_n(in_vals, N, out_vals);

  // _sort_stable_pairs_omp_greater_start
  RAJA::stable_sort_pairs<RAJA::omp_parallel_for_exec>(RAJA::make_span(out, N),
                                                       RAJA::make_span(out_vals, N),
                                                       RAJA::operators::greater<int>{});
  // _sort_stable_pairs_omp_greater_end

  //checkStableSortResult<RAJA::operators::greater<int>>(in, out, in_vals, out_vals, N);
  CHECK_STABLE_SORT_PAIR_RESULT(OP_GREATER);
  printArray(out, out_vals, N);
  std::cout << "\n";

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

//----------------------------------------------------------------------------//
// Perform a couple of CUDA sorts...
//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA sort_pairs (non-increasing)...\n";

  std::copy_n(in, N, out);
  std::copy_n(in_vals, N, out_vals);

  // _sort_pairs_cuda_greater_start
  RAJA::sort_pairs<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::make_span(out, N),
                                                     RAJA::make_span(out_vals, N),
                                                     RAJA::operators::greater<int>{});
  // _sort_pairs_cuda_greater_end

  //checkUnstableSortResult<RAJA::operators::greater<int>>(in, out, in_vals, out_vals, N);
  CHECK_UNSTABLE_SORT_PAIR_RESULT(OP_GREATER);
  printArray(out, out_vals, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA stable_sort (non-decreasing)...\n";

  std::copy_n(in, N, out);

  // _sort_stable_cuda_less_start
  RAJA::stable_sort<RAJA::cuda_exec<CUDA_BLOCK_SIZE>>(RAJA::make_span(out, N),
                                                      RAJA::operators::less<int>{});
  // _sort_stable_cuda_less_end

  //checkStableSortResult<RAJA::operators::less<int>>(in, out, N);
  CHECK_STABLE_SORT_RESULT(OP_LESS);
  printArray(out, N);
  std::cout << "\n";

#endif

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

//----------------------------------------------------------------------------//
// Perform a couple of HIP sorts...
//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP sort_pairs (non-decreasing)...\n";

  std::copy_n(in, N, out);
  std::copy_n(in_vals, N, out_vals);

  int* d_out = memoryManager::allocate_gpu<int>(N);
  int* d_out_vals = memoryManager::allocate_gpu<int>(N);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_out, out, N * sizeof(int), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_out_vals, out_vals, N * sizeof(int), hipMemcpyHostToDevice);

  RAJA::sort_pairs<RAJA::hip_exec<HIP_BLOCK_SIZE>>(RAJA::make_span(d_out, N),
                                                   RAJA::make_span(d_out_vals, N),
                                                   RAJA::operators::less<int>{});

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, out_vals, d_out_vals, N * sizeof(int), hipMemcpyDeviceToHost);

  //checkUnstableSortResult<RAJA::operators::less<int>>(in, out, in_vals, out_vals, N);
  CHECK_UNSTABLE_SORT_PAIR_RESULT(OP_LESS);
  printArray(out, out_vals, N);
  std::cout << "\n";

//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP stable_sort (non-increasing)...\n";

  std::copy_n(in, N, out);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_out, out, N * sizeof(int), hipMemcpyHostToDevice);

  // _sort_stable_hip_greater_start
  RAJA::stable_sort<RAJA::hip_exec<HIP_BLOCK_SIZE>>(
    RAJA::make_span(d_out, N),
    RAJA::operators::greater<int>{});
  // _sort_stable_hip_greater_end

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, out, d_out, N * sizeof(int), hipMemcpyDeviceToHost);

  //checkStableSortResult<RAJA::operators::greater<int>>(in, out, N);
  CHECK_STABLE_SORT_RESULT(OP_GREATER);
  printArray(out, N);
  std::cout << "\n";

  memoryManager::deallocate_gpu(d_out);
  memoryManager::deallocate_gpu(d_out_vals);

#endif


//----------------------------------------------------------------------------//

//
// Clean up.
//
  memoryManager::deallocate(in);
  memoryManager::deallocate(out);

  memoryManager::deallocate(in_vals);
  memoryManager::deallocate(out_vals);

  std::cout << "\n DONE!...\n";

  return 0;
}

template <typename Comparator, typename T>
bool equivalent(T const& a, T const& b, Comparator comp)
{
  return !comp(a, b) && !comp(b, a);
}

//
// Function to check unstable sort result
//
template <typename Comparator, typename T>
void checkUnstableSortResult(const T* in, const T* out, int N)
{
  Comparator comp;
  bool correct = true;

  // make map of keys to keys
  using val_map = std::unordered_multiset<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(in[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(in[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(in[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(out[i], out[i-1])) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << out[i-1] << ", " << out[i]
                << " out of order"
                << " (at index " << i-1 << ")\n";
    }
    // test there is an item with this
    auto key_iter = keys.find(out[i]);
    if (key_iter == keys.end()) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << out[i]
                << " unknown or duplicate key"
                << " (at index " << i << ")\n";
    }
    auto val_iter = key_iter->second.find(out[i]);
    if (val_iter == key_iter->second.end()) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << out[i]
                << " unknown or duplicate val"
                << " (at index " << i << ")\n";
    }
    key_iter->second.erase(val_iter);
    if (key_iter->second.size() == 0) {
      keys.erase(key_iter);
    }
  }
  if (correct) {
    std::cout << "\n\t result -- CORRECT\n";
  }
}

template <typename Comparator, typename T, typename U>
void checkUnstableSortResult(const T* in, const T* out,
                             const U* in_vals, const U* out_vals, int N)
{
  Comparator comp;
  bool correct = true;

  // make map of keys to vals
  using val_map = std::unordered_multiset<U>;
  std::unordered_map<T, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(in[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(in[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace(in_vals[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(out[i], out[i-1])) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << "("  << out[i-1] << "," << out_vals[i-1] << "),"
                << " (" << out[i]   << "," << out_vals[i]   << ")"
                << " out of order"
                << " (at index " << i-1 << ")\n";
    }
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(out[i]);
    if (key_iter == keys_to_vals.end()) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << "(" << out[i]   << "," << out_vals[i]   << ")"
                << " unknown or duplicate key"
                << " (at index " << i << ")\n";
    }
    auto val_iter = key_iter->second.find(out_vals[i]);
    if (val_iter == key_iter->second.end()) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << "(" << out[i]   << "," << out_vals[i]   << ")"
                << " unknown or duplicate val"
                << " (at index " << i << ")\n";
    }
    key_iter->second.erase(val_iter);
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  if (correct) {
    std::cout << "\n\t result -- CORRECT\n";
  }
}

//
// Function to check stable sort result
//
template <typename Comparator, typename T>
void checkStableSortResult(const T* in, const T* out, int N)
{
  Comparator comp;
  bool correct = true;

  // make map of keys to keys
  using val_map = std::list<T>;
  std::unordered_map<T, val_map> keys;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys.find(in[i]);
    if (key_iter == keys.end()) {
      auto ret = keys.emplace(in[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(in[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(out[i], out[i-1])) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << out[i-1] << ", " << out[i]
                << " out of order "
                << " (at index " << i-1 << ")\n";
    }
    // test there is an item with this
    auto key_iter = keys.find(out[i]);
    if (key_iter == keys.end()) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << out[i]
                << " unknown or duplicate key "
                << " (at index " << i << ")\n";
    }
    if (key_iter->second.front() != out[i]) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << out[i]
                << " out of stable order or unknown val "
                << " (at index " << i << ")\n";
    }
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys.erase(key_iter);
    }
  }
  if (correct) {
    std::cout << "\n\t result -- CORRECT\n";
  }
}

template <typename Comparator, typename T, typename U>
void checkStableSortResult(const T* in, const T* out,
                           const U* in_vals, const U* out_vals, int N)
{
  Comparator comp;
  bool correct = true;

  // make map of keys to vals
  using val_map = std::list<U>;
  std::unordered_map<T, val_map> keys_to_vals;
  for (RAJA::Index_type i = 0; i < N; i++) {
    auto key_iter = keys_to_vals.find(in[i]);
    if (key_iter == keys_to_vals.end()) {
      auto ret = keys_to_vals.emplace(in[i], val_map{});
      assert(ret.second);
      key_iter = ret.first;
    }
    key_iter->second.emplace_back(in_vals[i]);
  }

  for (RAJA::Index_type i = 0; i < N; i++) {
    // test ordering
    if (i > 0 && comp(out[i], out[i-1])) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << "("  << out[i-1] << "," << out_vals[i-1] << "),"
                << " (" << out[i]   << "," << out_vals[i]   << ")"
                << " out of order "
                << " (at index " << i-1 << ")\n";
    }
    // test there is a pair with this key and val
    auto key_iter = keys_to_vals.find(out[i]);
    if (key_iter == keys_to_vals.end()) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << "(" << out[i]   << "," << out_vals[i]   << ")"
                << " unknown or duplicate key "
                << " (at index " << i << ")\n";
    }
    if (key_iter->second.front() != out_vals[i]) {
      if (correct) {
        std::cout << "\n\t result -- WRONG\n";
        correct = false;
      }
      std::cout << "\t"
                << "(" << out[i]   << "," << out_vals[i]   << ")"
                << " out of stable order or unknown val "
                << " (at index " << i << ")\n";
    }
    key_iter->second.pop_front();
    if (key_iter->second.size() == 0) {
      keys_to_vals.erase(key_iter);
    }
  }
  if (correct) {
    std::cout << "\n\t result -- CORRECT\n";
  }
}


//
// Function to print vector.
//
template <typename T>
void printArray(const T* k, int N)
{
  std::cout << std::endl;
  for (int i = 0; i < N; ++i) { std::cout << " " << k[i]; }
  std::cout << std::endl;
}

template <typename T, typename U>
void printArray(const T* k, const U* v, int N)
{
  std::cout << std::endl;
  for (int i = 0; i < N; ++i) { std::cout << " (" << k[i] << "," << v[i] << ")"; }
  std::cout << std::endl;
}

