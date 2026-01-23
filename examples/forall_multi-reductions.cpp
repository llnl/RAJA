//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <limits>

#include "RAJA/RAJA.hpp"

#include "camp/tuple.hpp"

/*
 *  MultiReduction Example using RAJA forall
 *
 *  This example illustrates use of the RAJA multi-reduction types: min, max,
 *  sum, and, and or.
 *
 *  RAJA features shown:
 *    - `forall' loop iteration template method
 *    -  Index range segment
 *    -  Execution policies
 *    -  MultiReduction types
 *
 */

template < typename t_exec_policy, typename t_multi_reduce_policy >
struct Backend
{
  using exec_policy = t_exec_policy;
  using multi_reduce_policy = t_multi_reduce_policy;

  std::string name;
};

auto example_policies = camp::make_tuple(

      Backend<RAJA::seq_exec, RAJA::seq_multi_reduce>{"Sequential"}

#if defined(RAJA_ENABLE_OPENMP)
    , Backend<RAJA::omp_parallel_for_exec, RAJA::omp_multi_reduce>{"OpenMP"}
#endif

#if defined(RAJA_ENABLE_CUDA)
    , Backend<RAJA::cuda_exec_async<256>, RAJA::cuda_multi_reduce_atomic>{"Cuda"}
#endif

#if defined(RAJA_ENABLE_HIP)
    , Backend<RAJA::hip_exec_async<256>, RAJA::hip_multi_reduce_atomic>{"Hip"}
#endif

    );

template < typename exec_policy, typename multi_reduce_policy >
void example_code(RAJA::RangeSegment arange, int num_bins, int* bins, int* a)
{
  RAJA::MultiReduceSum<multi_reduce_policy, int>    multi_reduce_sum(num_bins);
  RAJA::MultiReduceMin<multi_reduce_policy, int>    multi_reduce_min(num_bins);
  RAJA::MultiReduceMax<multi_reduce_policy, int>    multi_reduce_max(num_bins);
  RAJA::MultiReduceBitAnd<multi_reduce_policy, int> multi_reduce_and(num_bins);
  RAJA::MultiReduceBitOr<multi_reduce_policy, int>  multi_reduce_or(num_bins);

  RAJA::forall<exec_policy>(arange,
      [=] RAJA_HOST_DEVICE(RAJA::Index_type i) {

    int bin = bins[i];

    multi_reduce_sum[bin] +=  a[i];
    multi_reduce_min[bin].min(a[i]);
    multi_reduce_max[bin].max(a[i]);
    multi_reduce_and[bin] &=  a[i];
    multi_reduce_or [bin] |=  a[i];

  });

  for (int bin = 0; bin < num_bins; ++bin) {
    std::cout << "\tsum[" << bin << "] = " << multi_reduce_sum.get(bin) << '\n';
    std::cout << "\tmin[" << bin << "] = " << multi_reduce_min.get(bin) << '\n';
    std::cout << "\tmax[" << bin << "] = " << multi_reduce_max.get(bin) << '\n';
    std::cout << "\tand[" << bin << "] = " << multi_reduce_and.get(bin) << '\n';
    std::cout << "\tor [" << bin << "] = " << multi_reduce_or .get(bin) << '\n';
    std::cout << '\n';
  }
}

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{

  // _multi_reductions_array_init_start
//
// Define array length
//
  const int N = 1000000;
  const int num_bins = 10;

//
// Allocate array data and initialize data to alternating sequence of 1, -1.
//
  camp::resources::Host host_res;
  int* host_bins = host_res.template allocate<int>(N);
  int* host_a    = host_res.template allocate<int>(N);

  for (int i = 0; i < N; ++i) {
    host_bins[i] = i % num_bins;
    host_a[i] = (i % (2*num_bins)) - num_bins;
  }

  // _multi_reductions_array_init_end

//
// Note: with this data initialization scheme, the following results will
//       be observed for all reduction kernels below:
//
// for bin in [0, num_bins)
//  - the sum will be (bin - num_bins/2) * N / num_bins
//  - the min will be bin - num_bins
//  - the max will be bin
//  - the and will be min & max
//  - the or  will be min | max
//

//
// Define index range for iterating over a elements in all examples
//
  // _multi_reductions_range_start
  RAJA::RangeSegment arange(0, N);
  // _multi_reductions_range_end

//----------------------------------------------------------------------------//

  RAJA::for_each_tuple(example_policies, [&](auto const& backend) {

    std::cout << "Running " << backend.name << " policies" << '\n';

    using exec_policy = typename std::decay_t<decltype(backend)>::exec_policy;
    using multi_reduce_policy = typename std::decay_t<decltype(backend)>::multi_reduce_policy;

    auto res = RAJA::resources::get_default_resource<exec_policy>();

    int* bins = res.template allocate<int>(N);
    int* a    = res.template allocate<int>(N);

    res.memcpy(bins, host_bins, N*sizeof(int));
    res.memcpy(a   , host_a   , N*sizeof(int));

    example_code<exec_policy, multi_reduce_policy>(arange, num_bins, bins, a);

    res.deallocate(bins);
    res.deallocate(a   );

    std::cout << std::endl;
  });

//----------------------------------------------------------------------------//

//
// Clean up.
//
  host_res.deallocate(host_bins);
  host_res.deallocate(host_a   );

  std::cout << "\n DONE!...\n";

  return 0;
}
