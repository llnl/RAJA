//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Header defining "for one" unit test utility so that constructs can be
// tested outside of standard RAJA kernel launch utilities (forall, kernel).
//

#ifndef __RAJA_test_for3d3d_HPP__
#define __RAJA_test_for3d3d_HPP__

#include "RAJA_unit-test-policy.hpp"

struct dim3d
{
  int dim[3];

  RAJA_HOST_DEVICE
  int& operator[](int i) { return dim[i]; }
  RAJA_HOST_DEVICE
  int const& operator[](int i) const { return dim[i]; }

  RAJA_HOST_DEVICE
  int product() const { return dim[0] * dim[1] * dim[2]; }
};

struct dim3d3d
{
  dim3d thread;
  dim3d block;

  RAJA_HOST_DEVICE
  int product() const { return thread.product() * block.product(); }
};

RAJA_HOST_DEVICE
int index(dim3d3d idx, dim3d3d dim)
{
  return               idx.thread[0] +
      dim.thread[0] * (idx.thread[1] +
      dim.thread[1] * (idx.thread[2] +
      dim.thread[2] * (idx.block[0] +
      dim.block[0]  * (idx.block[1] +
      dim.block[1]  * (idx.block[2])))));
}

///
/// for3d3d<test_policy>( dim,
///     [=] RAJA_HOST_DEVICE(dim3d3d idx,
///                          dim3d3d dim)
/// {
///   /* code to test */
/// } );
///
template < typename test_policy, typename L >
inline void for3d3d(dim3d3d dim, L&& run);

// test_seq implementation
template < typename L >
inline void for3d3d(test_seq, dim3d3d dim, L&& run)
{
  for (int bz = 0; bz < dim.block[2]; ++bz) {
  for (int by = 0; by < dim.block[1]; ++by) {
  for (int bx = 0; bx < dim.block[0]; ++bx) {
    for (int tz = 0; tz < dim.thread[2]; ++tz) {
    for (int ty = 0; ty < dim.thread[1]; ++ty) {
    for (int tx = 0; tx < dim.thread[0]; ++tx) {
      run(dim3d3d{{tx,ty,tz}, {bx,by,bz}}, dim);
    }}}
  }}}
}

#if defined(RAJA_ENABLE_TARGET_OPENMP)

// test_openmp_target implementation
template < typename L >
inline void for3d3d(test_openmp_target, dim3d3d dim, L&& run)
{
#pragma omp target teams distribute collapse(3)
  for (int bz = 0; bz < dim.block[2]; ++bz) {
  for (int by = 0; by < dim.block[1]; ++by) {
  for (int bx = 0; bx < dim.block[0]; ++bx) {
#pragma omp parallel for collapse(3)
    for (int tz = 0; tz < dim.thread[2]; ++tz) {
    for (int ty = 0; ty < dim.thread[1]; ++ty) {
    for (int tx = 0; tx < dim.thread[0]; ++tx) {
      run(dim3d3d{{tx,ty,tz}, {bx,by,bz}}, dim);
    }}}
  }}}
}

#endif

#if defined(RAJA_ENABLE_CUDA)

template <typename L>
__global__ void for3d3d_cuda_global(L run)
{
  run(dim3d3d{{static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z)},
              {static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.z)}},
      dim3d3d{{static_cast<int>(blockDim.x), static_cast<int>(blockDim.y), static_cast<int>(blockDim.z)},
              {static_cast<int>(gridDim.x), static_cast<int>(gridDim.y), static_cast<int>(gridDim.z)}});
}

// test_cuda implementation
template < typename L >
inline void for3d3d(test_cuda, dim3d3d dim, L&& run)
{
   for3d3d_cuda_global<<<dim3(dim.block[0], dim.block[1], dim.block[2]),
                         dim3(dim.thread[0], dim.thread[1], dim.thread[2])>>>(std::forward<L>(run));
   CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
   CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
}

#endif

#if defined(RAJA_ENABLE_HIP)

template <typename L>
__global__ void for3d3d_hip_global(L run)
{
  run(dim3d3d{{static_cast<int>(threadIdx.x), static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z)},
              {static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.z)}},
      dim3d3d{{static_cast<int>(blockDim.x), static_cast<int>(blockDim.y), static_cast<int>(blockDim.z)},
              {static_cast<int>(gridDim.x), static_cast<int>(gridDim.y), static_cast<int>(gridDim.z)}});
}

// test_hip implementation
template < typename L >
inline void for3d3d(test_hip, dim3d3d dim, L&& run)
{
   hipLaunchKernelGGL(for3d3d_hip_global<camp::decay<L>>,
                      dim3(dim.block[0], dim.block[1], dim.block[2]),
                      dim3(dim.thread[0], dim.thread[1], dim.thread[2]),
                      0, 0,
                      std::forward<L>(run));
   CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
   CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
}

#endif

template < typename test_policy, typename L >
void for3d3d(dim3d3d dim, L&& run)
{
  for3d3d(test_policy{}, dim, std::forward<L>(run));
}

#endif // RAJA_test_for3d3d_HPP__
