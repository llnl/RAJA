//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __BASIC_FISSION_FUSION_LOOP_SEGMENTS_IMPL_HPP__
#define __BASIC_FISSION_FUSION_LOOP_SEGMENTS_IMPL_HPP__

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <vector>

template <typename IDX_TYPE,
          typename EXEC_POLICY,
          typename WORKING_RES,
          typename SEG_TYPE>
void KernelBasicFissionFusionLoopTestImpl(
    const SEG_TYPE& seg,
    const std::vector<IDX_TYPE>& seg_idx,
    WORKING_RES working_res,
    camp::resources::Resource erased_working_res)
{
  IDX_TYPE data_len = IDX_TYPE(0);

  if (seg_idx.size() > 0) {
    data_len = seg_idx[seg_idx.size() - 1] + 1;
  }

  using DATA_TYPE = int;
  DATA_TYPE* working_array_x;
  DATA_TYPE* working_array_y;
  DATA_TYPE* check_array_x;
  DATA_TYPE* check_array_y;
  DATA_TYPE* test_array_x;
  DATA_TYPE* test_array_y;

  allocateForallTestData<DATA_TYPE>(RAJA::stripIndexType(data_len),
                                    erased_working_res,
                                    &working_array_x,
                                    &check_array_x,
                                    &test_array_x);

  allocateForallTestData<DATA_TYPE>(RAJA::stripIndexType(data_len),
                                    erased_working_res,
                                    &working_array_y,
                                    &check_array_y,
                                    &test_array_y);


  working_res.memset(working_array_x,
                     0,
                     sizeof(DATA_TYPE) * RAJA::stripIndexType(data_len));

  RAJA::kernel<EXEC_POLICY>(
      RAJA::make_tuple(seg, seg),

      [=] RAJA_HOST_DEVICE(IDX_TYPE i) {
        RAJA::atomicAdd<RAJA::auto_atomic>(
            &working_array_x[RAJA::stripIndexType(i)], (DATA_TYPE)1);
      },

      [=] RAJA_HOST_DEVICE(IDX_TYPE i) {
        RAJA::atomicAdd<RAJA::auto_atomic>(
            &working_array_x[RAJA::stripIndexType(i)], (DATA_TYPE)2);
      }

  );

  working_res.memcpy(check_array_x,
                     working_array_x,
                     sizeof(DATA_TYPE) * RAJA::stripIndexType(data_len));

  memset(static_cast<void*>(check_array_y),
         0,
         sizeof(DATA_TYPE) * RAJA::stripIndexType(data_len));

  RAJA::forall<RAJA::seq_exec>(seg_idx, [=](IDX_TYPE i) {
    check_array_y[RAJA::stripIndexType(i)] += 1;
    check_array_y[RAJA::stripIndexType(i)] += 2;
  });


  for (IDX_TYPE i = IDX_TYPE(0); i < data_len; ++i) {
    ASSERT_EQ(check_array_x[RAJA::stripIndexType(i)],
              check_array_y[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<DATA_TYPE>(erased_working_res,
                                      working_array_x,
                                      check_array_x,
                                      test_array_x);


  deallocateForallTestData<DATA_TYPE>(erased_working_res,
                                      working_array_y,
                                      check_array_y,
                                      test_array_y);
}

#endif  // __BASIC_FISSION_FUSION_LOOP_SEGMENTS_IMPL_HPP__
