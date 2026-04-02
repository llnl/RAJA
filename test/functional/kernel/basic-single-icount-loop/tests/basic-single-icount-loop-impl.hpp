//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __BASIC_SINGLE_ICOUNT_LOOP_SEGMENTS_IMPL_HPP__
#define __BASIC_SINGLE_ICOUNT_LOOP_SEGMENTS_IMPL_HPP__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <vector>

template <typename IDX_TYPE, typename EXEC_POLICY, typename WORKING_RES, typename SEG_TYPE>
void KernelBasicSingleICountLoopTestImpl(const SEG_TYPE& seg, 
                                   const std::vector<IDX_TYPE>& seg_idx,
                                   WORKING_RES working_res,
                                   camp::resources::Resource erased_working_res)
{
  IDX_TYPE idx_len = static_cast<IDX_TYPE>( seg_idx.size() );
  IDX_TYPE data_len = IDX_TYPE(0);
  if ( seg_idx.size() > 0 ) {
    data_len = seg_idx[seg_idx.size() - 1] + 1;
  }

  IDX_TYPE* working_array;
  IDX_TYPE* working_array_i;
  IDX_TYPE* check_array;
  IDX_TYPE* check_array_i;
  IDX_TYPE* test_array;
  IDX_TYPE* test_array_i;

  if ( RAJA::stripIndexType(data_len) == 0 ) {
    data_len++;
  }

  allocateForallTestData<IDX_TYPE>(data_len,
                                   erased_working_res,
                                   &working_array,
                                   &check_array,
                                   &test_array);

  allocateForallTestData<IDX_TYPE>(data_len,
                                   erased_working_res,
                                   &working_array_i,
                                   &check_array_i,
                                   &test_array_i);

  memset(static_cast<void*>(test_array), 0, 
         sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  working_res.memcpy(working_array, test_array, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  working_res.memcpy(working_array_i, test_array_i, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  if ( RAJA::stripIndexType(idx_len) > 0 ) {

    for (IDX_TYPE i = IDX_TYPE(0); i < idx_len; ++i) {
      test_array  [ RAJA::stripIndexType(seg_idx[RAJA::stripIndexType(i)]) ] = 
        seg_idx[RAJA::stripIndexType(i)];
      test_array_i[ RAJA::stripIndexType(RAJA::stripIndexType(i)) ] = 
        IDX_TYPE(i);
    }
 
    RAJA::kernel_param<EXEC_POLICY>(
      RAJA::make_tuple(seg),
      RAJA::make_tuple(IDX_TYPE(0)),
      
      [=] RAJA_HOST_DEVICE(IDX_TYPE idx, IDX_TYPE i_idx) {
        working_array[RAJA::stripIndexType(idx)] = IDX_TYPE(idx) ;
        working_array_i[RAJA::stripIndexType(i_idx)] = IDX_TYPE(i_idx) ;
      }
    );

  } else { // zero-length segment

    RAJA::kernel_param<EXEC_POLICY>(
      RAJA::make_tuple(seg),
      RAJA::make_tuple(IDX_TYPE(0)),
      
      [=] RAJA_HOST_DEVICE(IDX_TYPE idx, IDX_TYPE i_idx) {
        (void) idx; (void) i_idx;
        working_array[0]++;
        working_array_i[0]++;
      }
    );

  }

  working_res.memcpy(check_array, working_array, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));
  working_res.memcpy(check_array_i, working_array_i, 
                     sizeof(IDX_TYPE) * RAJA::stripIndexType(data_len));

  for (IDX_TYPE i = IDX_TYPE(0); i < data_len; ++i) {
    ASSERT_EQ( test_array[RAJA::stripIndexType(i)],
               check_array[RAJA::stripIndexType(i)] );
    ASSERT_EQ( test_array_i[RAJA::stripIndexType(i)],
               check_array_i[RAJA::stripIndexType(i)] );
  }

  deallocateForallTestData<IDX_TYPE>(erased_working_res,
                                     working_array,
                                     check_array,
                                     test_array);

  deallocateForallTestData<IDX_TYPE>(erased_working_res,
                                     working_array_i,
                                     check_array_i,
                                     test_array_i);
}

#endif  // __BASIC_SINGLE_ICOUNT_LOOP_SEGMENTS_IMPL_HPP__
