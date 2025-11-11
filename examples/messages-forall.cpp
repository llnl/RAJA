//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "RAJA/RAJA.hpp"
#include "RAJA/util/resource.hpp"

/*
 *  Vector Addition Example
 *
 *  Computes c = a + b, where a, b, c are vectors of ints.
 *  It illustrates similarities between a  C-style for-loop and a RAJA 
 *  forall loop with using RAJA messages to print out if a value is 
 *  negative.
 *
 *  Note: this example is the same as resource-forall.cpp with
 *  some additional logger to store how the message handler can
 *  be used as a basic logger for both host and device. 
 *
 *  RAJA features shown:
 *    - `forall` loop iteration with messages
 *    -  Create message handler with Resource argument
 *
 */

// This is a simplified example fixed string to show how
// custom types can be used with the message queue.
template <std::size_t N>
class my_string
{
public:
  char m_data[N];

  my_string(const char* str)
  {
    if (str == NULL) { return; }  
    std::size_t i = 0;
    for (i = 0; *str != '\0' && i < N; str++, i++) {
      m_data[i] = *str;
    } 

    std::size_t len = (i < N) ? i : N-1;
    m_data[len] = '\0';
  }

  const char* c_str() const
  {
    return m_data;
  }
};

//
// Functions for checking and printing results
//
void checkResult(int* res, int len); 
void printResult(int* res, int len);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA vector addition example...\n";

  RAJA::resources::Host host{};

//
// Define number of messages that can be stored
//
  const std::size_t num_messages = 1;
  const std::size_t message_sz   = RAJA::align(sizeof(RAJA::msg_header)) + 
    RAJA::align(sizeof(RAJA::msg_args<my_string<128>, int*, int, int>));

  const std::size_t buf_sz       = num_messages*message_sz;

//
// Allocate and initialize message handler and queue
//
  auto logger = RAJA::make_message_manager(buf_sz, host);
  auto cpu_msg_queue = logger.get_queue<RAJA::mpsc_queue>(0,
    [](const my_string<128>& str, int* ptr, int idx, int value) {
      std::cout << "\n " << str.c_str() << " " << ptr << " a[" << idx << "] = " << value << "\n";
    }
  );

//
// Define vector length
//
  const int N = 100000;

//
// Allocate and initialize vector data
//

  int *a = host.allocate<int>(N);
  int *b = host.allocate<int>(N);
  int *c = host.allocate<int>(N);

  int *a_ = host.allocate<int>(N);
  int *b_ = host.allocate<int>(N);
  int *c_ = host.allocate<int>(N);


  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = 2 * i;
    a_[i] = -i;
    b_[i] = 2 * i;

  }


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style vector addition...\n";

  for (int i = 0; i < N; ++i) {
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from C-style loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i];
  }

  checkResult(c, N);
  logger.wait_all();


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential vector addition...\n";


  RAJA::forall<RAJA::seq_exec>(host, RAJA::RangeSegment(0, N), [=] (int i) { 
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from RAJA seq_exec loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i]; 
  });

  checkResult(c, N);
  logger.wait_all();

//----------------------------------------------------------------------------//
// RAJA::sind_exec policy enforces simd execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA simd_exec vector addition...\n";

  RAJA::forall<RAJA::simd_exec>(host, RAJA::RangeSegment(0, N), [=] (int i) { 
    c[i] = a[i] + b[i]; 
  });

  checkResult(c, N);

#if defined(RAJA_ENABLE_OPENMP)
//----------------------------------------------------------------------------//
// RAJA::omp_for_parallel_exec policy execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_parallel_for_exec vector addition...\n";

  RAJA::forall<RAJA::omp_parallel_for_exec>(host, RAJA::RangeSegment(0, N),
  [=] (int i) {
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from RAJA omp_parallel_for_exec loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i]; 
  });

  checkResult(c, N);
  logger.wait_all();

//----------------------------------------------------------------------------//
// RAJA::omp_parallel_for_static_exec policy execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_parallel_for_static_exec (default chunksize) vector addition...\n";

  RAJA::forall<RAJA::omp_parallel_for_static_exec< >>(host, RAJA::RangeSegment(0, N),
  [=] (int i) {
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from RAJA omp_parallel_for_static_exec loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i]; 
  });

  checkResult(c, N);
  logger.wait_all();

//----------------------------------------------------------------------------//
// RAJA::omp_parallel_for_dynamic_exec policy execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_for_dynamic_exec (chunksize = 16) vector addition...\n";

  RAJA::forall<RAJA::omp_parallel_for_dynamic_exec<16>>(host, RAJA::RangeSegment(0, N),
  [=] (int i) {
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from RAJA omp_parallel_for_dynamic_exec<16> loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i]; 
  });

  checkResult(c, N);
  logger.wait_all();
#endif



#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP) || defined(RAJA_ENABLE_SYCL)

/*
  GPU_BLOCK_SIZE - specifies the number of threads in a CUDA/HIP thread block
*/
const int GPU_BLOCK_SIZE = 256;

//----------------------------------------------------------------------------//
// RAJA::cuda/hip_exec policy execution.... 
//----------------------------------------------------------------------------//
{
  std::cout << "\n Running RAJA GPU vector addition on 2 seperate streams...\n";
#if defined(RAJA_ENABLE_CUDA)
  RAJA::resources::Cuda res_gpu1;
  RAJA::resources::Cuda res_gpu2;
  using EXEC_POLICY = RAJA::cuda_exec_async<GPU_BLOCK_SIZE>;
#elif defined(RAJA_ENABLE_HIP)
  RAJA::resources::Hip res_gpu1;
  RAJA::resources::Hip res_gpu2;
  using EXEC_POLICY = RAJA::hip_exec_async<GPU_BLOCK_SIZE>;
#elif defined(RAJA_ENABLE_SYCL)
  RAJA::resources::Sycl res_gpu1;
  RAJA::resources::Sycl res_gpu2;
  using EXEC_POLICY = RAJA::sycl_exec<GPU_BLOCK_SIZE>;
#endif
  auto gpu_logger1    = RAJA::make_message_manager(buf_sz, res_gpu1);
  auto gpu_msg_queue1 = gpu_logger1.get_queue<RAJA::mpsc_queue>(0,
    [](const my_string<128>& str, int* ptr, int idx, int value) {
      std::cout << "\n " << str.c_str() << " " << ptr << " a[" << idx << "] = " << value << "\n";
    }
  );

  auto gpu_logger2    = RAJA::make_message_manager(buf_sz, res_gpu2); 
  auto gpu_msg_queue2 = gpu_logger2.get_queue<RAJA::mpsc_queue>(0,
    [](const my_string<128>& str, int* ptr, int idx, int value) {
      std::cout << "\n " << str.c_str() << " " << ptr << " a[" << idx << "] = " << value << "\n";
    }
  );

  int* d_a1 = res_gpu1.allocate<int>(N);
  int* d_b1 = res_gpu1.allocate<int>(N);
  int* d_c1 = res_gpu1.allocate<int>(N);

  int* d_a2 = res_gpu2.allocate<int>(N);
  int* d_b2 = res_gpu2.allocate<int>(N);
  int* d_c2 = res_gpu2.allocate<int>(N);

  res_gpu1.memcpy(d_a1, a, sizeof(int)* N);
  res_gpu1.memcpy(d_b1, b, sizeof(int)* N);

  res_gpu2.memcpy(d_a2, a, sizeof(int)* N);
  res_gpu2.memcpy(d_b2, b, sizeof(int)* N);


  RAJA::forall<EXEC_POLICY>(res_gpu1, RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    if (d_a1[i] < 0) { 
      gpu_msg_queue1.try_post_message("message from GPU stream 1: pointer =", d_a1, i, d_a1[i]); 
    }
    d_c1[i] = d_a1[i] + d_b1[i]; 
  });    

  RAJA::forall<EXEC_POLICY>(res_gpu2, RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    if (d_a2[i] < 0) { 
      gpu_msg_queue2.try_post_message("message from GPU stream 2: pointer =", d_a2, i, d_a2[i]); 
    }
    d_c2[i] = d_a2[i] + d_b2[i]; 
  }); 

  res_gpu1.memcpy(c, d_c1, sizeof(int)*N );

  res_gpu2.memcpy(c_, d_c2, sizeof(int)*N );

  checkResult(c, N);
  checkResult(c_, N);

  gpu_logger1.wait_all();
  gpu_logger2.wait_all();

  res_gpu1.deallocate(d_a1);
  res_gpu1.deallocate(d_b1);
  res_gpu1.deallocate(d_c1);

  res_gpu2.deallocate(d_a2);
  res_gpu2.deallocate(d_b2);
  res_gpu2.deallocate(d_c2);
}


//----------------------------------------------------------------------------//
// RAJA::cuda/hip_exec policy with waiting event.... 
//----------------------------------------------------------------------------//
{
  std::cout << "\n Running RAJA GPU vector with dependency between two seperate streams...\n";
#if defined(RAJA_ENABLE_CUDA)
  // _raja_res_defres_start
  RAJA::resources::Cuda res_gpu1;
  RAJA::resources::Cuda res_gpu2;
  RAJA::resources::Host res_host;

  using EXEC_POLICY = RAJA::cuda_exec_async<GPU_BLOCK_SIZE>;
  // _raja_res_defres_end
#elif defined(RAJA_ENABLE_HIP)
  RAJA::resources::Hip res_gpu1;
  RAJA::resources::Hip res_gpu2;
  RAJA::resources::Host res_host;

  using EXEC_POLICY = RAJA::hip_exec_async<GPU_BLOCK_SIZE>;
#elif defined(RAJA_ENABLE_SYCL)
  RAJA::resources::Sycl res_gpu1;
  RAJA::resources::Sycl res_gpu2;
  RAJA::resources::Host res_host;

  using EXEC_POLICY = RAJA::sycl_exec<GPU_BLOCK_SIZE>;
#endif
  auto gpu_logger1    = RAJA::make_message_manager(buf_sz, res_gpu1); 
  auto gpu_msg_queue1 = gpu_logger1.get_queue<RAJA::mpsc_queue>(0,
    [](int* ptr, int idx, int value) {
      std::cout << "\n gpu stream 1: pointer (" << ptr << ") d_array1[" << idx << "] = " << value << "\n";
    }
  );

  auto gpu_logger2    = RAJA::make_message_manager(buf_sz, res_gpu2);
  auto gpu_msg_queue2 = gpu_logger2.get_queue<RAJA::mpsc_queue>(0,
    [](int* ptr, int idx, int value) {
      std::cout << "\n gpu stream 2: pointer (" << ptr << ") d_array2[" << idx << "] = " << value << "\n";
    }
  );

  // _raja_res_alloc_start
  int* d_array1 = res_gpu1.allocate<int>(N);
  int* d_array2 = res_gpu2.allocate<int>(N);
  int* h_array  = res_host.allocate<int>(N);
  // _raja_res_alloc_end

  // _raja_res_k1_start
  RAJA::forall<EXEC_POLICY>(res_gpu1, RAJA::RangeSegment(0,N),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array1[i] = i;
      gpu_msg_queue1.try_post_message(d_array1, i, d_array1[i]);
    }
  );
  // _raja_res_k1_end
   
  // Log message for stream 1 
  gpu_logger1.wait_all();   

  // _raja_res_k2_start
  RAJA::resources::Event e = RAJA::forall<EXEC_POLICY>(res_gpu2, RAJA::RangeSegment(0,N),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array2[i] = -1;
      gpu_msg_queue2.try_post_message(d_array2, i, d_array2[i]);
    }
  );
  // _raja_res_k2_end

  // _raja_res_wait_start
  res_gpu2.wait_for(&e);
  // _raja_res_wait_end

  // _raja_res_k3_start
  RAJA::forall<EXEC_POLICY>(res_gpu1, RAJA::RangeSegment(0,N),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array1[i] *= d_array2[i];
      gpu_msg_queue1.try_post_message(d_array1, i, d_array1[i]);
    }
  );
  // _raja_res_k3_end

  // Log message for stream 2
  gpu_logger2.wait_all();   
  
  // _raja_res_memcpy_start
  res_gpu1.memcpy(h_array, d_array1, sizeof(int) * N);
  // _raja_res_memcpy_end

  // Log message for stream 1 
  gpu_logger1.wait_all();   

  // _raja_res_k4_start
  bool check = true;
  RAJA::forall<RAJA::seq_exec>(res_host, RAJA::RangeSegment(0,N),
    [&check, h_array] (int i) {
      if(h_array[i] != -i) {check = false;} 
    }
  );
  // _raja_res_k4_end
  
  std::cout << "\n         result -- ";
  if (check) std::cout << "PASS\n";
  else std::cout << "FAIL\n";


  res_gpu1.deallocate(d_array1);
  res_gpu2.deallocate(d_array2);
  res_host.deallocate(h_array);

}

#endif
//
//
// Clean up.
//
  host.deallocate(a);
  host.deallocate(b);
  host.deallocate(c);

  host.deallocate(a_);
  host.deallocate(b_);
  host.deallocate(c_);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* res, int len) 
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( res[i] != i ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print result.
//
void printResult(int* res, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "result[" << i << "] = " << res[i] << std::endl;
  }
  std::cout << std::endl;
}
