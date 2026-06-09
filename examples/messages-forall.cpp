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
 *  RAJA::messages example
 *
 *  The purpose of this example to show how RAJA::messages can be used.
 *  For this example, the kernels will be launched using RAJA::forall 
 *  along with a vector addition calculation. This will show how messages
 *  can be used:
 *  - to store messages on various execution policies (serial, OpenMP, GPU)
 *  - to print to a file on the GPU using the RAJA::messages
 *  - to interact with multiple GPU streams
 *  - to create custom types that can be stored (note: these types should be
 *    trivially destructible and trivially copyable)
 *
 */

// This is a simplified example fixed length string to show how
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

  std::cout << "\n\nRAJA RAJA::messages with vector addition example...\n";

  RAJA::resources::Host res_host{};

//
// Define number of messages that can be stored
//
  const std::size_t num_messages = 1;
  const std::size_t message_sz   = RAJA::align_sz(sizeof(RAJA::msg_header)) + 
    RAJA::align_sz(sizeof(RAJA::msg_args<my_string<128>, int*, int, int>));

  const std::size_t buf_sz       = num_messages*message_sz;

//
// Allocate and initialize message handler and queue
//
// _raja_msg_manager_start
  auto msg_manager = RAJA::make_message_manager(buf_sz, res_host);
// _raja_msg_manager_end

// _raja_msg_subscribe_start
  auto cpu_msg_queue = msg_manager.subscribe<RAJA::mpsc_queue>(
    [](const my_string<128>& str, int* ptr, int idx, int value) {
      std::cout << "\n " << str.c_str() << " " << ptr << " a[" << idx << "] = " << value << "\n";
    }
  );

  auto err_callback = [] (const my_string<128>& str, int* ptr, int idx, int value) {
      std::cerr << "\n " << str.c_str() << " " << ptr << " a[" << idx << "] = " << value << "\n";
    };
  msg_manager.subscribe(cpu_msg_queue.get_id(), err_callback);
// _raja_msg_subscribe_end


// _raja_msg_unsubscribe_start
  msg_manager.unsubscribe(cpu_msg_queue.get_id(), err_callback);
// _raja_msg_unsubscribe_end
  
  constexpr int N = 100000;

  int *a = res_host.allocate<int>(N);
  int *b = res_host.allocate<int>(N);
  int *c = res_host.allocate<int>(N);

  int *a_ = res_host.allocate<int>(N);
  int *b_ = res_host.allocate<int>(N);
  int *c_ = res_host.allocate<int>(N);


  for (int i = 0; i < N; ++i) {
    a[i] = -i;
    b[i] = 2 * i;
    a_[i] = -i;
    b_[i] = 2 * i;

  }


//----------------------------------------------------------------------------//

  std::cout << "\n Running C-style vector addition...\n";

// _raja_msg_k1_start
  for (int i = 0; i < N; ++i) {
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from C-style loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i];
  }
// _raja_msg_k1_end

  checkResult(c, N);
// _raja_msg_wait_start
  msg_manager.wait_all();
// _raja_msg_wait_end


//----------------------------------------------------------------------------//
// RAJA::seq_exec policy enforces sequential execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA sequential vector addition...\n";


// _raja_msg_k2_start
  RAJA::forall<RAJA::seq_exec>(res_host, RAJA::RangeSegment(0, N), [=] (int i) { 
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from RAJA seq_exec loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i]; 
  });
// _raja_msg_k2_end

  checkResult(c, N);
  msg_manager.wait_all();

#if defined(RAJA_ENABLE_OPENMP)
//----------------------------------------------------------------------------//
// RAJA::omp_for_parallel_exec policy execution.... 
//----------------------------------------------------------------------------//

  std::cout << "\n Running RAJA omp_parallel_for_exec vector addition...\n";

  RAJA::forall<RAJA::omp_parallel_for_exec>(res_host, RAJA::RangeSegment(0, N),
  [=] (int i) {
    if (a[i] < 0) { 
      cpu_msg_queue.try_post_message("message from RAJA omp_parallel_for_exec loop", a, i, a[i]); 
    }
    c[i] = a[i] + b[i]; 
  });

  checkResult(c, N);
  msg_manager.wait_all();
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
  std::cout << "\n Running RAJA GPU vector addition on RAJA's default streams...\n";
#if defined(RAJA_ENABLE_CUDA)
  using gpu_policy = RAJA::cuda_exec_async<GPU_BLOCK_SIZE>;
#elif defined(RAJA_ENABLE_HIP)
  using gpu_policy = RAJA::hip_exec_async<GPU_BLOCK_SIZE>;
#elif defined(RAJA_ENABLE_SYCL)
  using gpu_policy = RAJA::sycl_exec<GPU_BLOCK_SIZE>;
#endif
  auto res  = RAJA::resources::get_default_resource<gpu_policy>(); 

  int* d_a1 = res.allocate<int>(N);
  int* d_b1 = res.allocate<int>(N);
  int* d_c1 = res.allocate<int>(N);

  res.memcpy(d_a1, a, sizeof(int)* N);
  res.memcpy(d_b1, b, sizeof(int)* N);

  // _raja_msg_gpu1_start
  auto msg_manager = RAJA::make_message_manager<gpu_policy>(message_sz*10);

  auto log = [](const my_string<32>& str, int idx, int value) {
    std::cout << "[INFO]: " << str.c_str() << "[" << idx << "] = " << value << "\n";
  };

  // Create two types of messages:
  // queue1 stores one message type and prints with one callback
  // queue2 stores the other types of messages and forwards the message to multiple callbacks
  auto msg_queue1 = msg_manager.subscribe<RAJA::mpsc_queue>(log);
  msg_manager.subscribe(msg_queue1.get_id(),
    [](const my_string<32>& str, int idx, int value) {
      std::cout << "echo msg: " << str.c_str() << "[" << idx << "] = " << value << "\n";
    }
  );
  auto msg_queue2 = msg_manager.subscribe<RAJA::mpsc_queue>(log);

  RAJA::forall<gpu_policy>(RAJA::RangeSegment(0, N), 
    [=] RAJA_DEVICE (int i) { 
    if (d_a1[i] < 0 && i == 1) { 
      msg_queue1.try_post_message("d_a1", i, d_a1[i]); 
    }
    if (d_b1[i] > 0 && i == 1) { 
      msg_queue2.try_post_message("d_b1", i, d_b1[i]); 
    }
    d_c1[i] = d_a1[i] + d_b1[i]; 
  });    
  msg_manager.wait_all();
  // _raja_msg_gpu1_end

  res.memcpy(c, d_c1, sizeof(int)*N );
  res.wait();

  res.deallocate(d_a1);
  res.deallocate(d_b1);
  res.deallocate(d_c1);
  
  checkResult(c, N);
}


//----------------------------------------------------------------------------//
// RAJA::cuda/hip_exec policy with waiting event.... 
//----------------------------------------------------------------------------//
{
  std::cout << "\n Running RAJA GPU vector with dependency between two seperate streams...\n";
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

  int* d_array1 = res_gpu1.allocate<int>(N);
  int* d_array2 = res_gpu2.allocate<int>(N);
  int* h_array  = res_host.allocate<int>(N);

  // _raja_msg_gpu2_start
  auto allocator1     = RAJA::ResourceAllocator<char, decltype(res_gpu1)>{res_gpu1};
  auto gpu_logger1    = RAJA::make_message_manager(buf_sz, res_gpu1, allocator1); 
  auto gpu_msg_queue1 = gpu_logger1.subscribe<RAJA::mpsc_queue>(
    [](int* ptr, int idx, int value) {
      std::cout << "\n gpu stream 1: pointer (" << ptr << ") d_array1[" << idx << "] = " << value << "\n";
    }
  );

  auto allocator2     = RAJA::ResourceAllocator<char, decltype(res_gpu2)>{res_gpu2};
  auto gpu_logger2    = RAJA::make_message_manager(buf_sz, res_gpu2, allocator2);
  auto gpu_msg_queue2 = gpu_logger2.subscribe<RAJA::mpsc_queue>(
    [](int* ptr, int idx, int value) {
      std::cout << "\n gpu stream 2: pointer (" << ptr << ") d_array2[" << idx << "] = " << value << "\n";
    }
  );

  RAJA::forall<EXEC_POLICY>(res_gpu1, RAJA::RangeSegment(0,N),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array1[i] = i;
      gpu_msg_queue1.try_post_message(d_array1, i, d_array1[i]);
    }
  );
   
  // Log message for stream 1 
  gpu_logger1.wait_all();   

  RAJA::forall<EXEC_POLICY>(res_gpu2, RAJA::RangeSegment(0,N),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array2[i] = -1;
      gpu_msg_queue2.try_post_message(d_array2, i, d_array2[i]);
    }
  );

  RAJA::forall<EXEC_POLICY>(res_gpu1, RAJA::RangeSegment(0,N),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array1[i] *= -1;
      gpu_msg_queue1.try_post_message(d_array1, i, d_array1[i]);
    }
  );

  // Log message for stream 2
  gpu_logger2.wait_all();   

  // Log message for stream 1 
  gpu_logger1.wait_all();   
  // _raja_msg_gpu2_end
  
  res_gpu1.memcpy(h_array, d_array1, sizeof(int) * N);
  res_gpu1.wait();

  res_gpu1.deallocate(d_array1);
  res_gpu2.deallocate(d_array2);

  bool check = true;
  RAJA::forall<RAJA::seq_exec>(res_host, RAJA::RangeSegment(0,N),
    [&check, h_array] (int i) {
      if(h_array[i] != -i) {check = false;} 
    }
  );
  res_host.deallocate(h_array);
  
  std::cout << "\n         result -- ";
  if (check) std::cout << "PASS\n";
  else std::cout << "FAIL\n";
}

#endif
//
//
// Clean up.
//
  res_host.deallocate(a);
  res_host.deallocate(b);
  res_host.deallocate(c);

  res_host.deallocate(a_);
  res_host.deallocate(b_);
  res_host.deallocate(c_);

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
