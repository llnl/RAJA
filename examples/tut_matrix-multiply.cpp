//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cmath>

#include "RAJA/RAJA.hpp"

/*
 *  Matrix Multiplication Example
 *
 *  Example computes the product of two square matrices and introduces
 *  RAJA nested loop capabilities via a sequence of implementations.
 *
 *  RAJA features shown:
 *    - Index range segment
 *    - View abstraction
 *    - Basic usage of 'RAJA::kernel' abstractions for nested loops
 *    - Collapsing loops under OpenMP and CUDA policies
 *    - Specifying lambda arguments through statements
 *
 * If CUDA is enabled, CUDA unified memory is used.
 */

/*
  Define number of threads in x and y dimensions of a CUDA thread block
*/
#if defined(RAJA_ENABLE_CUDA)
#define CUDA_BLOCK_SIZE 16
#endif

#if defined(RAJA_ENABLE_HIP)
#define HIP_BLOCK_SIZE 16
#endif

//
// Define dimensionality of matrices.
//
const int DIM = 2;

//
// Define macros to simplify row-col indexing (non-RAJA implementations only)
//
// _matmult_macros_start
#define A(r, c) A[c + N * r]
#define B(r, c) B[c + N * r]
#define C(r, c) C[c + N * r]
// _matmult_macros_end

/*
  Define CUDA matrix multiplication kernel for comparison to RAJA version
*/
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
__global__ void matMultKernel(int N, double* C, double* A, double* B)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( row < N && col < N ) {
    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += A(row, k) * B(k, col);
    }

    C(row, col) = dot;
  }
}
#endif

//
// Functions for checking results
//
template <typename T>
void checkResult(T *C, int N);

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);

//
// Functions for printing results
//
template <typename T>
void printResult(T *C, int N);

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N);


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  std::cout << "\n\nRAJA matrix multiplication example...\n";

//
// Define num rows/cols in matrix
//
  const int N = 1000;
//const int N = CUDA_BLOCK_SIZE * CUDA_BLOCK_SIZE;

//
// Allocate and initialize matrix data.
//
  RAJA::resources::Host host_res;
  double *A = host_res.allocate<double>(N * N);
  double *B = host_res.allocate<double>(N * N);
  double *C = host_res.allocate<double>(N * N);

  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      A(row, col) = row;
      B(row, col) = col;
    }
  }

//----------------------------------------------------------------------------//

  std::cout << "\n Running C-version of matrix multiplication...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_cstyle_start
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += A(row, k) * B(k, col);
      }
      C(row, col) = dot;

    }
  }
  // _matmult_cstyle_end

  checkResult<double>(C, N);
//printResult<double>(C, N);


//----------------------------------------------------------------------------//

//
// We define RAJA range segments to define the ranges of
// row, column, and dot-product loops for RAJA variants
//
  // _matmult_ranges_start
  RAJA::TypedRangeSegment<int> row_range(0, N);
  RAJA::TypedRangeSegment<int> col_range(0, N);
  RAJA::TypedRangeSegment<int> dot_range(0, N);
  // _matmult_ranges_end

//----------------------------------------------------------------------------//

//
// For the RAJA implementations of matrix multiplication, we
// use RAJA 'View' objects to access the matrix data. A RAJA view
// holds a pointer to a data array and enables multi-dimensional indexing
// into that data, similar to the macros we defined above.
//
  // _matmult_views_start
  RAJA::View<double, RAJA::Layout<DIM>> Aview(A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Bview(B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> Cview(C, N, N);
  // _matmult_views_end

//----------------------------------------------------------------------------//

//
// In the next few examples, we show ways that we can use RAJA::forall
// statements for the matrix multiplication kernel. This usage is not
// recommended for performance reasons. Specifically, it limits the amount
// of parallelism that can be exposed to less than is possible. We show
// this usage here, to make this point clear. Later in this file, we
// introduce RAJA nested loop abstractions and show that we can extract all
// available parallelism.
//
//
// In the first RAJA implementation, we replace the outer 'row' loop
// with a RAJA::forall statement. The lambda expression contains the
// inner loops.
//

//----------------------------------------------------------------------------//

  std::cout << "\n Running sequential mat-mult (RAJA-row)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_outerforall_start
  RAJA::forall<RAJA::seq_exec>( row_range, [=](int row) {

    for (int col = 0; col < N; ++col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      Cview(row, col) = dot;

    }

  });
  // _matmult_outerforall_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

//
// Next, we replace the outer 'row' loop and the inner 'col' loop
// with RAJA::forall statements. This will also work with parallel
// execution policies, such as OpenMP and CUDA, with caveats and
// restrictions.
//
// However, nesting RAJA::forall calls like this is not recommended as
// it limits the ability to expose parallelism and flexibility for
// implementation alternatives.
//

  std::cout << "\n Running sequential mat-mult (RAJA-row, RAJA-col)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_nestedforall_start
  RAJA::forall<RAJA::seq_exec>( row_range, [=](int row) {

    RAJA::forall<RAJA::seq_exec>( col_range, [=](int col) {

      double dot = 0.0;
      for (int k = 0; k < N; ++k) {
        dot += Aview(row, k) * Bview(k, col);
      }
      Cview(row, col) = dot;

    });

  });
  // _matmult_nestedforall_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

//
// Next, we use a RAJA::kernel method to execute the kernel. These examples,
// illustrate the basic kernel interface and mechanics. The execution policies
// express the outer row and col loops using the RAJA kernel interface. Later,
// in this file we show some more complex policy examples where we express all
// three loops using the kernel interface and use additional kernel features.
//
// This is different than RAJA::forall and so a few points of exmplanation
// are in order:
//
// 1) A range and lambda index argument are required for each level in
//    the loop nest. Here, we have two of each since we have a doubly-nested
//    loop.
// 2) A range for each loop nest level is specified in a RAJA tuple object.
//    The order of ranges in the tuple must match the order of args to the
//    lambda for this to be correct, in general. RAJA provides strongly-typed
//    indices to help with this. However, this example does not use them.
// 3) An execution policy is required for each level in the loop nest. These
//    are specified in the 'RAJA::statement::For' templates in the
//    'RAJA::KernelPolicy type.
// 4) The loop nest ordering is specified in the nested execution policy --
//    the first 'For' policy is the outermost loop, the second 'For' policy
//    is the loop nested inside the outermost loop, and so on.
// 5) The integer values that are the first template arguments to the policies
//    indicate which range/lambda argument, the policy applies to.
//

  std::cout << "\n Running sequential mat-mult (RAJA-nested)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_basickernel_start
  using EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,    // row
        RAJA::statement::For<0, RAJA::seq_exec,  // col
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<EXEC_POL>(RAJA::make_tuple(col_range, row_range),
    [=](int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
    Cview(row, col) = dot;

  });
  // _matmult_basickernel_end

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)
  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp outer)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_ompkernel_start
  using EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec,  // row
        RAJA::statement::For<0, RAJA::seq_exec,            // col
          RAJA::statement::Lambda<0>
        >
      >
    >;
  // _matmult_ompkernel_end

  RAJA::kernel<EXEC_POL1>(RAJA::make_tuple(col_range, row_range),
    [=](int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
    Cview(row, col) = dot;

  });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - omp inner)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // Swapping the template arguments in this nested policy swaps the loop
  // nest ordering so the col loop is on the outside and the row loop is
  // nested within it. The execution policies on each loop remain the same
  // as the previous implementation; i.e., col (outer) iterations run
  // sequentially, while row (inner) iterations execute in parallel.
  //
  // _matmult_ompkernel_swap_start
  using EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::seq_exec,                  // col
        RAJA::statement::For<1, RAJA::omp_parallel_for_exec,    // row
          RAJA::statement::Lambda<0>
        >
      >
    >;
  // _matmult_ompkernel_swap_end

  RAJA::kernel<EXEC_POL2>( RAJA::make_tuple(col_range, row_range),
    [=](int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
    Cview(row, col) = dot;

  });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running OpenMP mat-mult (RAJA-nested - collapse)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy collapses the row and col loops in an OpenMP parallel region.
  // This is the same as using an OpenMP 'parallel for' directive on the
  // outer loop with a 'collapse(2) clause.
  //
  using EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<1, 0>,   // row, col
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<EXEC_POL3>(RAJA::make_tuple(col_range, row_range),
    [=](int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
    Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_OPENMP

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA mat-mult (RAJA-nested - POL4)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy replaces the loop nest with a single CUDA kernel launch
  // (kernel body is the lambda loop body) where the row indices are
  // assigned to thread blocks and the col indices are assigned to
  // threads within each block.
  //
  // This is equivalent to launching a CUDA kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
  //
  using EXEC_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop,
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL4>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
    Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA tiled mat-mult (RAJA-POL5)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy collapses the col and row loops into a single CUDA kernel
  // using two-dimensional CUDA thread blocks with x and y dimensions defined
  // by CUDA_BLOCK_SIZE arguments.
  //
  // When the matrix dimension N is an integer multiple of CUDA_BLOCK_SIZE,
  // the CUDA grid and thread dimension kernel launch parameters will be the
  // same as in this kernel and the one above.
  //
  using EXEC_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::cuda_block_x_loop,
            RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
              RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL5>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += Aview(row, k) * Bview(k, col);
    }
    Cview(row, col) = dot;

  });
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)
  using ResourceType = RAJA::resources::Hip;
  ResourceType res;
  double *d_A = res.allocate<double>(N * N);
  double *d_B = res.allocate<double>(N * N);
  double *d_C = res.allocate<double>(N * N);

  std::cout << "\n Running HIP mat-mult (RAJA-nested - POL4)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_A, A, N * N * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_B, B, N * N * sizeof(double), hipMemcpyHostToDevice);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  RAJA::View<double, RAJA::Layout<DIM>> d_Aview(d_A, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> d_Bview(d_B, N, N);
  RAJA::View<double, RAJA::Layout<DIM>> d_Cview(d_C, N, N);

  //
  // This policy replaces the loop nest with a single HIP kernel launch
  // (kernel body is the lambda loop body) where the row indices are
  // assigned to thread blocks and the col indices are assigned to
  // threads within each block.
  //
  // This is equivalent to launching a HIP kernel with grid dimension N
  // and blocksize N; i.e., kernel<<<N, N>>> and defining row = blockIdx.x
  // and col = threadIdx.x in the kernel.
  //
  using EXEC_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<1, RAJA::hip_block_x_loop,
          RAJA::statement::For<0, RAJA::hip_thread_x_loop,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL4>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += d_Aview(row, k) * d_Bview(k, col);
    }

    d_Cview(row, col) = dot;

  });
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP tiled mat-mult (RAJA-POL5)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  //
  // This policy collapses the col and row loops into a single HIP kernel
  // using two-dimensional HIP thread blocks with x and y dimensions defined
  // by HIP_BLOCK_SIZE arguments.
  //
  // When the matrix dimension N is an integer multiple of HIP_BLOCK_SIZE,
  // the HIP grid and thread dimension kernel launch parameters will be the
  // same as in this kernel and the one above.
  //
  using EXEC_POL5 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<HIP_BLOCK_SIZE>,
                                 RAJA::hip_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<HIP_BLOCK_SIZE>,
                                   RAJA::hip_block_x_loop,
            RAJA::statement::For<1, RAJA::hip_thread_y_loop,
              RAJA::statement::For<0, RAJA::hip_thread_x_loop,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;

  RAJA::kernel<EXEC_POL5>(RAJA::make_tuple(col_range, row_range),
    [=] RAJA_DEVICE (int col, int row) {

    double dot = 0.0;
    for (int k = 0; k < N; ++k) {
      dot += d_Aview(row, k) * d_Bview(k, col);
    }

    d_Cview(row, col) = dot;

  });
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_HIP

//----------------------------------------------------------------------------//

//
// The following examples use execution policies to express the outer row and
// col loops as well as the inner dot product loop using the RAJA kernel
// interface. They show some more complex policy examples and use additional
// kernel features.
//

  std::cout << "\n Running sequential mat-mult with multiple lambdas (RAJA-POL6a)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  //
  // This policy executes the col, row and k (inner dot product) loops
  // sequentially using a triply-nested loop execution policy and three
  // lambda expressions that
  //    -- initialize the dot product variable,
  //    -- define the 'k' inner loop row-col dot product body, and
  //    -- store the computed row-col dot product in the proper location
  //       in the result matrix.
  //
  // Note that we also pass the scalar dot product variable into each lambda
  // via a single value tuple parameter. This enables the same variable to be
  // by all three lambdas.
  //
  // _matmult_3lambdakernel_seq_start
  using EXEC_POL6a =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0, RAJA::Params<0>>,  // dot = 0.0
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<1> // inner loop: dot += ...
          >,
          RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>   // set C(row, col) = dot
        >
      >
    >;

  RAJA::kernel_param<EXEC_POL6a>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] (int col, int row, double& dot) {
       Cview(row, col) = dot;
    }

  );
  // _matmult_3lambdakernel_seq_end

  checkResult<double>(Cview, N);
  //printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::memset(C, 0, N*N * sizeof(double));

//
// The following examples uses an extension of the lambda statement
// to specify lambda arguments. By specifying arguments within statements
// we remove the requirement that lambdas require all of the tuple contents.
//

  std::cout << "\n Running sequential mat-mult with multiple lambdas - lambda args in statements (RAJA-POL6b)...\n";

  // _matmult_3lambdakernel_args_seq_start
  // Alias for convenience
  using RAJA::Segs;
  using RAJA::Params;

  using EXEC_POL6b =
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::seq_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0, Params<0>>,  // dot = 0.0
          RAJA::statement::For<2, RAJA::seq_exec,
            RAJA::statement::Lambda<1, Segs<0,1,2>, Params<0>> // dot += ...
          >,
          RAJA::statement::Lambda<2, Segs<0,1>, Params<0>>  // C(row, col) = dot
        >
      >
    >;

  RAJA::kernel_param<EXEC_POL6b>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] (int col, int row, double& dot) {
       Cview(row, col) = dot;
    }

  );
  // _matmult_3lambdakernel_args_seq_end

  checkResult<double>(Cview, N);
  //printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::cout << "\n Running OpenMP mat-mult with multiple lambdas and loop collapse (RAJA-POL7)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_3lambdakernel_ompcollapse_start
  using EXEC_POL7 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<1, 0>,   // row, col
        RAJA::statement::Lambda<0, RAJA::Params<0>>,  // dot = 0.0
        RAJA::statement::For<2, RAJA::seq_exec,
          RAJA::statement::Lambda<1> // inner loop: dot += ...
        >,
        RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>   // set C(row, col) = dot
      >
    >;
  // _matmult_3lambdakernel_ompcollapse_end

  RAJA::kernel_param<EXEC_POL7>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] (int col, int row, double& dot) {
       Cview(row, col) = dot;
    }

  );

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_OPENMP

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA mat-mult with multiple lambdas (RAJA-POL8)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_3lambdakernel_cuda_start
  using EXEC_POL8 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::For<1, RAJA::cuda_block_x_loop,    // row
          RAJA::statement::For<0, RAJA::cuda_thread_x_loop, // col
            RAJA::statement::Lambda<0, RAJA::Params<0>>,    // dot = 0.0
            RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1>                  // dot += ...
            >,
            RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>   // set C = ...
          >
        >
      >
    >;
  // _matmult_3lambdakernel_cuda_end

  RAJA::kernel_param<EXEC_POL8>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row, double& dot) {
       Cview(row, col) = dot;
    }

  );

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA mat-mult with multiple lambdas (RAJA-POL9a)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // _matmult_3lambdakernel_cudatiled_start
  using EXEC_POL9a =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<CUDA_BLOCK_SIZE>,
                                 RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<CUDA_BLOCK_SIZE>,
                                   RAJA::cuda_block_x_loop,
            RAJA::statement::For<1, RAJA::cuda_thread_y_loop,   // row
              RAJA::statement::For<0, RAJA::cuda_thread_x_loop, // col
                RAJA::statement::Lambda<0, RAJA::Params<0>>,    // dot = 0.0
                RAJA::statement::For<2, RAJA::seq_exec,
                    RAJA::statement::Lambda<1>                 // dot += ...
                >,
                RAJA::statement::Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>   // set C = ...
              >
            >
          >
        >
      >
    >;
  // _matmult_3lambdakernel_cudatiled_end

  RAJA::kernel_param<EXEC_POL9a>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row,  double& dot) {
       Cview(row, col) = dot;
    }

  );

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running CUDA mat-mult with multiple lambdas - lambda args in statements (RAJA-POL9b)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  using EXEC_POL9b =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<CUDA_BLOCK_SIZE>,
                                 RAJA::cuda_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<CUDA_BLOCK_SIZE>,
                                   RAJA::cuda_block_x_loop,
            RAJA::statement::For<1, RAJA::cuda_thread_y_loop, // row
              RAJA::statement::For<0, RAJA::cuda_thread_x_loop, // col
                RAJA::statement::Lambda<0, Params<0>>,  // dot = 0.0
                RAJA::statement::For<2, RAJA::seq_exec,
                  RAJA::statement::Lambda<1, Segs<0,1,2>, Params<0>> // dot += ...
                >,
                  RAJA::statement::Lambda<2, Segs<0,1>, Params<0>>   // set C = ...
              >
            >
          >
        >
      >
    >;

  RAJA::kernel_param<EXEC_POL9b>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row, double& dot) {
       Cview(row, col) = dot;
    }

  );

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running  mat-mult with tiling + shared memory...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // This example builds on the RAJA tiling capabilities presented earlier
  // and uses RAJA LocalArray's to load tiles of the global matrix
  // and perform matrix-matrix multiplication within the tiles.

  // This example illustrates using CUDA shared memory, and thread
  // synchronization. We recommend viewing tut_matrix-transpose-local-array.cpp
  // for an introduction to RAJA LocalArray types and thread synchronization.

  using Shmem      = RAJA::LocalArray<double, RAJA::PERM_IJ, RAJA::SizeList<CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>;

  using shmem_Lambda0 = RAJA::statement::Lambda<0, RAJA::Offsets<0, 2>, RAJA::Params<2>>;
  using shmem_Lambda1 = RAJA::statement::Lambda<1, RAJA::Segs<0, 1>, RAJA::Offsets<0, 1>, RAJA::Params<0>>;
  using shmem_Lambda2 = RAJA::statement::Lambda<2, RAJA::Segs<1, 2>, RAJA::Offsets<1, 2>, RAJA::Params<1>>;
  using shmem_Lambda3 = RAJA::statement::Lambda<3, RAJA::Offsets<0, 1, 2>, RAJA::Params<0, 1, 2>>;
  using shmem_Lambda4 = RAJA::statement::Lambda<4, RAJA::Segs<0, 2>, RAJA::Offsets<0, 2>, RAJA::Params<2>>;

  using EXEC_POL10 =
    RAJA::KernelPolicy<
      RAJA::statement::CudaKernelFixed<CUDA_BLOCK_SIZE*CUDA_BLOCK_SIZE,
        //Initalize thread private value
        RAJA::statement::InitLocalMem<RAJA::cuda_shared_mem, RAJA::ParamList<2,1,0>,

          // Tile rows and cols of C (the result matrix C)
          RAJA::statement::Tile<0, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::cuda_block_x_direct,
            RAJA::statement::Tile<2, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::cuda_block_y_direct,

            // zero out shmem tile of C
            RAJA::statement::For<2, RAJA::cuda_thread_y_loop,
              RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                shmem_Lambda0 > >,

                // Slide window across matrix: Load tiles of global matrices A, B and compute
                // local dot products
                RAJA::statement::Tile<1, RAJA::tile_fixed<CUDA_BLOCK_SIZE>, RAJA::seq_exec,

                  // Load tile of A into shmem
                  RAJA::statement::For<1, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                      shmem_Lambda1
                    >
                   >,

                  // Load tile of B into shmem
                  RAJA::statement::For<2, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<1, RAJA::cuda_thread_x_loop,
                      shmem_Lambda2
                    >
                  >,

                  RAJA::statement::CudaSyncThreads,

                  //Partial multiplication
                  RAJA::statement::For<2, RAJA::cuda_thread_y_loop,
                    RAJA::statement::For<1, RAJA::seq_exec,
                      RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                        shmem_Lambda3
                      >
                    >
                  >,

                  RAJA::statement::CudaSyncThreads
                >, //sliding window

               //Write memory out to global matrix
               RAJA::statement::For<2, RAJA::cuda_thread_y_loop,
                RAJA::statement::For<0, RAJA::cuda_thread_x_loop,
                shmem_Lambda4 > >
             >
            >
           > //Create shared memory
         >//Cuda kernel
        >;

    Shmem aShared, bShared, cShared;

    RAJA::kernel_param<EXEC_POL10>(
      RAJA::make_tuple(RAJA::TypedRangeSegment<int>(0, N),
                       RAJA::TypedRangeSegment<int>(0, N),
                       RAJA::TypedRangeSegment<int>(0, N)),
      RAJA::make_tuple(aShared, bShared, cShared),

    // Zero out thread local memory for storing dot products
    [=] RAJA_HOST_DEVICE (int tn, int tp, Shmem &cShared) {

      cShared(tn,tp) = 0.0;

    },

    // Load tile of A
    [=] RAJA_HOST_DEVICE (int n, int m, int tn, int tm, Shmem &aShared) {

      aShared(tn, tm) = Aview(n, m);

    },

    // Load tile of B
    [=] RAJA_HOST_DEVICE (int m, int p, int tm, int tp, Shmem &bShared) {

      bShared(tm, tp) = Bview(m, p);

    },

    // Do partial update in shmem
    [=] RAJA_HOST_DEVICE (int tn, int tm, int tp, Shmem &aShared,  Shmem &bShared, Shmem & cShared) {

      cShared(tn,tp) += aShared(tn,tm) * bShared(tm, tp);

    },

    // Write out complete result
    [=] RAJA_HOST_DEVICE (int n, int p, int tn, int tp,  Shmem &cShared) {

      Cview(n,p) = cShared(tn,tp);

    });

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);
#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_CUDA)

  std::cout << "\n Running CUDA tiled mat-mult (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));

  // Define thread block dimensions
  dim3 blockdim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  // Launch CUDA kernel defined near the top of this file.
  matMultKernel<<<griddim, blockdim>>>(N, C, A, B);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

#endif // if RAJA_ENABLE_CUDA

//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_HIP)

  std::cout << "\n Running HIP mat-mult with multiple lambdas (RAJA-POL8)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  // _matmult_3lambdakernel_hip_start
  using EXEC_POL8 =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::For<1, RAJA::hip_block_x_loop,    // row
          RAJA::statement::For<0, RAJA::hip_thread_x_loop, // col
            RAJA::statement::Lambda<0, RAJA::Params<0>>,   // dot = 0.0
            RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1>                 // dot += ...
            >,
            RAJA::statement::Lambda<2,
              RAJA::Segs<0,1>, RAJA::Params<0>>            // set C = ...
          >
        >
      >
    >;
  // _matmult_3lambdakernel_hip_end

  RAJA::kernel_param<EXEC_POL8>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += d_Aview(row, k) * d_Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row, double& dot) {
       d_Cview(row, col) = dot;
    }

  );

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);


  //----------------------------------------------------------------------------//

  std::cout << "\n Running HIP mat-mult with multiple lambdas - lambda args in statements (RAJA-POL9)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  // _matmult_3lambdakernel_hiptiled_start
  using EXEC_POL9b =
    RAJA::KernelPolicy<
      RAJA::statement::HipKernel<
        RAJA::statement::Tile<1, RAJA::tile_fixed<HIP_BLOCK_SIZE>,
                                 RAJA::hip_block_y_loop,
          RAJA::statement::Tile<0, RAJA::tile_fixed<HIP_BLOCK_SIZE>,
                                   RAJA::hip_block_x_loop,
            RAJA::statement::For<1, RAJA::hip_thread_y_loop,    // row
              RAJA::statement::For<0, RAJA::hip_thread_x_loop,  // col
                RAJA::statement::Lambda<0, Params<0>>,          // dot = 0.0
                RAJA::statement::For<2, RAJA::seq_exec,
                  RAJA::statement::Lambda<1, Segs<0,1,2>, Params<0>> // dot += ...
                >,
                  RAJA::statement::Lambda<2, Segs<0,1>, Params<0>>   // set C = ...
              >
            >
          >
        >
      >
    >;
 // _matmult_3lambdakernel_hiptiled_end

  RAJA::kernel_param<EXEC_POL9b>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double>{0.0},    // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE (double& dot) {
       dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE (int col, int row, int k, double& dot) {
       dot += d_Aview(row, k) * d_Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE (int col, int row, double& dot) {
       d_Cview(row, col) = dot;
    }

  );

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

//----------------------------------------------------------------------------//

  std::cout << "\n Running HIP tiled mat-mult (no RAJA)...\n";

  std::memset(C, 0, N*N * sizeof(double));
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, d_C, C, N * N * sizeof(double), hipMemcpyHostToDevice);

  // Define thread block dimensions
  dim3 blockdim(HIP_BLOCK_SIZE, HIP_BLOCK_SIZE);
  // Define grid dimensions to match the RAJA version above
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N,blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N,blockdim.y));

//printf("griddim = (%d,%d), blockdim = (%d,%d)\n", (int)griddim.x, (int)griddim.y, (int)blockdim.x, (int)blockdim.y);

  // Launch HIP kernel defined near the top of this file.
  hipLaunchKernelGGL((matMultKernel), dim3(griddim), dim3(blockdim), 0, 0, N, d_C, d_A, d_B);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, C, d_C, N * N * sizeof(double), hipMemcpyDeviceToHost);
  checkResult<double>(Cview, N);
//printResult<double>(Cview, N);

  res.deallocate(d_A);
  res.deallocate(d_B);
  res.deallocate(d_C);
#endif // if RAJA_ENABLE_HIP

//----------------------------------------------------------------------------//

//
// Clean up.
//
  host_res.deallocate(A);
  host_res.deallocate(B);
  host_res.deallocate(C);

  std::cout << "\n DONE!...\n";

  return 0;
}

//
// Functions to check result and report P/F.
//
template <typename T>
void checkResult(T* C, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( C(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

template <typename T>
void checkResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  bool match = true;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      if ( std::abs( Cview(row, col) - row * col * N ) > 10e-12 ) {
        match = false;
      }
    }
  }
  if ( match ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
};

//
// Functions to print result.
//
template <typename T>
void printResult(T* C, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << C(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}

template <typename T>
void printResult(RAJA::View<T, RAJA::Layout<DIM>> Cview, int N)
{
  std::cout << std::endl;
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      std::cout << "C(" << row << "," << col << ") = "
                << Cview(row, col) << std::endl;
    }
  }
  std::cout << std::endl;
}
