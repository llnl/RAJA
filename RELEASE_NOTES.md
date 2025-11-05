[comment]: # (#################################################################)
[comment]: # (Copyright 2016-25, Lawrence Livermore National Security, LLC)
[comment]: # (and RAJA project contributors. See the RAJA/LICENSE file)
[comment]: # (for details.)
[comment]: # 
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

Version YYYY.MM.PP -- Release date 20yy-mm-dd
============================================

This release contains ...

Notable changes include:

  * New features / API changes:

  * Build changes/improvements:

  * Bug fixes/improvements:


Version 2025.09.1 -- Release date 2025-10-01
============================================

This release contains a bugfix and a submodule update

Notable changes include:
  * CAMP submodule was updated to point at the patch release v2025.09.2
  * A build conflict with Caliper and ROCTX/NVTX has been fixed
  * BlueOS cmake scripts have been removed
  * toss3/clangcuda_6_0_0_nvcc_8_0.cmake has been removed
  * C++17 features have been used to simplify RAJA internals 
  * Missing use of const has been corrected in the CompareFirst struct methods


Version 2025.09.0 -- Release date 2025-09-12
============================================

This release contains a variety of new features, bug fixes, and build changes.

Notable changes include:

  * New features / API changes:
    * The RAJA::expt::Reduce interface now works with RAJA::kernel. There is
      an example of this usage in the RAJA/examples/kernel-reduction.cpp file.
    * Builtin atomics for unsigned fixed width integer types that support 
      Windows builds have been added.
    * Added a global function to turn on/off Caliper profiling, when enabled.
    * Added some quality-of-life improvements to RAJA MultiView construct, 
      such as the ability to construct and set a MultiView with const data,
      the ability to construct an empty MultiView, and new accessors to data and layout.
    * Added `size` methods to the RAJA IndexLayout construct. These can be
      used to check the size of the layout or to determine whether the size
      is non-zero. All the size methods directly call the base Layout
      implementation. Also added host-device decorators to the methods
      `make_tuple_index` and `make_index_layout`.
    * Added the `grid_constant` decorator to global function parameters for
      CUDA and make most global function parameters const for CUDA and HIP.
      This allows nvcc to better optimize parameter usage in some cases.
      ROCm compilers do not support this decorator and do not appear to 
      optimize use of this parameter. For more details, please see
      https://docs.nvidia.com/cuda/cuda-c-programming-guide/#grid-constant
    * Added an experimental feature to support printing of arguments to GPU
      API functions on error. This capability will continue to improve and
      mature. Hopefully, it will help RAJA users understand what went wrong
      when a GPU kernel fails. This capability is currently supported for RAJA
      CUDA and HIP back-ends.

  * Build changes/improvements:
    * **RAJA now requires C++17 as the minimum C++ standard.**
    * Updated BLT submodule to v0.7.1 release.
    * Updated Camp submodule to v2025.09.2 release.
    * Updated to NVTX3 profiling library to support CUDA 12.9 and above
      (also supports back to CUDA 10)
    * [BREAKING CHANGE] Renamed CMake option `RAJA_ENABLE_NV_TOOLS_EXT` to `RAJA_ENABLE_NVTX`
    * Updated desul submodule to 6114dd25b54782678c555c0c1d2197f13cc8d2a0
      commit.
    * The CUB and rocPRIM submodules in RAJA have been removed. Moving forward,
      the versions of these that are deployed with the CUDA and ROCm compiler
      stacks will be used.
    * Fixed the RAJA minimum architecture check, which did not work on 
      Blackwell cards. Now, if `CMAKE_CUDA_ARCHITECTURES` is not set, the 
      compiler will select a reasonable architecture default, which is 
      guaranteed to be at least `sm_35` for nvcc since RAJA now requires
      CUDA 11 as the minimum CUDA version.

  * Bug fixes/improvements:
    * Race conditions due to inconsistent usage of Camp resources (CUDA/HIP
      streams) in RAJA tests have been fixed.
    * Resolved a bunch of shadow variable warnings reported by some users.


Version 2025.03.2 -- Release date 2025-05-14
============================================

This release contains several improvements

     * Build changes/improvements:
       * Removed unused variables related to kernel naming
       * Added missing host device annotations on missing param reducers
       * CMake build option to allow for use of OpenMP 5.1 atomics for min/max
         operations. The option is on by default.
       * Full backwards compatibility of kernel naming and lambda capture
         style reducers.
       * Removed compiler warnings related to NVCC and loop unrolling


Version 2025.03.1 -- Release date 2025-04-17
============================================

This release contains one new feature and a bugfix.

Notable changes include:

  * New features / API changes:
    * Added initial support for Caliper to gather profiling data for kernels.
      See user docs and examples for configuration instructions and examples
      of usage.

  * Build changes/improvements:
    * None

  * Bug fixes/improvements:
    * Fix header file include issue when vectorization enabled in a HIP build.


Version 2025.03.0 -- Release date 2025-03-17
============================================

This release contains new features, bug fixes, and updates to submodule
dependencies.

Notable changes include:

  * New features / API changes:
    * Added improved support for perfectly nested loops in RAJA::launch.
    * Added helper methods to simplify the creation of RAJA View objects
      with permutations of stride ordering. Examples and user docs have also
      been added. 
    * Added GPU policies for CUDA and HIP that do not check loop bounds when
      they do not need to be checked in a kernel. This can help improve
      performance by up to 5%. The new policies are documented in the RAJA
      user guide and include `direct_unchecked` in their names.
    * Refactored the new (experimental) RAJA reduction interface to have 
      consistent min/max/loc operator semantics and added type safety to 
      reduce erroneous usage. Changes are described in the RAJA User Guide.
    * Added support for new RAJA reduction interface to RAJA::dynamic_forall 
      and pulled dynamic_forall out of RAJA `expt` namespace.
    * Added `RAJA_HIP_WAVESIZE` CMake option to set the wave size for HIP
      builds. It defaults to 64 but can be set to 32, for example, to 
      build RAJA to run on Radeon gaming cards.

  * Build changes/improvements:
     * Update BLT to v0.7.0 release.
     * Update camp submodule to v2025.03.0 release.
     * Update desul submodule to 6114dd25b54782678c555c0c1d2197f13cc8d2a0
       commit.
     * Added clang-format CI check (clang 14) that must pass before a PR can
       be merged -- noted here so external contributors are aware.

  * Bug fixes/improvements:
    * Resolved undefined behavior related to constructing 
      uniform_int_distribution with min > max. This was causing some Windows
      tests to fail.
    * Corrected call to wrong global function when using a fixed CUDA policy
      and reductions in RAJA::launch kernel -- potential performance issue.
    * Fixed memory leak in RAJA::launch OpenMP back-end.
    * Added missing host-device decorations to some math utility functions.
    * Fixed MSVC compilation failures with 64-bit intrinsics in x86 Windows 
      builds.
    * Fixed issue so that a kernel will no longer be launched when there is no
      work for it to do; i.e., no active iteration space entries.
    * Removed invalid C++ usage in implementation of RAJA::kernel `initLocalMem`
      statement, which was causing large warning messages during compilation.


Version 2024.07.0 -- Release date 2024-07-24
============================================

This release contains new features, improvements, and bugfixes.

Notable changes include:

  * New features / API changes:
     * Added support for a "multi-reduction" operation which allows users to
       perform a run time-defined number of reduction operations in a kernel.
       Please see the RAJA User Guide for details and examples.
     * Added first couple of sections for a "RAJA Cookbook" in the RAJA User
       Guide. The goal is to provide users with more detailed guidance about
       using RAJA features, choosing execution policies, etc. Additional
       content will be provided in future releases.
     * Added atomicLoad and atomicStore routines for correctness in some
       use cases.
     * Added OpenMP 5.1 implementations for atomicMin and atomicMax.
     * Add SYCL reduction support in RAJA::launch

  * Build changes/improvements:
     * Update camp submodule to v2024.07.0 release. This will be a version
       constraint for this release in RAJA Spack package.
     * Minimum required CMake version bumped to 3.23.

  * Bug fixes/improvements:
     * Fix CMake issue for case when RAJA is used as a submodule dependency.
     * Various fixes and improvements to builtin atomic support.
     * Fixes and improvements to other atomic operations:
        * Modified HIP and CUDA generic atomic compare and swap algorithms
          to use atomic loads instead of relying on volatile.
        * Re-implemented atomic loads in terms of builtin atomics for CUDA
          and HIP so that the generic compare and swap functions can use it.
        * Removes volatile qualifier in atomic function signatures.
        * Use cuda::atomic_ref in newer versions of CUDA to back 
          atomicLoad/atomicStore.
        * Use atomicAdd as a fallback for atomicSub in CUDA.
        * Removed checks where __CUDA_ARCH__ is less than 350 since RAJA 
          requires that as the minimum supported architecture (CMake check).
     * Fixed issues with naming RAJA forall::kernels when using CUDA.
     * Fixes in SYCL back-end for RAJA::launch.
     * Fixed some issues in examples.
     * Bugfixes and cleanup in parts of the SYCL back-end needed to
       support a bunch of new SYCL kernels that will appear in 
       RAJA Performance Suite release.
     * Fix type naming issue that was exposed with a new version of the
       Intel oneAPI compiler.
     * Fix issue in User Guide documentation for configuring a project
       using RAJA CMake configuration.


Version 2024.02.2 -- Release date 2024-05-08
============================================

This release contains a bugfix and new execution policies that improve
performance for GPU kernels with reductions.

Notable changes include:

  * New features / API changes:
     * New GPU execution policies for CUDA and HIP added which provide
       improved performance for GPU kernels with reductions. Please see the 
       RAJA User Guide for more information. Short summary:
         * Option added to change max grid size in policies that use the
           occupancy calculator.
         * Policies added to run with max occupancy, a fraction of of the
           max occupancy, and to run with a "concretizer" which allows a 
           user to determine how to run based on what the occupancy 
           calculator determines about a kernel.
         * Additional options to tune kernels containing reductions, such as
             * an option to initialize data on host for reductions that use
               atomic operations
             * an option to avoid device scope memory fences 
     * Change ordering of SYCL thread index ordering in RAJA::launch to 
       follow the SYCL "row-major" convention. Please see RAJA User Guide
       for more information.

  * Build changes/improvements:
     * NONE.

  * Bug fixes/improvements:
     * Fixed issue in bump-style allocator used internally in RAJA::launch.


Version 2024.02.1 -- Release date 2024-04-03
============================================

This release contains submodule updates and minor RAJA improvements.

Notable changes include:

  * New features / API changes:
     * NONE.

  * Build changes/improvements:
     * Update BLT submodule to v0.6.2 release.
     * Update camp submodule to v2024.02.1 release.

  * Bug fixes/improvements:
     * Various changes to quiet compiler warnings in SYCL builds related
       to deprecated usage. 


Version 2024.02.0 -- Release date 2024-02-14
============================================

This release contains several RAJA improvements and submodule updates.

Notable changes include:

  * New features / API changes:
     * BREAKING CHANGE (ALMOST): The loop_exec and associated policies such as 
       loop_atomic, loop_reduce, etc. were deprecated in the v2023.06.0 release
       (please see the release notes for that version for details). Users
       should replace these with `seq_exec` and associated policies for
       sequential CPU execution. The code behavior will be identical to what
       you observed with `loop_exec`, etc. However, due to a request from 
       some users with special circumstances, the `loop_*` policies still
       exist in this release as type aliases to their `seq_*` analogues. The
       `loop_*` policies will be removed in a future release.
     * BREAKING CHANGE: RAJA TBB back-end support has been removed. It was 
       not feature complete and the TBB API has changed so that the code no 
       longer compiles with newer Intel compilers. Since it doesn't appear
       that anyone depends on it, we have removed it.
     * An `IndexLayout` concept was added, which allows for accessing elements
       of a RAJA `View` via a collection of indicies and use a different 
       indexing strategy along different dimensions of a multi-dimensional 
       `View`. Please the RAJA User Guide for more information.
     * Add support for SYCL reductions using the new RAJA reduction API.
     * Add support for new reduction API for all back-ends in RAJA::launch.

  * Build changes/improvements:
     * Update BLT submodule to v0.6.1 and incorporate its new macros for
       managing TPL targets in CMake.
     * Update camp submodule to v2024.02.0, which contains changes to support
       ROCm 6.x compilers. 
     * Update desul submodule to afbd448.
     * Replace internal use of HIP and CUDA platform macros to their newer
       versions to support latest compilers.

  * Bug fixes/improvements:
     * Change internal memory allocation for HIP to use coarse-grained pinned
       memory, which improves performance because it can be cached on a device.
     * Fix compilation error resulting from incorrect namespacing of OpenMP
       execution policy.
     * Several fixes to internal implementaion of Reducers and Operators.


Version 2023.06.1 -- Release date 2023-08-16
============================================

This release contains various smaller RAJA improvements. 

Notable changes include:

  * New features / API changes:
     * Add compile time block size optimization for new reduction interface.
     * Changed default stream usage for Workgroup constructs to use the 
       stream associated with the default (camp) resource. Previously, we were 
       using stream zero. Specifically, this change affects where we memset 
       memory in the zeroed device memory pool and where we get device function
       pointers for WorkGroup.  

  * Build changes/improvements:
     * RAJA_ENABLE_OPENMP_TASK CMake option added to enable/disable algorithm 
       options based on OpenMP task construct. Currently, this only applies
       to RAJA's OpenMP sort implementation. The default is 'Off'. The option
       allows users to choose a task implementation if they wish.
     * Resolve several compiler warnings.

  * Bug fixes/improvements:
     * Fix compilation of GPU occupancy calculator and use common types for 
       HIP and CUDA backends in the occupancy calculator, kernel policies, 
       and kernel launch helper routines.
     * Fix direct cudaMalloc/hipMalloc calls and memory leaks.


Version 2023.06.0 -- Release date 2023-07-06
============================================

This release contains new features to improve GPU kernel performance and some
bug fixes. It contains one breaking change described below and an execution
policy deprecation also described below. The policy deprecation is not a
breaking change in this release, but will result in a breaking change in the
next release.

Notable changes include:

  * New features / API changes:
     * In this release the loop_exec execution policy is deprecated and will
       be removed in the next release. RAJA has had two sequential execution
       policies for some time, seq_exec and loop_exec. When using the seq_exec
       execution policy, RAJA would attach #pragma novector, or similar
       depending on the compiler, to loop kernel execution. This prevented
       a compiler from vectorizing a loop, even if it was correct to do so.
       When the loop_exec policy was specified, the compiler was allowed to
       apply any optimizations, including SIMD, that its heuristics determined
       were appropriate. In this release, seq_exec behaves the same as how
       loop_exec behaved historically. The loop_exec and associated policies
       such as loop_atomic, loop_reduce, etc. are type aliases to the
       analogous seq_exec policies. This prevents breaking user code with this
       release. However, users should prepare to switch loop_exec policies
       to the seq_exec policy variants in the future.
     * GPU global (thread and block) indexing has been refactored to abstract
       indexing in a given dimension. The result is that users can now specify
       a block size or a grid size at compile time or get those values at run
       time. You can also ignore blocks and index only with threads and
       vice versa. Kernel and launch policies are now shared. Such policies are
       now multi-part and contain global indexing information, a way to map
       global indices like direct or strided loops, and have a synchronization
       requirement. The synchronization allows one to request that all threads
       complete even if some have no work so you can synchronize a block.
       Aliases have been added for all of the pre-existing policies and some
       are deprecated in favor of policies named more consistently. One
       **BREAKING CHANGE** is that thread loop policies are no longer safe to
       block synchronize. That feature still exists but can only be
       accessed with a custom policy. The RAJA User Guide contains descriptions
       of the new policy mechanics.

  * Build changes/improvements:
     * Update BLT submodule to v0.5.3
     * Update camp submodule to v2023.06.0

  * Bug fixes/improvements:
     * Fixes a Windows build issue due to macro file definition logic in a
       RAJA header file. RAJA_COMPILER_MSVC was not getting defined properly
       when building on a Windows platform using a compiler other than MSVC.
     * Kernels using the RAJA OpenMP target back-end were not properly
       seg faulting when expected to do so. This has been fixed.
     * Various improvements, compilation and execution, in RAJA SIMD support.
     * Various improvements and additions to RAJA tests to cover more end-user
       cases.


Version 2022.10.5 -- Release date 2023-02-28
============================================

This release fixes an issue that was found after the v2022.10.4 release.

  * Fixes CUDA and HIP separable compilation option that was broken before the 
    v2022.10.0 release. For the curious reader, the issue was that resources
    were constructed and calling CUDA/HIP API routines before either runtime
    was initialized.


Version 2022.10.4 -- Release date 2022-12-14
============================================

This release fixes an issue that was found after the v2022.10.3 release.

  * Fixes device alignment bug in workgroups which led to missing symbol errors
    with the AMD clang compiler.


Version 2022.10.3 -- Release date 2022-12-01
============================================

This release fixes a few issues that were found after the v2022.10.2 release.

Notable changes include:

  * Update camp submodule to v2022.10.1
  * Update BLT submodule to commit 8c229991 (includes fixes for crayftn + hip)

  * Properly export 'roctx' target when CMake variable RAJA_ENABLE_ROCTX is on. 
  * Fix CMake logic for exporting desul targets when desul atomics are enabled.
  * Fix the way we use CMake to find the rocPRIM module to follow CMake
    best practices.
  * Add missing template parameter pack argument in RAJA::statement::For
    execution policy construct used in RAJA::kernel implementation for OpenMP 
    target back-end.
  * Change to use compile-time GPU thread block size in RAJA::forall 
    implementation. This improves performance of GPU kernels, especially 
    those using the RAJA HIP back-end.
  * Added RAJA plugin support, including CHAI support, for RAJA::launch.
  * Replaced 'DEVICE' macro with alias to 'device_mem_pool_t' to prevent name 
    conflicts with other libraries.
  * Updated User Guide documentation about CMake variable used to pass 
    compiler flags for OpenMP target back-end. This changed with CMake
    minimum required version bump in v2022.10.0.
  * Adjust ordering of BLT and camp target inclusion in RAJA CMake usage to 
    fix an issue with projects using external camp vs. RAJA submodule.
    


Version 2022.10.2 -- Release date 2022-11-08
============================================

This release fixes a few issues that were found after the v2022.10.1 patch
release and updates a few things. Sorry for the churn, folks.

Notable changes include:

  * Update desul submodule to commit e4b65e00.

  * CUDA compute architecture must now be set using the 
    'CMAKE_CUDA_ARCHITECTURES' CMake variable. For example, by passing
    '-DCMAKE_CUDA_ARCHITECTURES=70' to CMake for 'sm_70' architecture. 
    Using '-DCUDA_ARCH=sm_*' will not no longer do the right thing. Please
    see the RAJA User Guide for more information.
  * A linking bug was fixed related to the usage of the new RAJA::KernelName
    capability.
  * A compilation bug was fixed in the new reduction interface support for 
    OpenMP target offload.  
  * An issue was fixed in AVX compiler checking logic for RAJA vectorization
    intrinsic capabilities.


Version 2022.10.1 -- Release date 2022-10-31
============================================

This release updates the RAJA release number in CMake, which was inadvertently
missed in the v2022.10.0 release.


Version 2022.10.0 -- Release date 2022-10-28
============================================

This release contains new features, bug fixes, and build improvements. Please
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * New features / API changes:
     * Introduced a new RAJA::forall and reduction interface that extend
       the execution behavior of reduction operations with RAJA::forall.
       The main difference with the pre-existing reduction interface in RAJA
       is that reduction variables and operations are passed into the 
       RAJA::forall method and lambda expression instead of using the lambda
       capture mechanism for reduction objects. This offers flexibility and
       potential performance advantages when using RAJA reductions as the
       new interface enables the ability to integrate with programming model 
       back-end reduction machinery directly, for OpenMP and SYCL for example.
       The interface also enables user-chosen kernel names to be passed to
       RAJA::forall for performance analysis annotations that are easier to
       understand. Example codes are included as well as a description of
       the new interface and comparison with the pre-existing interface in
       the RAJA User Guide.
     * Added support for run time execution policy selection for RAJA::forall
       kernels. Users can specify any number of execution policies in their
       code and then select which to use at run time. There is no discussion 
       of this in the RAJA User Guide yet. However, there are a couple of 
       example codes in files RAJA/examples/*dynamic-forall*.cpp.
     * The RAJA::launch framework has been moved out of the experimental 
       namespace, into the RAJA:: namespace, which introduces an API change.
     * Add support for all RAJA segment types in the RAJA::launch framework.
     * Add SYCL back-end support for RAJA::launch and dynamic shared memory
       for all back-ends in RAJA::launch. These changes introduce API changes.
     * Add additional policies to WorkGroup construct that allow for different
       methods of dispatching work.
     * Add special case implementations to CUDA atomicInc and atomicDec 
       functions to use special hardware support when available. This can
       result in a significant performance boost.
     * Rework HIP atomic implementations to support more native data types.
     * Added RAJA_UNROLL_COUNT macro which enables users to unroll loops for
       a fix unroll count.
     * Major User Guide rework:
         * New RAJA tutorial sections, including new exercise source files
           to work through. Material used in recent RADIUSS/AWS RAJA Tutorial.
         * Cleaned up and expanded RAJA feature sections to be more like a
           reference guide with links to associated tutorial sections for
           implementation examples.
         * Improved presentation of build configuration sections.

  * Build changes / improvements:
     * Submodule updates:
         * BLT updated to v0.5.2 release.
         * Camp updated to v2022.10.0 release.
     * The minimum CMake version required has changed. For a HIP build,
       CMake 3.23 or newer is required. For all other builds CMake 3.20
       or newer is required.
     * OpenMP back-end support is now off by default to match behavior of
       all other RAJA parallel back-end support. To enable OpenMP, users
       must now run CMake with the -DENABLE_OPENMP=On option.
     * Support OpenMP back-end enablement in a HIP build configuration.
     * RAJA_ENABLE_VECTORIZATION CMake option added to enable/disable
       new SIMD/SIMT vectorization support. The default is 'On'. The option
       allows users to disable if they wish.
     * Improvements to build target export mechanics coordinated with camp,
       BLT, and Spack projects.
     * Improve HIP builds to better support evolving ROCm software stack.
     * Add CMake variable RAJA_ALLOW_INCONSISTENT_OPTIONS and CMake messages
       to allow users more control when using CMake dependent options. When
       CMake is run, the code now checks for cases when RAJA_ENABLE_X=On and
       but ENABLE_X=Off. Previously, this was confusing because X would not
       be enabled despite the value of the RAJA-specific option.
     * Build system refactoring to make CMake configurations more robust; added
       test to check for installed CMake config. 
     * Added basic support to compile with C++20 standard.
     * Add missing compilation macro guards for HIP and CUDA policies in
       vectorization support when not running on a GPU device.
     * Various compiler warnings squashed. 

  * Bug fixes / improvements:
     * Expanded test coverage to catch more cases that users have run into.
     * Various fixes in SIMD/SIMT support for different compilers and versions
       users have hit recently. Also, changes to internal implementations to
       improve run time performance for those features.


Version 2022.03.1 -- Release date 2022-08-10
============================================

This release contains a bug fix.

Notable changes include:

  * Fix for guarding GPU vectorization when not running on the device.


Version 2022.03.0 -- Release date 2022-03-15
============================================

This release contains new features, bug fixes, and build improvements. Please
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * Important note: As of this release, the coordinated release of RAJA 
                    Portability Suite components (RAJA, Umpire, CHAI) will be 
                    tagged as YYYY.MM.pp for year, month, and patch number. For
                    example, This release is tagged as 2022.03.0 meaning March 
                    2022 release. The intent is to indicate that all components 
                    with a common year-month release tag are compatible and to
                    make the association amongst them clear for users. If an 
                    individual component requires a patch release independent 
                    of the others, the release for that component will be 
                    labeled 2022.03.1, for example, to indicate that it is one 
                    patch release beyond the original combined Suite release.

  * New features / API changes:
      * BREAKING CHANGE: RAJA OffsetLayout constructor was changed to take
        (begin, end) args (where end is one past the last index) instead of
        (first, last) args (where last index was included). This is consistent 
        with expected behavior and other RAJA Layout/View concepts.
      * New experimental features that support SIMD/SIMT programming by 
        guaranteeing vectorization without the need to rely on compiler
        auto-vectorization. Basic documentation for this is included in the
        RAJA User Guide and should provide enough description for interested
        users to try it out. 
      * "Flatten" policies were added for RAJA Teams. This reshapes
        multi-dimensional GPU thread blocks to 1D.
      * RAJA Teams now allows a single execution policy to be provided. 
        Previously, it required two; e.g., a CPU policy and a GPU policy.
      * ROCTX support has been added to enable kernel naming with RAJA Teams.
      * Details of CUDA and HIP errors are now added to the reported exception
        string. Previously, this information was going to stderr.
      * All CUDA execution policies have been expanded to allow users to specify
        a minimum number of blocks per SM, if they wish to do that. An analogous
        capability for HIP execution policies is being hashed out. 
      * Changes were made to RAJA scans to address a consistency issue and
        allow const pointers to be passed as the input span.
      * RAJA View pointer type is fixed to properly allow CHAI ManagedArray 
        type to be passed through to View instead of the raw pointer type. This
        fixes an issue where some required CHAI memory transfers were not 
        occurring.
      * A "combining adapter" concept has been added that allows 
        multi-dimensional loops to be run using one-dimensional interfaces.
        Please see the RAJA User Guide for more description.
      * Additional feature support and improvements have been made to the 
        RAJA SYCL back-end (please see the RAJA User Guide for more 
        information):
         * "nontrivially copyable" SYCL interface has been removed 
           (i.e., 'RAJA::sycl_exec_nontrivial<...>' and 
           'RAJA::SyclKernelNonTrivial<...>') as these constructs are no longer
           needed when using recent updates to the Intel OneAPI compiler.
           Execution is now dispatched based on the C++ 'is_trivially_copyable'
           type trait.
         * Support for RAJA::kernel loop tiling policies is now available for
           SYCL execution.
         * The naming scheme for SYCL 'group' and 'local' policies has been
           changed from 1-based to 0-based for block dimensions.
         * The use of the SYCL atomic OneAPI extension namespace has been 
           cleaned up.

  * Build changes/improvements:
      * AS OF THIS RELEASE, RAJA REQUIRES A C++14-COMPLIANT COMPILER TO BUILD!! 
      * AS OF THIS RELEASE, RAJA REQUIRES CMAKE version 3.14.5 or newer.
      * The BLT submodule is updated to v0.5.0, which includes improved
        support for ROCm/HIP builds. Although the option CMAKE_HIP_ARCHITECTURES
        to specify the HIP target architecture is not available until CMake 
        version 3.21, the option is supported in the new BLT version and works 
        with all versions of CMake.
      * The camp submodule is updated to v2022.03.0. If you do not use the 
        RAJA submodule and build RAJA with an external version of camp, you 
        will need to use camp v2022.03.0 or newer.
      * The "RAJA_" prefix has been added to all CMake options. Options that 
        shadow a CMake or BLT option are turned into cmake_dependent_option 
        calls, ensuring that they can be controlled independently and have the 
        correct dependence on the underlying CMake or BLT support;
        e.g., RAJA_ENABLE_CUDA requires ENABLE_CUDA.
      * The camp_DIR export has been removed. Camp paths will be searched 
        using the default logic which is consistent with camp.
      * The raja-config.cmake package file is now "relocatable", meaning it
        can be moved to another directory location after an install and still
        work. This should make it easier to use for applications that use 
        RAJA and CMake, but do not use BLT.
      * CMake logic for using CUB in RAJA for a CUDA build has been changed.
        The default behavior is now that when the CUDA version is < 11, the
        RAJA CUB submodule will be used. When the CUDA version is >= 11, the
        CUB version that is included in the associated CUDA toolkit will be 
        used. Users have the ability to override these defaults and select
        a specific version of CUB if they wish.
      * CMake logic for using rocPRIM in RAJA for a HIP build is similar.
        The default behavior is now that when the HIP version is < 4, the
        RAJA rocPRIM submodule will be used. When the HIP version is >= 4, the
        rocPRIM version that is included in the associated ROCm toolkit will 
        be used. Users have the ability to override these defaults and select
        a specific version of rocPRIM if they wish.
      * The RAJA Spack package was updated to include the version of this 
        release and address some issues.
      * Added a concept of RAJA_HIP_ACTIVE that mirrors RAJA_CUDA_ACTIVE.
      * The CMake option RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL has been 
        removed. Now the choice is made based on the rocm compiler version.

  * Bug fixes/improvements:
      * A bug in TBB non-inplace scan implementation was fixed.
      * RAJA StaticLayout was fixed to avoid compiler warnings due to
        converting a negative integer value to an unsigned integral type.
      * Various improvements, updates, and fixes (formatting, typos, etc.) 
        in RAJA User Guide.


Version v0.14.1 -- Release date 2021-11-15
============================================

This is a patch release for v0.14.0. It updates the camp submodule to v0.2.3 and 
fixes a couple of broken macro include guards in RAJA.


Version v0.14.0 -- Release date 2021-08-20
============================================

This release contains new features, bug fixes, and build improvements. Please
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * New features / API changes:
      * Initial release of some SYCL execution back-end features for supporting
        Intel GPUs. Users should be able to exercise RAJA::forall, basic
        RAJA::kernel, and reductions. Future releases will contain additional
        RAJA feature support for SYCL. 
      * Various enhancements to the experimental RAJA "teams" capability,
        including documentation and complete code examples illustrating usage.
      * The RAJA "teams" interface was expanded to initial support for RAJA 
        resources.
      * The RAJA "teams" interface was expanded to allow users to label
        kernels with name strings to easily attribute execution timings and 
        other details to specific kernels with NVIDIA profiling tools, 
        for example. Usage information is available in the RAJA User Guide.
        Kernel naming will be available for all other RAJA kernel execution
        methods in a future release.
      * Deprecated sort and scan methods taking iterators have been removed,
        Now, these methods take RAJA span arguments. For example,
        (begin, end) args are replaced with RAJA::make_span(begin, N), where
        N = end - begin.  Please see the RAJA User Guide documentation for
        scan and sort operations for details and usage examples.
      * Sort and scan methods now accept an optional resource argument.
      * Methods were added to the RAJA::kernel API to accept a resource 
        argument; specifically 'kernel_resource' and 'kernel_param_resource'.
        These kernel methods return an Event object similar to the RAJA::forall
        interface.
      * Kernel launch methods for the RAJA "teams" interface now use the 
        CAMP default resource based on the specified execution back-end.
        Future work will expand the interface to allow users to pass a
        resource object.
      * RAJA resource support added to RAJA workgroup and worksite constructs.
      * OpenMP CPU multithreading policies have been reworked so that usage
        involving OpenMP scheduling are consistent. Specification of a chunk
        size for scheduling policies is optional, which is consistent with
        native OpenMP usage. In addition, no-wait policies are more constrained
        to prevent potentially non-conforming (to the OpenMP spec) usage. 
        Finally, additional policy type aliases have been added to make common 
        use cases less verbose. Please see the RAJA policy documentation in 
        the User Guide for policy descriptions. 
      * Host implementation of HIP atomics added.
      * Add ability to specify atomic to use on the host in CUDA and HIP
        atomic policies (i.e., added host atomic template parameter), This
        is useful for host-device decorated lambda expressions that may be
        used for either host or device execution. It also fixes compilation 
        issues with HIP atomic compilation in host-device contexts.
      * The RAJA Registry API has been changed to return raw pointers to
        registry objects rather than shared_ptr type. This is better for
        performance.
      * New content has been added to the RAJA Developer Guide available in the
        Read The Docs Sphinx documentation. This should help folks align their
        work with RAJA processes when making contributions to RAJA.
      * Basic doxygen source code documentation is now available via a link
        in our Read The Docs Sphinx documentation.
      * Unified memory implementation for storing indices in TypedListSegment, 
        which was marked deprecated in v0.12.0 release has been removed. Now,
        TypedListSegment constructor requires a camp resource object to be 
        passed which indicates the memory space where the indices will live.
        Specifically, the array of indices passed to the constructor by a user
        (assumed to live in host memory for the "owned" case) will be copied
        to an internally owned allocation in the memory space defined by the
        resource object. 
      * The ListSegment constructor takes a resource by value now, previously
        taken by reference, which allows more resource argument types to be 
        passed more seamlessly to the List Segment constructor.
      * 'CudaKernelFixedSM' and 'CudaKernelFixedSMAsync' methods were added
        which allow users to specify the minimum number of thread blocks to 
        launch per SM. This resulted in a performance improvement for an
        application use case. Future work will expand this concept to other GPU
        kernel execution methods in RAJA.
      * RAJA Modules is deprecated and no longer uses the "-fmodules" flag
        since it can cause issues. The RAJA_ENABLE_MODULES option
        will be removed in the next release.

  * Build changes/improvements:
      * Update BLT submodule to latest release, v0.4.1.
      * Update camp submodule to latest tagged release, v0.2.2
      * The RAJA_CXX_STANDARD_FLAG CMake variable was removed. The BLT_CXX_STD
        variable is now used instead. 
      * Support for building RAJA as a shared library on Windows has been added.
      * A build system adjustment was made to address an issue when RAJA is 
        built with an external version of camp (e.g., through Spack).
      * The build default has been changed to use the version of CUB that
        is installed in the specified version of the CUDA toolkit, if available,
        when CUDA is enabled. Similarly, for the analogous functionality in
        HIP. Specific versions of these libraries can still be specified for
        a RAJA build. Please see the RAJA User Guide for details. 
      * The build system now uses the BLT cmake_dependent_options support for
        options defined by BLT. This avoids shadowing of BLT options by options
        defined in RAJA and in the cases where RAJA is used as a sub-module in
        another BLT project. For example, it provides the ability to disable 
        RAJA tests and examples at a more fine granularity.
      * Checks were added to the RAJA CMake build system to check for minimum
        required versions of CUDA (9.2) and HIP (3.5).
      * A build system bug was fixed so that targets for third-party 
        dependencies provided by BLT (e.g., CUDA and HIP) are exported properly.
        This allows non-BLT projects to use the imported RAJA target.
      * An issue was fixed to appease the MSVC 2019 compiler.
      * Improvements to build system to address HIP linking issues.

  * Bug fixes/improvements:
      * HIP and CUDA block reductions were tweaked to fix the number of steps
        in the final wavefront/warp reduction. This saves a couple rounds of
        warp shfls.
      * A runtime bug resulting from defaulted View constructors not being 
        implemented correctly in CUDA 10.1 is fixed. This fixes an issue
        with CHAI managed arrays not having their copy constructor being 
        triggered properly.
      * Fix bug that caused a CUDA or HIP synchronization error when a zero
        length loop was enqueued in a workgroup.
      * Added missing HIP workgroup unordered execution policy, so HIP 
        version is consistent with CUDA version.
      * Fixed issue where the RAJA non-resource API returns an EventProxy object
        with a dangling resource pointer, by getting a reference to the 
        default resource for the execution context.
      * IndexSet utility methods for collecting indices into a separate 
        container now work with any index type. 
      * The volatile qualifier was removed from a type conversion function used
        in RAJA atomics. This fixes a performance issue with HIP where the 
        value was written to stack memory during type conversion.
      * Numerous improvements, updates, and fixes (formatting, typos, etc.) 
        in RAJA User Guide.


Version v0.13.0 -- Release date 2020-10-30
============================================

This release contains new features, bug fixes, and build improvements. Please 
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * New features:
      * Execution policies for the RAJA HIP back-end and examples have been
        added to the RAJA User Guide and Tutorial.
      * Strongly-typed indices now work with Multiview.

  * Build changes/improvements:
      * Update BLT to latest develop branch (SHA-1: cbe99c93d)
      * Added option to enable/disable runtime plugin loading. This is now 
        off by default. Previously, it was always enabled and there was no
        way to disable it.

  * Bug fixes/improvements:
      * Issues have been addressed so that the OpenMP target back-end is
        now working properly for all supported features. This has been
        verified with multiple clang compilers, including clang 10, and the
        XL compiler.
      * Various data structures have been made trivially copyable to 
        ensure they are mapped properly to device memory with OpenMP 
        target execution.
      * Numerous improvements and fixes (formatting, typos, etc.) in User Guide.

Version v0.12.1 -- Release date 2020-09-09
============================================

This release contains fixes for errors when using a CUDA build with a
non-CUDA compiler and compiler warnings, plus some other bug fixes related
to OpenMP target compilation.

Version v0.12.0 -- Release date 2020-09-03
============================================

This release contains new features, notable changes, and bug fixes. Please
see the RAJA user guide for more information about items in this release.

Notable changes include:

  * Notable repository change:
      * The 'master' branch in the RAJA git repo has been renamed to 'main'.

  * New features:
      * New RAJA "work group" capability added. This allows multiple GPU
        kernels to be fused into one kernel launch, greatly reducing the
        run time overhead of launching CUDA kernels.
      * Added support for dynamic plug-ins in RAJA, which enable the use of
        things like Kokkos Performance Profiline Tools to be used with RAJA
        (https://github.com/kokkos/kokkos-tools)
      * Added ability to pass a resource object to RAJA::forall methods to
        enable asynchronous execution for CUDA and HIP back-ends.
      * Added "Multi-view" that works like a regular view, except that it
        can wrap multiple arrays so their accesses can share index arithmetic.
      * Multiple sort algorithms added. This provides portable parallel sort 
        operations, which are basic parallel algorithm building blocks.
      * Introduced RAJA "Teams" concept as an experimental feature. This
        enables hierarchical parallelism and additional nested loop patterns
        beyond what RAJA::kernel supports. Please note that this is very much
        a work-in-progress and is not yet documented in the user guide.
      * Added initial support for dynamic loop tiling.
      * New OpenMP execution policies added to support static, dynamic, and 
        guided scheduling.
      * Added support for const iterators to be used with RAJA scans.
      * Support for bitwise "and" and "or" reductions have been added.
      * The RAJA::kernel interface has been expanded to allow only segment 
        index arguments used in a lambda to be passed to the lambda. In 
        previous versions of RAJA, every lambda invoked in a kernel had to 
        accept an index argument for every segment in the segment tuple passed 
        to RAJA::kernel execution templates, even if not all segment indices 
        were used in a lambda. This release still allows that usage pattern.
        The new capability requires an additional template parameter to be 
        passed to the RAJA::statement::Lambda type, which identifies the segment 
        indices that will be passed and in which order.
     
  * API Changes:
      * The RAJA 'VarOps' namespace has been removed. All entities previously
        in that namespace are now in the 'RAJA' namespace.
      * RAJA span is now public for users to access and has been made more like
        std::span.
      * RAJA::statement::tile_fixed has been moved to RAJA::tile_fixed
        (namespace change).
      * RAJA::statement::{Segs, Offsets, Params, ValuesT} have been moved to
        RAJA::{Segs, Offsets, Params, ValuesT} (namespace change).
      * RAJA ListSegment constructors have been expanded to accept a camp
        Resource object. This enables run time specification of the memory
        space where the data for list segment indices will live. In earlier
        RAJA versions, the space in which list segment index data lived was a 
        compile-time choice based on whether CUDA or HIP was enabled and the 
        data resided in unified memory for either case. This is still supported
        in this release, but is marked as a DEPRECATED FEATURE. In the next RAJA
        release, ListSegment construction will require a camp Resource object.
        When compiling RAJA with your application, you will see deprecation
        warnings if you are using the deprecated ListSegment constructor. 
      * A reset method was added to OpenMP target offload reduction classes
        so they contain the same functionality as reductions for all other 
        back-ends.

  * Build changes/improvements:
      * The BLT, camp, CUB, and rocPRIM submodules have all been updated to 
        more recent versions. Please note that RAJA now requires rocm version 
        3.5 or newer to use the HIP back-end.
      * Build for clang9 on macosx has been fixed.
      * Build for Intel19 on Windows has been fixed.
      * Host/device annotations have been added to reduction operations to
        eliminate compiler warnings for certain use cases.
      * Several warnings generated by the MSVC compiler have been eliminated.
      * A couple of PGI compiler warnings have been removed.
      * CMake improvements to make it easier to use an external camp or 
        CUB library with RAJA. 
      * Note that the RAJA tests are undergoing a substantial overhaul. Users,
        who chose to build and run RAJA tests, should know that many tests
        are now being generated in the build space directory structure which
        mimics the RAJA source directory structure. As a result, only some
        test executables appear in the top-level 'test' subdirectory of the 
        build directory; others can be found in lower-level directories. The
        reason for this change is to reduce test build times for certain 
        compilers.

  * Bug fixes:
      * An issue with SIMD privatization with the Intel compiler, required
        to generate correct code, has been fixed.
      * An issue with the atomicExchange() operation for the RAJA HIP back-end
        has been fixed.
      * A type issue in the RAJA::kernel implementation involving RAJA span
        usage has been fixed.
      * Checks for iterator ranges and container sizes have been added to
        RAJA scans, which fixes an issue when users attempted to run a 
        scan over a range of size zero.
      * Several type errors in the Layout.hpp header file have been fixed.
      * Several fixes have been made in the Layout and Static Layout types.
      * Several fixes have been made to the OpenMP target offload back-end
        to address host-device memory issues.
      * A variety of RAJA User Guide issues have been addressed, as well as
        issues in RAJA example codes.

Version v0.11.0 -- Release date 2020-01-29
==========================================

This release contains new features, several notable changes, and some bug fixes.

Notable changes include:

  * New features:
      * HIP compiler back-end added to support AMD GPUs. Usage is essentially
        the same as for CUDA. Note that this feature is considered a
        work-in-progress and not yet production ready. It is undocumented,
        but noted here, for friendly users who would like to try it out. 
      * Updated version of camp third-party library, which includes variety
        of portability fixes. Most users should not need to concern 
        themselves with the details of camp.
      * Added new tutorial material and exercises.
      * Documentation improvements.

  * API Changes:
      * None.
 
  * Build changes/improvements:
      * RAJA version number is now accessible as #define macro variable 
        constants so that users who need to parameterize their code to support 
        multiple RAJA versions can do this more easily. See the file 
        RAJA/include/RAJA/config.hpp for details. RAJA version numbers 
        are also experted as CMake variables.
      * Added support to link to external camp library. By default, the camp
        git submodule will be used. If you prefer to use a different version
        of camp, set the RAJA CMake variable 'EXTERNAL_CAMP_SOURCE_DIR' to
        the location of the desired camp directory.
      * BLT submodule (CMake-based build system) has been updated to latest
        BLT release (v0.3.0). The release contains a new version of GoogleTest,
        which required us to modify our use of gtest macros and our own 
        testing macros. For the most part, this change should be invisible to 
        users. However, the new GoogleTest does not work with CUDA versions 
        9.1.x or earlier. Therefore, if you compile RAJA with CUDA enabled and 
        also wish to enable RAJA tests, you must use CUDA 9.2.x or newer.

  * Bug fixes:
      * Fixed various issues to make internal implementations more robust,
        resolved issues with non fully-qualified types in some places, 
        and work arounds for some compiler issues.


Version v0.10.0 -- Release date 2019-10-31
==========================================

This release contains new features, several notable changes, and some bug fixes.

Notable changes include:

  * New features:
      * Added CUDA block direct execution policies, which can be used to map
        loop iterations directly to CUDA thread block. These are analogous to
        the pre-existing thread direct policies. The new block direct policies
        can provide better performance for kernels than the block loop policies
        when load balancing may be an issue. Please see the RAJA User Guide for 
        a description of all available RAJA execution policies.
      * Added a plugin registry feature that will allow plugins to be linked
        into RAJA that can act before and after kernel launches. One benefit
        of this is that RAJA no longer has an explicit CHAI dependency if RAJA 
        is used with CHAI. Future benefits will include integration with other
        tools for performance analysis, etc.
      * Added a shift method to RAJA::View, which allows one to create a new
        view object from an existing one that is shifted in index space from 
        the original. Please see the RAJA User Guide for details.
      * Added support for RAJA::TypedView and RAJA::TypedOffsetLayout, so that 
        the index type can be specified as a template parameter.
      * Added helper functions to convert a RAJA::Layout object to a 
        RAJA::OffsetLayout object and RAJA::TypedLayout to 
        RAJA::TypedOffsetLayout. Please see the RAJA User Guide for details.
      * Added a bounds checking option to RAJA Layout types as a debugging
        feature. This is a compile-time option that will report user errors
        when given View or Layout indices are out-of-bounds. See View/Layout
        section in the RAJA User Guide for instructions on enabling this and 
        how this feature works. 
      * We've added a RAJA Template Project on GitHub, which shows how to
        use RAJA in an application, either as a Git submodule or as an
        externally installed library that you link your application against.
        It is available here: https://github.com/LLNL/RAJA-project-template.
        It is also linked to the main RAJA project page on GitHub. 
      * Various user documentation improvements.

  * API Change.
      * The type alias RAJA::IndexSet that was marked deprecated previously
        has been removed. Now, all index set usage must use the type 
        RAJA::TypedIndexSet and specify all segment types (as template 
        parameters) that the index set may potentially hold.

  * Bug fixes:
      * Fix for issue in OpenMP target offload back-end that previously caused
        some RAJA Performance Suite kernels to seg fault when built with the
        XL compiler.
      * Removed an internal RAJA class constructor to prevent users to do
        potentially incorrect, and very difficult to hunt down, things in 
        their code that are technically not supported in RAJA, such as
        inserting RAJA::statement::CudaSyncThreads() in arbitrary places 
        inside a lambda expression.

  * Build changes/improvements:
      * RAJA now enforces a minimum CUDA compute capability of sm_35. Users
        can use the CMake variable 'CUDA_ARCH' to specify this. If not 
        specified, the value of sm_35 will be used and an informational 
        message will be emitted indicating this. If a user attempts to set
        the value lower than sm_35, CMake will error out and a message will
        be emitted indicating why this happened.
      * Transition to using camp as a submodule after its open source release
        (https://github.com/llnl/camp). 
      * Made minimum required CMake version 3.9.
      * Update BLT build system submodule to newer version 
        (SHA-1 hash: 96419df).
      * Cleaned up compiler warnings in OpenMP target back-end implementation.


Version v0.9.0 -- Release date 2019-07-25
=========================================

This release contains feature enhancements, one breaking change, and some 
bug fixes. 

  * Breaking change
    * The atomic namespace in RAJA has been removed. Now, use atomic operations
      as RAJA::atomicAdd(), not RAJA::atomic::atomicAdd(), for example. This
      was done to make atomic usage consistent with other RAJA features, such
      as reductions, scans, etc.

Other notable changes include:

  * Features
    * The lambda statement interface has been extended in the RAJA kernel API.
      Earlier, when multiple lambda expressions were used in a kernel, they
      were required to all have the same arguments, although not all 
      arguments had to be used in each lambda expression. Now, lambda 
      arguments may be specified in the RAJA::statement::Lambda type so 
      that each lambda expression need only take the arguments it uses.
      However, the previous usage pattern will continue to be supported.
      To support the new interface, new statement types have been introduced 
      to indicate iteration space variables (Segs), local variable/array 
      parameters (Params), and index offsets (Offsets). The offsets can be used
      with a For statement as a replacement for the ForICount statement. The
      new API features are described in the RAJA User Guide.
    * Minloc and maxloc reductions now support a tuple of index values. So 
      now if you have a nested loop kernel with i, j, k loops, you can get
      the 'loc' value out as an i, j, k triple.

  * Bug Fixes:
    * Small change to make RAJA Views work properly with OpenMP target kernels.
    * Changes to fix OpenMP target back-end for XL compilers.
    * Fix build issue with older versions of GNU compiler.
    * Fixes to resolve issues associated with corner cases in choosing 
      improper number of threads per block or number of thread blocks for
      CUDA execution policies.

  * Build changes/improvements:
    * A few minor portability improvements


Version v0.8.0 -- Release date 2019-03-28
=========================================

This release contains one major change and some minor improvements to 
compilation and performance.

Major changes include:

  * Build system updated to use the latest version of BLT (or close to it). 
    Depending on how one builds RAJA, this could require changes to how 
    information is passed to CMake. Content has been added to the relevant 
    sections of the RAJA User Guide which describes how this is done.

Other notable changes include:

  * Features (These are not yet documented and should be considered 
    experimental. There will be documentation and usage examples in the
    next RAJA release.)
    * New thread, warp, and bitmask policies for CUDA. These are not
      yet documented and should be considered experimental.
    * Added AtomicLocalArray type which returns data elements wrapped
      in an AtomicRef object.

  * Bug Fixes:
    * Fixed issue in RangeStrideSegment iteration.
    * Fix 'align hint' macro to eliminate compile warning when XL compiler
      is used with nvcc.
    * Fix issues associated with CUDA architecture level (i.e., sm_*) set
      too low and generated compiler warning/errors. Caveats for RAJA features
      (mostly atomic operations) available at different CUDA architecture 
      levels added to User Guide.

  * Performance Improvements:
    * Some performance improvements in RAJA::kernel usage with CUDA back-end.


Version v0.7.0 -- Release date 2019-02-07
=========================================

This release contains several major changes, new features, a variety of bug 
fixes, and expanded user documentation and accompanying example codes. For
more information and details about any of the changes listed below, please 
consult the RAJA documentation for the 0.7.0 release which is linked to 
our Github project.

Major changes include:

  * RAJA::forallN methods were marked deprecated in the 0.6.0 release. They 
    have been removed. All applications that contain nested loops and 
    have been using forallN methods should convert them to use the RAJA::kernel
    interface.
  * RAJA::forall methods that take explicit loop bounds rather than segments  
    (e.g., RAJA::forall(beg, end, ...) were marked deprecated in the 0.6.0
    release. They have been removed. Hopefully, this will result in faster
    compile times due to simpler template resolution. Users who have been 
    passing loop bounds directly to forall methods should convert those
    cases to use RAJA segments instead.
  * CUDA execution policies for use in RAJA::kernel policies have been 
    significantly reworked and redefined. The new set of policies are
    much more flexible and provide improved run time performance.
  * New, improved support for loop tiling algorithms and support for 
    CPU cache blocking, CUDA GPU thread local data and shared memory is 
    available. This includes RAJA::kernel policy statement types to make tile
    numbers and local tile indices available in user kernels (TileTCount and
    ForICount statement types), and a new RAJA::LocalArray type with various 
    CPU and GPU memory policies. Due to these new features, RAJA 'shmem window'
    statements have been removed.
  * This release contains expanded documentation and example codes for the
    RAJA::kernel interface, including loop tiling algorithms and support for
    CPU cache blocking, CUDA GPU thread local data and shared memory.

Other notable changes include:

  * Features:
    * Initial support for OpenMP target execution policies with RAJA::kernel
      added.
    * The RAJA::AtomicRef interface is now consistent with the 
      C++20 std::atomic_ref interface.
    * Atomic compare-exchange operations added.
    * CUDA reduce policies no longer require a thread-block size parameter.
    * New features considered preliminary with no significant documentation or
      examples available yet:
        * RAJA::statement::Reduce type for use in RAJA::kernel execution 
          policies. This enables the ability to perform reductions and access
          reduced values inside user kernels.
        * Warp-level execution policies added for CUDA.

  * Performance improvements:
    * Better use of inline directives to improve likelihood of SIMD 
      instruction generation with the Intel compiler.

  * Bug fixes:
    * Several CHAI integration issues resolved.
    * Resolve issue with alignx directive when using XL compiler as host
      compiler with CUDA.
    * Fix issue associated with how XL compiler interprets OpenMP region
      definition.
    * Various tweaks to camp implementation to improve robustness.

  * Build changes/improvements:
    * The minimum required version of CMake has changed to 3.8 for all
      programming model back-ends, except CUDA. The minimum CMake version
      for CUDA support is 3.9.
    * Improved support for clang-cuda compiler. Some features still do not
      work with that compiler.
    * Update NVIDIA cub module to version 1.8.0.
    * Enable use of 'BLT_SOURCE_DIR' CMake variable to help prevent conflicts
      with BLT versions in RAJA and other libraries used in applications.


Version v0.6.0 -- Release date 2018-07-27
=========================================

This release contains two major changes, a variety of bug fixes and feature
enhancements, and expanded user documentation and accompanying example codes.

Major changes include:

  * RAJA::forallN methods are marked deprecated. They will be removed in 
    the 0.7.0 release.
  * RAJA::forall methods that take loop bounds rather than segments  (e.g.,
    RAJA::forall(beg, end, ...) are marked deprecated. They will be removed 
    in the 0.7.0 release.
  * RAJA::nested has been replaced with RAJA::kernel. The RAJA::kernel interface
    is much more flexible and full featured. Going forward, it will be the 
    supported interface for nested loops and more complex kernels in RAJA. 
  * This release contains new documentation and example codes for the 
    RAJA::kernel interface. The documentation described key features and
    summarizes available 'statement' types. However, it remains a 
    work-in-progress and expanded documentation with more examples will be 
    available in future releases.
  * Documentation of other RAJA features has been expanded and improved in
    this release along with additional example codes.

Other notable changes include:

  * New or improved features: 
      * RAJA CUDA reductions now work with host/device lambdas 
      * List segments now work with RAJA::kernel loops.
      * New and expanded collection of build files for LC and ALCF machines.
        Hopefully, these will be helpful to folks getting started.

  * Performance improvements: 
      * Some RAJA::View use cases
      * Unnecessary operations removed in min/max atomics
    
  * Bug fixes: 
      * Issues in View with OffsetLayout fixed.
      * Construction of a const View from a non-const View now works
      * CUDA kernel no longer launched in RAJA::kernel loops when iteration 
        space has size zero


Version v0.5.3 -- Release date 2018-01-31
=========================================

This is a bugfix release that fixes bugs in the IndexSetBuilder methods. These
methods now work correctly with the strongly-typed IndexSet.


Version v0.5.2 -- Release date 2018-01-30
=========================================

This release fixes some small bugs, including compiler warnings issued for
deprecated features, type narrowing, and the slice method for the
RangeStrideSegment class.

It also adds a new CMake variable, RAJA_LOADED, that is used to determine
whether RAJA's CMakeLists file has already been processed. This is useful when
including RAJA as part of another CMake project.


Version v0.5.1 -- Release date 2018-01-17
=========================================

This release contains fixes for compiler warnings with newer GCC and Clang
compilers, and allows strongly-typed indices to work with RangeStrideSegment.

Additionally, the index type for all segments in an IndexSet needs to be the
same. This requirement is enforced with a static_assert.


Version v0.5.0 -- Release date 2018-01-11
=========================================

This release contains a variety of bug fixes, removes nvcc compiler
warnings, addition of unit tests to expand coverage, and a variety of 
other code cleanup and improvements. The most notable changes in this 
version include:

  * New RAJA User Guide and Tutorial along with a set of example codes
    that illustrate basic usage of RAJA features and which accompany
    the tutorial. The examples are in the ``RAJA/examples`` directory.
    The user guide is available online here:
    [RAJA User Guide and Tutorial](http://raja.readthedocs.io/en/master/).

  * ``RAJA::IndexSet`` is now deprecated. You may still use it until it is
    removed in a future release -- you will see a notification message at
    compile time that it is deprecated.

    Index set functionality will now be available via ``RAJA::TypedIndexSet``
    where you specify all segment types as template parameters when you 
    declare an instance of it. This change allows us to: remove all virtual
    methods from the index set, be able to use index set objects to CUDA
    GPU kernels and all of their functionality, and support any arbitrary
    segment type even user-defined. Please see User Guide for details.

    Segment dependencies are being developed for the typed index set and 
    will be available in a future release.

  * ``RAJA::nested::forall`` changes:

    * Addition of CUDA and OpenMP collapse policies for nested loops.
      OpenMP collapse will do what the OpenMP collapse clause does. 
      CUDA collapse will collapse a loop nest into a single CUDA kernel based 
      on how nested policies specify how the loop levels should be distributed 
      over blocks and threads.

    * Added new policy ``RAJA::cuda_loop_exec`` to enable inner loops to run
      sequentially inside a CUDA kernel with ``RAJA::nested::forall``.

    * Fixed ``RAJA::nested::forall`` so it now works with RAJA's CUDA Reducer 
      types.

    * Removed ``TypedFor`` policies. For type safety of nested loop iteration
      variables, it makes more sense to use ``TypedRangeSegment`` since the
      variables are associated with the loop kernel and not the execution 
      policy, which may be applied to multiple loops with different variables.

  * Fixed OpenMP scans to calculate chunks of work based on actual number of
    threads the OpenMP runtime makes available.

  * Enhancements and fixes to RAJA/CHAI interoperability.

  * Added aliases for several ``camp`` types in the RAJA namespace; e.g.,
    ``camp::make_tuple`` can now be accessed as ``RAJA::make_tuple``. This 
    change makes the RAJA API more consistent and clear.


Version v0.4.1 -- Release date 2017-10-11
=========================================

This release contains a bugfix for warnings when using the -Wpedantic flag.


Version v0.4.0 -- Release date 2017-10-11
=========================================

This release contains minor fixes for issues in the previous v0.3.1 release, 
plus some improvements to documentation, reduction performance, improved 
portability across a growing set of compilers and environments (e.g., Windows),
namespace refactoring to avoid cyclic dependencies and leverage argument-
dependent lookup, etc. In addition, the RAJA backend for Intel TBB is now 
off by default, whereas previously it was on by default.

A few major changes are included in this version:

  * Changes to the way RAJA is configured and built. We are now using the 
    BLT build system which is a Git submodule of RAJA. In addition to 
    requiring the '--recursive' option to be passed to 'git clone', this 
    introduces the following major change: RAJA_ENABLE_XXX options passed to 
    CMake are now just ENABLE_XXX.

  * A new API and implementation for nested-loop RAJA constructs has been 
    added. It is still a work in progress, but users are welcome to try it 
    out and provide feedback. Eventually, RAJA::nested::forall will replace 
    RAJA::forallN.


Version v0.3.1 -- Release date 2017-09-21
=========================================

This release contains some new RAJA features, plus a bunch of internal changes 
including more tests, conversion of nearly all unit tests to use Google Test, 
improved testing coverage, and compilation portability improvements (e.g., 
Intel, nvcc, msvc). Also, the prefix for all RAJA source files has been changed
from 'cxx'to 'cpp' for consistency with the header file prefix conversion in 
the last release. The source file prefix change should not require users to 
change anything.

New features included in this release:

  * Execution policy modifications and additions: 
     
      * seq_exec is now strictly sequential (no SIMD, etc.) 
      * simd_exec will force SIMD vectorization 
      * loop_exec (new policy) will allow compiler to optimize however it can, 
        including SIMD. 

    So, loop_exec is really what our previous simd_exec policy was before, 
    and 'no vector' pragmas have been added to all sequential implementations. 

    NOTE: SIMD changes are still being evaluated with different compilers on 
    different platforms. More information will be provided as we learn more.

  * Added support for atomic operations (min, max, inc, dec, and, or, xor, 
    exchange, and CAS) for all programming model backends. These appear in the 
    RAJA::atomic namespace. 

  * Support added for Intel Threading Building Blocks backend (considered 
    experimental at this point). 

  * Added macros that will be used to mark features for future deprecation 
    (please watch for this as we will be deprecating some features in the 
    next release).

  * Added support for C++17 if CMake knows about it.

  * Remove limit on number of ordered OpenMP reductions that can be used in 
    a kernel.

  * Remove compile-time error from memutils, add portable aligned allocator.

  * Improved ListSegment implementation.

  * RAJA::Index_type is now ptrdiff_t instead of int.

Notable bug fixes included in this release:

  * Fixed strided_numeric_iterator to apply stride sign in comparison.

  * Bug in RangeStrideSegment when using CUDA is fixed.

  * Fixed reducer logic for openmp_ordered policy.


Version v0.3.0 -- Release date 2017-07-13
=========================================

This release contains breaking changes and is not backward compatible with 
prior versions. The largest change is a re-organization of header files, 
and the switch to .hpp as a file extension for all headers.

New features included in this release:

  * Re-organization of header files.

  * Renaming of file extensions.

  * Rudimentary OpenMP 4.5 support.

  * CHAI support.


Version v0.2.5 -- Release date 2017-03-28
=========================================

This release includes some small fixes, as well as an initial re-organization 
of the RAJA header files as we move towards a more flexible usage model.


Version v0.2.4 -- Release date 2017-02-22
=========================================

This release includes the following changes:

 * Initial support of clang-cuda compiler.

 * New, faster OpenMP reductions.

N.B. The default OpenMP reductions are no longer performed in a ordered 
     fashion, so results may not be reproducible. The old reductions are 
     still available with the policy RAJA::omp_reduce_ordered.


Version v0.2.3 -- Release date 2016-12-15
=========================================

Hotfix to update the URLs used for fetching clang during Travis builds.


Version v0.2.2 -- Release date 2016-12-14
=========================================

Bugfix release that address an error when launching forall cuda kernels 
with a 0-length range.


Version v0.2.1 -- Release date 2016-12-07
=========================================

This release contains fixes for compiler warnings and removes the usage of 
the custom FindCUDA CMake package.


Version v0.2.0 -- Release date 2016-12-02
=========================================

Includes internal changes for performance and code maintenance.


Version v0.1.0 -- Release date 2016-06-22
=========================================

Initial release.
