/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Main RAJA header file.
 *
 *          This is the main header file to include in code that uses RAJA.
 *          It provides a single access point to all RAJA features by
 *          including other RAJA headers.
 *
 *          IMPORTANT: If changes are made to this file, note that contents
 *                     of some header files require that they are included
 *                     in the order found here.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_HPP
#define RAJA_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/camp_aliases.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/math.hpp"
#include "RAJA/util/plugins.hpp"
#include "RAJA/util/Registry.hpp"
#include "RAJA/util/for_each.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/launch.hpp"

//
// Generic templates to describe SIMD/SIMT registers and vectors
//
#if defined(RAJA_ENABLE_VECTORIZATION)
#include "RAJA/pattern/tensor.hpp"
#endif

//
// All platforms must support sequential execution.
//
#include "RAJA/policy/sequential.hpp"

//
// All platforms should support simd and vector execution.
//
#include "RAJA/policy/simd.hpp"
#if defined(RAJA_ENABLE_VECTORIZATION)
#include "RAJA/policy/tensor.hpp"
#endif

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda.hpp"
#endif

#if defined(RAJA_ENABLE_HIP)
#include "RAJA/policy/hip.hpp"
#endif

#if defined(RAJA_ENABLE_SYCL)
#include "RAJA/policy/sycl.hpp"
#endif

#if defined(RAJA_ENABLE_OPENMP)
#include "RAJA/policy/openmp.hpp"

#if defined(RAJA_ENABLE_TARGET_OPENMP)
#include "RAJA/policy/openmp_target.hpp"
#endif
#endif

#if defined(RAJA_ENABLE_DESUL_ATOMICS)
#include "RAJA/policy/desul.hpp"
#endif

#include "RAJA/index/IndexSet.hpp"

//
// Strongly typed index class
//
#include "RAJA/index/IndexValue.hpp"


//
// Generic iteration templates require specializations defined
// in the files included below.
//
#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

#include "RAJA/policy/MultiPolicy.hpp"


//
// Multidimensional layouts and views
//
#include "RAJA/util/Layout.hpp"
#include "RAJA/util/OffsetLayout.hpp"
#include "RAJA/util/PermutedLayout.hpp"
#include "RAJA/util/StaticLayout.hpp"
#include "RAJA/util/IndexLayout.hpp"
#include "RAJA/util/View.hpp"


//
// View for sequences of objects
//
#include "RAJA/util/Span.hpp"

//
// zip iterator to iterator over sequences simultaneously
//
#include "RAJA/util/zip.hpp"

//
// Atomic operations support
//
#include "RAJA/pattern/atomic.hpp"

// Thread operations support
//
#include "RAJA/pattern/thread.hpp"

//
// Shared memory view patterns
//
#include "RAJA/util/LocalArray.hpp"

//
// Bit masking operators
//
#include "RAJA/util/BitMask.hpp"

//
// sort algorithms
//
#include "RAJA/util/sort.hpp"

//
// reduce algorithms
//
#include "RAJA/util/reduce.hpp"

//
// WorkPool, WorkGroup, WorkSite objects
//
#include "RAJA/policy/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

//
// Reduction objects
//
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/pattern/multi_reduce.hpp"


//
// Synchronization
//
#include "RAJA/pattern/synchronize.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// These contents of the header files included here define index set
// and segment execution methods whose implementations depend on
// programming model choice.
//
// The ordering of these file inclusions must be preserved since there
// are dependencies among them.
//
//////////////////////////////////////////////////////////////////////
//

#include "RAJA/index/IndexSetUtils.hpp"
#include "RAJA/index/IndexSetBuilders.hpp"

#include "RAJA/pattern/scan.hpp"

#if defined(RAJA_ENABLE_RUNTIME_PLUGINS)
#include "RAJA/util/PluginLinker.hpp"
#endif

#include "RAJA/pattern/sort.hpp"

namespace RAJA
{
namespace expt
{}

//  // provide a RAJA::expt namespace for experimental work, but bring alias
//  // it into RAJA so it doesn't affect user code
//  using namespace expt;
}  // namespace RAJA

#endif  // closing endif for header file include guard
