/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA memory utility routines.
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

#ifndef RAJA_memory_HPP
#define RAJA_memory_HPP

#include "RAJA/config.hpp"

#include <cstddef>

namespace RAJA
{

/*!
 * \brief Release unused backing allocations from RAJA internal memory pools.
 *
 * This routine does not synchronize backend work. Callers must ensure that
 * relevant RAJA and device work has completed before calling it.
 *
 * \return Total bytes released from enabled backend pools.
 */
RAJASHAREDDLL_API size_t release_unused_internal_memory();

}  // namespace RAJA

#endif  // RAJA_memory_HPP
