/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with aliases to camp types.
 *
 * The aliases included here are the ones that may be exposed through the
 * RAJA API based on our unit tests and examples. As you build new tests
 * and examples and you find that other camp types are exposed, please
 * add them to this file.
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

#ifndef RAJA_CAMP_ALIASES_HPP
#define RAJA_CAMP_ALIASES_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"

#include "camp/defines.hpp"
#include "camp/list/list.hpp"
#include "camp/tuple.hpp"
#include "camp/resource.hpp"

namespace RAJA
{

using ::camp::at_v;

using ::camp::list;

using ::camp::idx_t;

using ::camp::make_tuple;

using ::camp::tuple;

using ::camp::tuple_element;

using ::camp::tuple_element_t;

using ::camp::get;

using ::camp::resources::Platform;

}  // end namespace RAJA

#endif /* RAJA_CAMP_ALIASES_HPP */
