/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA concept definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
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

#ifndef RAJA_concepts_HPP
#define RAJA_concepts_HPP

#include <iterator>
#include <type_traits>

#include "camp/concepts.hpp"

namespace RAJA
{

namespace concepts
{
using namespace camp::concepts;

template<typename From, typename To>
struct ConvertibleTo
    : DefineConcept(::RAJA::concepts::convertible_to<To>(camp::val<From>()))
{};

}  // namespace concepts

namespace type_traits
{
using namespace camp::type_traits;

DefineTypeTraitFromConcept(convertible_to, concepts::ConvertibleTo);
}  // namespace type_traits

}  // end namespace RAJA

#endif
