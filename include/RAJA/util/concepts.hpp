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
#define DefineTypeTraitFromConceptTwoTypeParams(TTName, ConceptName)           \
  template<class T, class U>                                                   \
  struct TTName : std::bool_constant<ConceptName<T, U>>                        \
  {};                                                                          \
  template<class T, class U>                                                   \
  inline constexpr bool TTName##_v = TTName<T, U>::value;

#define DefineTypeTraitFromConceptThreeTypeParams(TTName, ConceptName)         \
  template<class T, class U, class V>                                          \
  struct TTName : std::bool_constant<ConceptName<T, U, V>>                     \
  {};                                                                          \
  template<class T, class U, class V>                                          \
  inline constexpr bool TTName##_v = TTName<T, U, V>::value;


}  // namespace concepts

namespace type_traits
{
using namespace camp::type_traits;

}  // namespace type_traits

}  // end namespace RAJA

#endif
