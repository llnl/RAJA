/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA HashCombiner.
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

#ifndef RAJA_HashCombiner_HPP
#define RAJA_HashCombiner_HPP

#include <type_traits>
#include <utility>

namespace RAJA
{
namespace detail
{
/*!
 * @brief Combines hashes of all types together.
 * @param values Objects to combine hashes for
 * @return Returns hash combined of objects
 *
 * This will combine the hash values of various objects. The
 * hash value of each type uses `std::hash`.
 *
 * Note: this is a simple hash combiner that can generate
 * collisons.
 *
 */
template<typename... Ts>
constexpr std::size_t hash_combine(Ts&&... values)
{
  std::size_t hash = 0;

  ((hash ^= (std::hash<std::remove_cvref_t<Ts>> {}(values) << 1)), ...);

  return hash;
}

/*!
 * @brief Combines hashes of pair types together.
 *
 * This will combine the hash values of the two stored types using
 * `RAJA::hash_combine`.
 *
 */
struct PairHash
{
  template<typename T1, typename T2>
  constexpr std::size_t operator()(const std::pair<T1, T2>& p) const
  {
    return RAJA::detail::hash_combine(p.first, p.second);
  }
};
}  // end namespace detail
}  // end namespace RAJA

#endif /* RAJA_HashComber_HPP */
