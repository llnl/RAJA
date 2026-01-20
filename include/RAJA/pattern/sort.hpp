/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
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

#ifndef RAJA_sort_HPP
#define RAJA_sort_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{

inline namespace policy_by_value_interface
{

/*!
******************************************************************************
*
* \brief  sort execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c RandomAccess Container
*range
* \param[in] comp comparison function to apply for sort
*
******************************************************************************
*/
template<
    typename ExecPolicy,
    typename Res,
    typename Container,
    typename Compare = operators::less<RAJA::detail::ContainerVal<Container>>>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>,
                      std::is_constructible<camp::resources::Resource, Res>,
                      type_traits::is_range<Container>>
sort(ExecPolicy&& p, Res r, Container&& c, Compare comp = Compare {})
{
  using std::begin;
  using std::distance;
  using std::end;
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");

  auto begin_it = begin(c);
  auto end_it   = end(c);
  auto N        = distance(begin_it, end_it);

  if (N > 1)
  {
    return impl::sort::unstable(r, std::forward<ExecPolicy>(p), begin_it,
                                end_it, comp);
  }
  else
  {
    return resources::EventProxy<Res>(r);
  }
}

///
template<
    typename ExecPolicy,
    typename Container,
    typename Compare = operators::less<RAJA::detail::ContainerVal<Container>>,
    typename Res     = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_execution_policy<ExecPolicy>,
    type_traits::is_range<Container>,
    concepts::negate<
        std::is_constructible<camp::resources::Resource, Container>>>
sort(ExecPolicy&& p, Container&& c, Compare comp = Compare {})
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::sort(
      std::forward<ExecPolicy>(p), r, std::forward<Container>(c), comp);
}

/*!
******************************************************************************
*
* \brief  stable sort execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c RandomAccess Container
*range
* \param[in] comp comparison function to apply for stable_sort
*
******************************************************************************
*/
template<
    typename ExecPolicy,
    typename Res,
    typename Container,
    typename Compare = operators::less<RAJA::detail::ContainerVal<Container>>>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>,
                      std::is_constructible<camp::resources::Resource, Res>,
                      type_traits::is_range<Container>>
stable_sort(ExecPolicy&& p, Res r, Container&& c, Compare comp = Compare {})
{
  using std::begin;
  using std::distance;
  using std::end;
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");

  auto begin_it = begin(c);
  auto end_it   = end(c);
  auto N        = distance(begin_it, end_it);

  if (N > 1)
  {
    return impl::sort::stable(r, std::forward<ExecPolicy>(p), begin_it, end_it,
                              comp);
  }
  else
  {
    return resources::EventProxy<Res>(r);
  }
}

///
template<
    typename ExecPolicy,
    typename Container,
    typename Compare = operators::less<RAJA::detail::ContainerVal<Container>>,
    typename Res     = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_execution_policy<ExecPolicy>,
    type_traits::is_range<Container>,
    concepts::negate<
        std::is_constructible<camp::resources::Resource, Container>>>
stable_sort(ExecPolicy&& p, Container&& c, Compare comp = Compare {})
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::stable_sort(
      std::forward<ExecPolicy>(p), r, std::forward<Container>(c), comp);
}

/*!
******************************************************************************
*
* \brief  sort pairs execution pattern
*
* \param[in] p Execution policy
* \param[in,out] keys RandomAccess Container or range of keys to be sorted
* \param[in,out] values RandomAccess Container or range of values to reorder
* along with keys
* \param[in] comp comparison function to apply to keys for sort
*
******************************************************************************
*/
template<typename ExecPolicy,
         typename Res,
         typename KeyContainer,
         typename ValContainer,
         typename Compare =
             operators::less<RAJA::detail::ContainerVal<KeyContainer>>>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>,
                      std::is_constructible<camp::resources::Resource, Res>,
                      type_traits::is_range<KeyContainer>,
                      type_traits::is_range<ValContainer>>
sort_pairs(ExecPolicy&& p,
           Res r,
           KeyContainer&& keys,
           ValContainer&& vals,
           Compare comp = Compare {})
{
  using std::begin;
  using std::distance;
  using std::end;
  using T = RAJA::detail::ContainerVal<KeyContainer>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<KeyContainer>::value,
                "KeyContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<ValContainer>::value,
                "ValContainer must model RandomAccessRange");

  auto begin_key = begin(keys);
  auto end_key   = end(keys);
  auto N         = distance(begin_key, end_key);

  if (N > 1)
  {
    return impl::sort::unstable_pairs(r, std::forward<ExecPolicy>(p), begin_key,
                                      end_key, begin(vals), comp);
  }
  else
  {
    return resources::EventProxy<Res>(r);
  }
}

///
template<typename ExecPolicy,
         typename KeyContainer,
         typename ValContainer,
         typename Compare =
             operators::less<RAJA::detail::ContainerVal<KeyContainer>>,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_execution_policy<ExecPolicy>,
    type_traits::is_range<KeyContainer>,
    concepts::negate<
        std::is_constructible<camp::resources::Resource, KeyContainer>>,
    type_traits::is_range<ValContainer>>
sort_pairs(ExecPolicy&& p,
           KeyContainer&& keys,
           ValContainer&& vals,
           Compare comp = Compare {})
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::sort_pairs(
      std::forward<ExecPolicy>(p), r, std::forward<KeyContainer>(keys),
      std::forward<ValContainer>(vals), comp);
}

/*!
******************************************************************************
*
* \brief  stable sort pairs execution pattern
*
* \param[in] p Execution policy
* \param[in,out] keys RandomAccess KeyContainer or range of keys to be sorted
* \param[in,out] vals RandomAccess Container or range of values to reorder
* along with keys
* \param[in] comp comparison function to apply to keys for stable_sort
*
******************************************************************************
*/
template<typename ExecPolicy,
         typename Res,
         typename KeyContainer,
         typename ValContainer,
         typename Compare =
             operators::less<RAJA::detail::ContainerVal<KeyContainer>>>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>,
                      std::is_constructible<camp::resources::Resource, Res>,
                      type_traits::is_range<KeyContainer>,
                      type_traits::is_range<ValContainer>>
stable_sort_pairs(ExecPolicy&& p,
                  Res r,
                  KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare {})
{
  using std::begin;
  using std::distance;
  using std::end;
  using T = RAJA::detail::ContainerVal<KeyContainer>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<KeyContainer>::value,
                "KeyContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<ValContainer>::value,
                "ValContainer must model RandomAccessRange");

  auto begin_key = begin(keys);
  auto end_key   = end(keys);
  auto N         = distance(begin_key, end_key);

  if (N > 1)
  {
    return impl::sort::stable_pairs(r, std::forward<ExecPolicy>(p), begin_key,
                                    end_key, begin(vals), comp);
  }
  else
  {
    return resources::EventProxy<Res>(r);
  }
}

///
template<typename ExecPolicy,
         typename KeyContainer,
         typename ValContainer,
         typename Compare =
             operators::less<RAJA::detail::ContainerVal<KeyContainer>>,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<
    resources::EventProxy<Res>,
    type_traits::is_execution_policy<ExecPolicy>,
    type_traits::is_range<KeyContainer>,
    concepts::negate<
        std::is_constructible<camp::resources::Resource, KeyContainer>>,
    type_traits::is_range<ValContainer>>
stable_sort_pairs(ExecPolicy&& p,
                  KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare {})
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::stable_sort_pairs(
      std::forward<ExecPolicy>(p), r, std::forward<KeyContainer>(keys),
      std::forward<ValContainer>(vals), comp);
}

}  // namespace policy_by_value_interface

// =============================================================================

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * sort
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<typename ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>>
sort(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::sort<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<typename ExecPolicy, typename Res, typename... Args>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>>
sort(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::sort(ExecPolicy(), r,
                                                 std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * stable_sort
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<typename ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>>
stable_sort(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::stable_sort<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<typename ExecPolicy, typename Res, typename... Args>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>>
stable_sort(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::stable_sort(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * sort_pairs
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<typename ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>>
sort_pairs(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::sort_pairs<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<typename ExecPolicy, typename Res, typename... Args>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>>
sort_pairs(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::sort_pairs(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * sort_pairs
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<typename ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>>
stable_sort_pairs(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::stable_sort_pairs<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<typename ExecPolicy, typename Res, typename... Args>
concepts::enable_if_t<resources::EventProxy<Res>,
                      type_traits::is_execution_policy<ExecPolicy>,
                      type_traits::is_resource<Res>>
stable_sort_pairs(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::stable_sort_pairs(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
