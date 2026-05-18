/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
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

#ifndef RAJA_scan_HPP
#define RAJA_scan_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"
#include "camp/concepts.hpp"

namespace RAJA
{

inline namespace policy_by_value_interface
{

/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c Random-Access Container
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
******************************************************************************
*/
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         concepts::RandomAccessRange Container,
         typename R = RAJA::detail::ContainerVal<Container>,
         concepts::BinaryFunction<R, R, R> Function = operators::plus<R>>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan_inplace(
    ExecPolicy&& p,
    Res r,
    Container&& c,
    Function binop = Function {})
{
  using std::begin;
  using std::end;
  if (begin(c) == end(c))
  {
    return resources::EventProxy<Res>(r);
  }
  return impl::scan::inclusive_inplace(r, std::forward<ExecPolicy>(p), begin(c),
                                       end(c), binop);
}

///
template<
    concepts::ExecutionPolicy ExecPolicy,
    concepts::RandomAccessRange Container,
    typename Function = operators::plus<RAJA::detail::ContainerVal<Container>>,
    typename Res      = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan_inplace(
    ExecPolicy&& p,
    Container&& c,
    Function binop = Function {})
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::inclusive_scan_inplace(
      std::forward<ExecPolicy>(p), r, std::forward<Container>(c), binop);
}

/*!
******************************************************************************
*
* \brief  exclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c RandomAccess Container
* \param[in] binop binary function to apply for scan
* \param[in] value identity for binary function, binop
*
******************************************************************************
*/
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         concepts::RandomAccessRange Container,
         typename T = RAJA::detail::ContainerVal<Container>,
         concepts::BinaryFunction<T, T, T> Function = operators::plus<T>>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan_inplace(
    ExecPolicy&& p,
    Res r,
    Container&& c,
    Function binop = Function {},
    T value        = Function::identity())
{
  using std::begin;
  using std::end;
  if (begin(c) == end(c))
  {
    return resources::EventProxy<Res>(r);
  }
  return impl::scan::exclusive_inplace(r, std::forward<ExecPolicy>(p), begin(c),
                                       end(c), binop, value);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::RandomAccessRange Container,
         typename T        = RAJA::detail::ContainerVal<Container>,
         typename Function = operators::plus<T>,
         typename Res      = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan_inplace(
    ExecPolicy&& p,
    Container&& c,
    Function binop = Function {},
    T value        = Function::identity())
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::exclusive_scan_inplace(
      std::forward<ExecPolicy>(p), r, std::forward<Container>(c), binop, value);
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution pattern
*
* \param[in] p Execution policy
* \param[in] c Random-Access Container
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         concepts::RandomAccessRange InContainer,
         concepts::RandomAccessRange OutContainer,
         typename T = RAJA::detail::ContainerVal<InContainer>,
         typename R = RAJA::detail::ContainerVal<OutContainer>,
         concepts::BinaryFunction<R, T, R> Function = operators::plus<T>>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan(
    ExecPolicy&& p,
    Res r,
    InContainer&& in,
    OutContainer&& out,
    Function binop = Function {})
{
  using std::begin;
  using std::end;
  if (begin(in) == end(in))
  {
    return resources::EventProxy<Res>(r);
  }
  return impl::scan::inclusive(r, std::forward<ExecPolicy>(p), begin(in),
                               end(in), begin(out), binop);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::RandomAccessRange InContainer,
         concepts::RandomAccessRange OutContainer,
         typename Function =
             operators::plus<RAJA::detail::ContainerVal<InContainer>>,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan(
    ExecPolicy&& p,
    InContainer&& in,
    OutContainer&& out,
    Function binop = Function {})
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::inclusive_scan(
      std::forward<ExecPolicy>(p), r, std::forward<InContainer>(in),
      std::forward<OutContainer>(out), binop);
}

/*!
******************************************************************************
*
* \brief  exclusive scan execution pattern
*
* \param[in] p Execution policy
* \param[in] c Random-Access Container
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         concepts::RandomAccessRange InContainer,
         concepts::RandomAccessRange OutContainer,
         typename T = RAJA::detail::ContainerVal<InContainer>,
         typename R = RAJA::detail::ContainerVal<OutContainer>,
         concepts::BinaryFunction<R, T, T> Function = operators::plus<T>>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan(
    ExecPolicy&& p,
    Res r,
    InContainer&& in,
    OutContainer&& out,
    Function binop = Function {},
    T value        = Function::identity())
{
  using std::begin;
  using std::end;
  if (begin(in) == end(in))
  {
    return resources::EventProxy<Res>(r);
  }
  return impl::scan::exclusive(r, std::forward<ExecPolicy>(p), begin(in),
                               end(in), begin(out), binop, value);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::RandomAccessRange InContainer,
         concepts::RandomAccessRange OutContainer,
         typename T        = RAJA::detail::ContainerVal<InContainer>,
         typename Function = operators::plus<T>,
         typename Res      = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan(
    ExecPolicy&& p,
    InContainer&& in,
    OutContainer&& out,
    Function binop = Function {},
    T value        = Function::identity())
{
  auto r = Res::get_default();
  return ::RAJA::policy_by_value_interface::exclusive_scan(
      std::forward<ExecPolicy>(p), r, std::forward<InContainer>(in),
      std::forward<OutContainer>(out), binop, value);
}

}  // namespace policy_by_value_interface

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * exclusive_scan
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<concepts::ExecutionPolicy ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::exclusive_scan<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         typename... Args>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::exclusive_scan(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * inclusive_scan
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<concepts::ExecutionPolicy ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::inclusive_scan<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         typename... Args>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan(Res r, Args&&... args)
{
  return ::RAJA::policy_by_value_interface::inclusive_scan(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * exclusive_scan_inplace
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<concepts::ExecutionPolicy ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan_inplace(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::exclusive_scan_inplace<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         typename... Args>
RAJA_INLINE resources::EventProxy<Res> exclusive_scan_inplace(Res r,
                                                              Args&&... args)
{
  return ::RAJA::policy_by_value_interface::exclusive_scan_inplace(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * inclusive_scan_inplace
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template<concepts::ExecutionPolicy ExecPolicy,
         typename... Args,
         typename Res = typename resources::get_resource<ExecPolicy>::type>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan_inplace(Args&&... args)
{
  Res r = Res::get_default();
  return ::RAJA::policy_by_value_interface::inclusive_scan_inplace<ExecPolicy>(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

///
template<concepts::ExecutionPolicy ExecPolicy,
         concepts::Resource Res,
         typename... Args>
RAJA_INLINE resources::EventProxy<Res> inclusive_scan_inplace(Res r,
                                                              Args&&... args)
{
  return ::RAJA::policy_by_value_interface::inclusive_scan_inplace(
      ExecPolicy(), r, std::forward<Args>(args)...);
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
