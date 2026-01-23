/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining permutations
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

#ifndef RAJA_FORALLN_PERMUTATIONS_HPP
#define RAJA_FORALLN_PERMUTATIONS_HPP

#include "RAJA/config.hpp"

#include <array>

#include "camp/camp.hpp"

namespace RAJA
{

template<typename Indices>
struct as_array;

template<camp::idx_t... Indices>
struct as_array<camp::idx_seq<Indices...>>
{
  static constexpr std::array<Index_type, sizeof...(Indices)> get()
  {
    return {{Indices...}};
  }
};

using PERM_I     = camp::idx_seq<0>;
using PERM_IJ    = camp::idx_seq<0, 1>;
using PERM_JI    = camp::idx_seq<1, 0>;
using PERM_IJK   = camp::idx_seq<0, 1, 2>;
using PERM_IKJ   = camp::idx_seq<0, 2, 1>;
using PERM_JIK   = camp::idx_seq<1, 0, 2>;
using PERM_JKI   = camp::idx_seq<1, 2, 0>;
using PERM_KIJ   = camp::idx_seq<2, 0, 1>;
using PERM_KJI   = camp::idx_seq<2, 1, 0>;
using PERM_IJKL  = camp::idx_seq<0, 1, 2, 3>;
using PERM_IJLK  = camp::idx_seq<0, 1, 3, 2>;
using PERM_IKJL  = camp::idx_seq<0, 2, 1, 3>;
using PERM_IKLJ  = camp::idx_seq<0, 2, 3, 1>;
using PERM_ILJK  = camp::idx_seq<0, 3, 1, 2>;
using PERM_ILKJ  = camp::idx_seq<0, 3, 2, 1>;
using PERM_JIKL  = camp::idx_seq<1, 0, 2, 3>;
using PERM_JILK  = camp::idx_seq<1, 0, 3, 2>;
using PERM_JKIL  = camp::idx_seq<1, 2, 0, 3>;
using PERM_JKLI  = camp::idx_seq<1, 2, 3, 0>;
using PERM_JLIK  = camp::idx_seq<1, 3, 0, 2>;
using PERM_JLKI  = camp::idx_seq<1, 3, 2, 0>;
using PERM_KIJL  = camp::idx_seq<2, 0, 1, 3>;
using PERM_KILJ  = camp::idx_seq<2, 0, 3, 1>;
using PERM_KJIL  = camp::idx_seq<2, 1, 0, 3>;
using PERM_KJLI  = camp::idx_seq<2, 1, 3, 0>;
using PERM_KLIJ  = camp::idx_seq<2, 3, 0, 1>;
using PERM_KLJI  = camp::idx_seq<2, 3, 1, 0>;
using PERM_LIJK  = camp::idx_seq<3, 0, 1, 2>;
using PERM_LIKJ  = camp::idx_seq<3, 0, 2, 1>;
using PERM_LJIK  = camp::idx_seq<3, 1, 0, 2>;
using PERM_LJKI  = camp::idx_seq<3, 1, 2, 0>;
using PERM_LKIJ  = camp::idx_seq<3, 2, 0, 1>;
using PERM_LKJI  = camp::idx_seq<3, 2, 1, 0>;
using PERM_IJKLM = camp::idx_seq<0, 1, 2, 3, 4>;
using PERM_IJKML = camp::idx_seq<0, 1, 2, 4, 3>;
using PERM_IJLKM = camp::idx_seq<0, 1, 3, 2, 4>;
using PERM_IJLMK = camp::idx_seq<0, 1, 3, 4, 2>;
using PERM_IJMKL = camp::idx_seq<0, 1, 4, 2, 3>;
using PERM_IJMLK = camp::idx_seq<0, 1, 4, 3, 2>;
using PERM_IKJLM = camp::idx_seq<0, 2, 1, 3, 4>;
using PERM_IKJML = camp::idx_seq<0, 2, 1, 4, 3>;
using PERM_IKLJM = camp::idx_seq<0, 2, 3, 1, 4>;
using PERM_IKLMJ = camp::idx_seq<0, 2, 3, 4, 1>;
using PERM_IKMJL = camp::idx_seq<0, 2, 4, 1, 3>;
using PERM_IKMLJ = camp::idx_seq<0, 2, 4, 3, 1>;
using PERM_ILJKM = camp::idx_seq<0, 3, 1, 2, 4>;
using PERM_ILJMK = camp::idx_seq<0, 3, 1, 4, 2>;
using PERM_ILKJM = camp::idx_seq<0, 3, 2, 1, 4>;
using PERM_ILKMJ = camp::idx_seq<0, 3, 2, 4, 1>;
using PERM_ILMJK = camp::idx_seq<0, 3, 4, 1, 2>;
using PERM_ILMKJ = camp::idx_seq<0, 3, 4, 2, 1>;
using PERM_IMJKL = camp::idx_seq<0, 4, 1, 2, 3>;
using PERM_IMJLK = camp::idx_seq<0, 4, 1, 3, 2>;
using PERM_IMKJL = camp::idx_seq<0, 4, 2, 1, 3>;
using PERM_IMKLJ = camp::idx_seq<0, 4, 2, 3, 1>;
using PERM_IMLJK = camp::idx_seq<0, 4, 3, 1, 2>;
using PERM_IMLKJ = camp::idx_seq<0, 4, 3, 2, 1>;
using PERM_JIKLM = camp::idx_seq<1, 0, 2, 3, 4>;
using PERM_JIKML = camp::idx_seq<1, 0, 2, 4, 3>;
using PERM_JILKM = camp::idx_seq<1, 0, 3, 2, 4>;
using PERM_JILMK = camp::idx_seq<1, 0, 3, 4, 2>;
using PERM_JIMKL = camp::idx_seq<1, 0, 4, 2, 3>;
using PERM_JIMLK = camp::idx_seq<1, 0, 4, 3, 2>;
using PERM_JKILM = camp::idx_seq<1, 2, 0, 3, 4>;
using PERM_JKIML = camp::idx_seq<1, 2, 0, 4, 3>;
using PERM_JKLIM = camp::idx_seq<1, 2, 3, 0, 4>;
using PERM_JKLMI = camp::idx_seq<1, 2, 3, 4, 0>;
using PERM_JKMIL = camp::idx_seq<1, 2, 4, 0, 3>;
using PERM_JKMLI = camp::idx_seq<1, 2, 4, 3, 0>;
using PERM_JLIKM = camp::idx_seq<1, 3, 0, 2, 4>;
using PERM_JLIMK = camp::idx_seq<1, 3, 0, 4, 2>;
using PERM_JLKIM = camp::idx_seq<1, 3, 2, 0, 4>;
using PERM_JLKMI = camp::idx_seq<1, 3, 2, 4, 0>;
using PERM_JLMIK = camp::idx_seq<1, 3, 4, 0, 2>;
using PERM_JLMKI = camp::idx_seq<1, 3, 4, 2, 0>;
using PERM_JMIKL = camp::idx_seq<1, 4, 0, 2, 3>;
using PERM_JMILK = camp::idx_seq<1, 4, 0, 3, 2>;
using PERM_JMKIL = camp::idx_seq<1, 4, 2, 0, 3>;
using PERM_JMKLI = camp::idx_seq<1, 4, 2, 3, 0>;
using PERM_JMLIK = camp::idx_seq<1, 4, 3, 0, 2>;
using PERM_JMLKI = camp::idx_seq<1, 4, 3, 2, 0>;
using PERM_KIJLM = camp::idx_seq<2, 0, 1, 3, 4>;
using PERM_KIJML = camp::idx_seq<2, 0, 1, 4, 3>;
using PERM_KILJM = camp::idx_seq<2, 0, 3, 1, 4>;
using PERM_KILMJ = camp::idx_seq<2, 0, 3, 4, 1>;
using PERM_KIMJL = camp::idx_seq<2, 0, 4, 1, 3>;
using PERM_KIMLJ = camp::idx_seq<2, 0, 4, 3, 1>;
using PERM_KJILM = camp::idx_seq<2, 1, 0, 3, 4>;
using PERM_KJIML = camp::idx_seq<2, 1, 0, 4, 3>;
using PERM_KJLIM = camp::idx_seq<2, 1, 3, 0, 4>;
using PERM_KJLMI = camp::idx_seq<2, 1, 3, 4, 0>;
using PERM_KJMIL = camp::idx_seq<2, 1, 4, 0, 3>;
using PERM_KJMLI = camp::idx_seq<2, 1, 4, 3, 0>;
using PERM_KLIJM = camp::idx_seq<2, 3, 0, 1, 4>;
using PERM_KLIMJ = camp::idx_seq<2, 3, 0, 4, 1>;
using PERM_KLJIM = camp::idx_seq<2, 3, 1, 0, 4>;
using PERM_KLJMI = camp::idx_seq<2, 3, 1, 4, 0>;
using PERM_KLMIJ = camp::idx_seq<2, 3, 4, 0, 1>;
using PERM_KLMJI = camp::idx_seq<2, 3, 4, 1, 0>;
using PERM_KMIJL = camp::idx_seq<2, 4, 0, 1, 3>;
using PERM_KMILJ = camp::idx_seq<2, 4, 0, 3, 1>;
using PERM_KMJIL = camp::idx_seq<2, 4, 1, 0, 3>;
using PERM_KMJLI = camp::idx_seq<2, 4, 1, 3, 0>;
using PERM_KMLIJ = camp::idx_seq<2, 4, 3, 0, 1>;
using PERM_KMLJI = camp::idx_seq<2, 4, 3, 1, 0>;
using PERM_LIJKM = camp::idx_seq<3, 0, 1, 2, 4>;
using PERM_LIJMK = camp::idx_seq<3, 0, 1, 4, 2>;
using PERM_LIKJM = camp::idx_seq<3, 0, 2, 1, 4>;
using PERM_LIKMJ = camp::idx_seq<3, 0, 2, 4, 1>;
using PERM_LIMJK = camp::idx_seq<3, 0, 4, 1, 2>;
using PERM_LIMKJ = camp::idx_seq<3, 0, 4, 2, 1>;
using PERM_LJIKM = camp::idx_seq<3, 1, 0, 2, 4>;
using PERM_LJIMK = camp::idx_seq<3, 1, 0, 4, 2>;
using PERM_LJKIM = camp::idx_seq<3, 1, 2, 0, 4>;
using PERM_LJKMI = camp::idx_seq<3, 1, 2, 4, 0>;
using PERM_LJMIK = camp::idx_seq<3, 1, 4, 0, 2>;
using PERM_LJMKI = camp::idx_seq<3, 1, 4, 2, 0>;
using PERM_LKIJM = camp::idx_seq<3, 2, 0, 1, 4>;
using PERM_LKIMJ = camp::idx_seq<3, 2, 0, 4, 1>;
using PERM_LKJIM = camp::idx_seq<3, 2, 1, 0, 4>;
using PERM_LKJMI = camp::idx_seq<3, 2, 1, 4, 0>;
using PERM_LKMIJ = camp::idx_seq<3, 2, 4, 0, 1>;
using PERM_LKMJI = camp::idx_seq<3, 2, 4, 1, 0>;
using PERM_LMIJK = camp::idx_seq<3, 4, 0, 1, 2>;
using PERM_LMIKJ = camp::idx_seq<3, 4, 0, 2, 1>;
using PERM_LMJIK = camp::idx_seq<3, 4, 1, 0, 2>;
using PERM_LMJKI = camp::idx_seq<3, 4, 1, 2, 0>;
using PERM_LMKIJ = camp::idx_seq<3, 4, 2, 0, 1>;
using PERM_LMKJI = camp::idx_seq<3, 4, 2, 1, 0>;
using PERM_MIJKL = camp::idx_seq<4, 0, 1, 2, 3>;
using PERM_MIJLK = camp::idx_seq<4, 0, 1, 3, 2>;
using PERM_MIKJL = camp::idx_seq<4, 0, 2, 1, 3>;
using PERM_MIKLJ = camp::idx_seq<4, 0, 2, 3, 1>;
using PERM_MILJK = camp::idx_seq<4, 0, 3, 1, 2>;
using PERM_MILKJ = camp::idx_seq<4, 0, 3, 2, 1>;
using PERM_MJIKL = camp::idx_seq<4, 1, 0, 2, 3>;
using PERM_MJILK = camp::idx_seq<4, 1, 0, 3, 2>;
using PERM_MJKIL = camp::idx_seq<4, 1, 2, 0, 3>;
using PERM_MJKLI = camp::idx_seq<4, 1, 2, 3, 0>;
using PERM_MJLIK = camp::idx_seq<4, 1, 3, 0, 2>;
using PERM_MJLKI = camp::idx_seq<4, 1, 3, 2, 0>;
using PERM_MKIJL = camp::idx_seq<4, 2, 0, 1, 3>;
using PERM_MKILJ = camp::idx_seq<4, 2, 0, 3, 1>;
using PERM_MKJIL = camp::idx_seq<4, 2, 1, 0, 3>;
using PERM_MKJLI = camp::idx_seq<4, 2, 1, 3, 0>;
using PERM_MKLIJ = camp::idx_seq<4, 2, 3, 0, 1>;
using PERM_MKLJI = camp::idx_seq<4, 2, 3, 1, 0>;
using PERM_MLIJK = camp::idx_seq<4, 3, 0, 1, 2>;
using PERM_MLIKJ = camp::idx_seq<4, 3, 0, 2, 1>;
using PERM_MLJIK = camp::idx_seq<4, 3, 1, 0, 2>;
using PERM_MLJKI = camp::idx_seq<4, 3, 1, 2, 0>;
using PERM_MLKIJ = camp::idx_seq<4, 3, 2, 0, 1>;
using PERM_MLKJI = camp::idx_seq<4, 3, 2, 1, 0>;

namespace internal
{


template<camp::idx_t I, camp::idx_t J, camp::idx_t N, typename Perm>
struct CalcInversePermutationElem
{
  static constexpr camp::idx_t value =
      camp::seq_at<J, Perm>::value == I
          ? J
          : CalcInversePermutationElem<I, J + 1, N, Perm>::value;
};

template<camp::idx_t I, camp::idx_t N, typename Perm>
struct CalcInversePermutationElem<I, N, N, Perm>
{
  static constexpr camp::idx_t value = I;
};


template<typename Range, typename Perm>
struct InversePermutationHelper;

template<camp::idx_t... Range, camp::idx_t... Perm>
struct InversePermutationHelper<camp::idx_seq<Range...>, camp::idx_seq<Perm...>>
{
  static_assert(sizeof...(Range) == sizeof...(Perm), "Fatal Error");
  using type = camp::idx_seq<
      CalcInversePermutationElem<Range,
                                 0,
                                 sizeof...(Range),
                                 camp::idx_seq<Perm...>>::value...>;
};


}  // namespace internal

/*!
  Inverts a permutation
*/
template<typename Perm>
using invert_permutation = typename internal::InversePermutationHelper<
    camp::make_idx_seq_t<camp::size<Perm>::value>,
    Perm>::type;

}  // namespace RAJA

#endif /* RAJA_FORALLN_PERMUTATIONS_HPP */
