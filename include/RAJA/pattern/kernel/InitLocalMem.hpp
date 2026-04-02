/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for shared memory window.
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

#ifndef RAJA_pattern_kernel_InitLocalMem_HPP
#define RAJA_pattern_kernel_InitLocalMem_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>
#include <memory>

namespace RAJA
{

// Policies for RAJA local arrays
struct cpu_tile_mem;

namespace statement
{


/*!
 * Kernel statements intialize local arrays
 * The statement takes parameter sequence to intialize
 * memory from the parameter tuple and additional statements.
 *
 * For example:
 * IntiLocalMem<Pol, RAJA::param_idx<0>, statements...>
 * Will intialize the 0th array in the param tuple
 */
template<typename Pol, typename Indices, typename... EnclosedStmts>
struct InitLocalMem : public internal::Statement<camp::nil>
{};

// Policy Specialization
template<camp::idx_t... Indices, typename... EnclosedStmts>
struct InitLocalMem<RAJA::cpu_tile_mem,
                    camp::idx_seq<Indices...>,
                    EnclosedStmts...> : public internal::Statement<camp::nil>
{};


}  // end namespace statement

namespace internal
{

// Statement executor to initalize RAJA local array
template<camp::idx_t... Indices, typename... EnclosedStmts, typename Types>
struct StatementExecutor<statement::InitLocalMem<RAJA::cpu_tile_mem,
                                                 camp::idx_seq<Indices...>,
                                                 EnclosedStmts...>,
                         Types>
{

  // Execute statement list
  template<class Data>
  static void RAJA_INLINE exec_expanded(Data&& data)
  {
    execute_statement_list<camp::list<EnclosedStmts...>, Types>(data);
  }

  // Intialize local array
  // Identifies type + number of elements needed
  template<camp::idx_t Pos, camp::idx_t... others, class Data>
  static void RAJA_INLINE exec_expanded(Data&& data)
  {
    using varType = typename camp::tuple_element_t<
        Pos, typename camp::decay<Data>::param_tuple_t>::value_type;

    // Initialize memory
    auto local_mem =
        std::make_unique<varType[]>(camp::get<Pos>(data.param_tuple).size());
    camp::get<Pos>(data.param_tuple).set_data(local_mem.get());

    // Initialize others and execute
    exec_expanded<others...>(data);

    // Cleanup and return
    camp::get<Pos>(data.param_tuple).set_data(nullptr);
  }

  template<typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    // Initalize local arrays + execute statements + cleanup
    exec_expanded<Indices...>(data);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
