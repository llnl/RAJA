//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2018-25, Lawrence Livermore National Security, LLC
// and Camp project contributors. See the camp/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/array.hpp"
#include "RAJA_test-base.hpp"

#include "gtest/gtest.h"

TEST(message_handler, initialize) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(msg_manager.test_any(), false);
  ASSERT_EQ(test, 0);
} 

TEST(message_handler, initialize_with_resource) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager(msg_sz, camp::resources::Host());

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(msg_manager.test_any(), false);
  ASSERT_EQ(test, 0);
} 

TEST(message_handler, clear) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(5), true);

  msg_manager.clear();
  msg_manager.wait_all();

  ASSERT_EQ(test, 0);
}

TEST(message_handler, try_post_message) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, try_post_message_overflow) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(5), true);
  ASSERT_EQ(q.try_post_message(7), false);

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, try_post_message_overwrite) {
  // TODO: implement
} 

TEST(message_handler, wait_all) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(1), true);

  msg_manager.wait_all();

  ASSERT_EQ(test, 1);
}

TEST(message_handler, wait_all_overalloc) {
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<int>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(2*msg_sz);

  int test = 0;
  auto q = msg_manager.get_queue<RAJA::spsc_queue>(msg_id, [&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(1), true);

  msg_manager.wait_all();

  ASSERT_EQ(test, 1);
}

TEST(message_handler, wait_all_array) {
#if 0
  constexpr std::size_t msg_sz = sizeof(RAJA::msg_header) +
                                 sizeof(RAJA::msg_args<camp::array<int, 3>>);
  constexpr int msg_id         = 0;

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  camp::array<int, 3> test = {0, 0, 0};
  auto q = msg_manager.get_queue<RAJA::mpsc_queue>(1, 
    [&](camp::array<int, 3> val) {
      test[0] = val[0];   
      test[1] = val[1];
      test[2] = val[2];
    }
  );

  camp::array<int, 3> a{1,2,3};
  ASSERT_EQ(q.try_post_message(a), true);

  msg_manager.wait_all();

  ASSERT_EQ(test[0], 1);
  ASSERT_EQ(test[1], 2);
  ASSERT_EQ(test[2], 3);
#endif
}

TEST(message_handler, wait_all_overflow) {
  // TODO: implement 
}

