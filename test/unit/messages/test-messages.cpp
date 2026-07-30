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
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(msg_manager.test_any(), false);
  ASSERT_EQ(test, 0);
} 

TEST(message_handler, initialize_with_resource) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager(msg_sz, camp::resources::Host());

  int test = 0;
  msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(msg_manager.test_any(), false);
  ASSERT_EQ(test, 0);
} 

TEST(message_handler, clear) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(5), true);

  msg_manager.clear();
  msg_manager.wait_all();

  ASSERT_EQ(test, 0);
}

TEST(message_handler, try_post_message) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, try_post_message_overflow) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(5), true);
  ASSERT_EQ(q.try_post_message(7), false);

  ASSERT_EQ(test, 0);
} 

TEST(message_handler, wait_all) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(1), true);

  msg_manager.wait_all();

  ASSERT_EQ(test, 1);
}

TEST(message_handler, wait_all_overalloc) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(2*msg_sz);

  int test = 0;
  auto q = msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(1), true);

  msg_manager.wait_all();

  ASSERT_EQ(test, 1);
}

TEST(message_handler, wait_all_array) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<camp::array<int, 3>>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  camp::array<int, 3> test = {0, 0, 0};
  auto q = msg_manager.subscribe<RAJA::mpsc_queue>( 
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
}

TEST(message_handler, wait_all_overflow) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test = val;   
  });

  ASSERT_EQ(q.try_post_message(5), true);
  ASSERT_EQ(q.try_post_message(7), false);

  msg_manager.wait_all();

  ASSERT_EQ(test, 5);
}

TEST(message_handler, subscribe) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto q = msg_manager.subscribe<RAJA::spsc_queue>([&] (int val) {
    test += val;
  });
  msg_manager.subscribe(q.get_id(), [&] (int val) {
    test *= val;
  });

  ASSERT_EQ(q.try_post_message(5), true);

  msg_manager.wait_all();

  ASSERT_EQ(test, 25);
} 

TEST(message_handler, unsubscribe) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(msg_sz);

  int test = 0;
  auto update = [&](int val) { test = val; };
  auto q = msg_manager.subscribe<RAJA::spsc_queue>(update);
  msg_manager.subscribe(q.get_id(), [&] (int val) {
    test += val;
  });

  ASSERT_EQ(q.try_post_message(1), true);

  msg_manager.unsubscribe(q.get_id(), update);
  msg_manager.wait_all();

  ASSERT_EQ(test, 1);
} 

TEST(message_handler, unsubscribe_all_id) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(2*msg_sz);

  int test1 = 0;
  int test2 = 0;
  auto q1 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test1 = 1;
  });
  auto q2 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test2 = 2;
  });

  ASSERT_EQ(q1.try_post_message(), true);
  ASSERT_EQ(q2.try_post_message(), true);

  msg_manager.unsubscribe_all(q1.get_id());
  msg_manager.wait_all();

  ASSERT_EQ(test1, 0);
  ASSERT_EQ(test2, 2);

  // Re-subscribe to same queue id
  msg_manager.subscribe(q1.get_id(), [&]() {
    test1 = 3;
  });
  ASSERT_EQ(q1.try_post_message(), true);
  msg_manager.wait_all();

  ASSERT_EQ(test1, 3);
} 

TEST(message_handler, unsubscribe_all) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(2*msg_sz);

  int test1 = 0;
  int test2 = 0;
  auto q1 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test1 = 1;
  });
  auto q2 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test2 = 2;
  });

  ASSERT_EQ(q1.try_post_message(), true);
  ASSERT_EQ(q2.try_post_message(), true);

  msg_manager.unsubscribe_all();
  msg_manager.wait_all();

  ASSERT_EQ(test1, 0);
  ASSERT_EQ(test2, 0);
} 

TEST(message_handler, erase_all_id) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(2*msg_sz);

  int test1 = 0;
  auto q1 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test1 = 1;
  });

  ASSERT_EQ(q1.try_post_message(), true);

  msg_manager.erase_all(q1.get_id());
  msg_manager.wait_all();

  ASSERT_EQ(test1, 0);
}

TEST(message_handler, get_messages) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<int>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(20*msg_sz);

  int test1 = 0;
  auto q1 = msg_manager.subscribe<RAJA::spsc_queue>([&](int val) {
    test1 = val;
  });

  ASSERT_EQ(q1.try_post_message(1), true);
  ASSERT_EQ(q1.try_post_message(2), true);
  ASSERT_EQ(q1.try_post_message(3), true);


  auto msg_list = msg_manager.get_messages();
  ASSERT_EQ(msg_list.size(), 3);
  ASSERT_EQ(test1, 0);
} 

TEST(message_handler, handle_all_sort) {
  constexpr std::size_t msg_sz = RAJA::align_sz(sizeof(RAJA::MsgHeader)) +
                                 RAJA::align_sz(sizeof(RAJA::MsgArgs<>));

  auto msg_manager = RAJA::make_message_manager<RAJA::seq_exec>(20*msg_sz);

  int test1 = 0;
  auto q1 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test1 = 1;
  });
  auto q2 = msg_manager.subscribe<RAJA::spsc_queue>([&]() {
    test1 = 2;
  });

  ASSERT_EQ(q1.try_post_message(), true);
  ASSERT_EQ(q2.try_post_message(), true);
  ASSERT_EQ(q1.try_post_message(), true);

  auto msg_list = msg_manager.get_messages();
  ASSERT_EQ(msg_list.size(), 3);

  // Forces all q2 messages to the end
  std::sort(msg_list.begin(), msg_list.end(), [] (auto msg1, auto msg2) {
    return msg1->type < msg2->type;
  });
  msg_manager.handle_all(msg_list);

  ASSERT_EQ(test1, 2);
} 
