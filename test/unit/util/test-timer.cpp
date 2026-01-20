//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing unit tests for Timer class
///

#include "RAJA_test-base.hpp"

#include "RAJA/util/Timer.hpp"

#include <iostream>
#include <sstream>
#include <string>

#include <chrono>
#include <thread>


TEST(TimerUnitTest, No1)
{
  auto timer = RAJA::Timer();

  timer.start("test_timer");

  {
    std::stringstream sink;
    sink << "Printing 1000 stars...\n";
    for (int i = 0; i < 1000; ++i)
      sink << "*";
    sink << std::endl;
  }

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_GT(elapsed, 0.0);

  // std::cout << "Printing 1000 stars took " << elapsed << " seconds.";
}


TEST(TimerUnitTest, No2)
{
  RAJA::Timer timer;

  timer.start("test_timer");

  for (int i = 2; i > 0; --i) {
    std::cout << i << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_GT(elapsed, 0.02);
#if !defined(__APPLE__)
  EXPECT_LT(elapsed, 0.05);
#endif
}


TEST(TimerUnitTest, No3)
{
  RAJA::Timer timer;

  timer.start("test_timer");

  for (int i = 2; i > 0; --i) {
    std::cout << i << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  timer.stop();

  RAJA::Timer::ElapsedType elapsed = timer.elapsed();

  EXPECT_GT(elapsed, 0.02);
#if !defined(__APPLE__)
  EXPECT_LT(elapsed, 0.05);
#endif

  timer.reset();
  elapsed = timer.elapsed();
  ASSERT_EQ(0, elapsed);

  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  timer.stop();
  elapsed = timer.elapsed();
  EXPECT_GT(elapsed, 0.01); 
}
