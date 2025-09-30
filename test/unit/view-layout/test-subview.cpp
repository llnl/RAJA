//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <gtest/gtest.h>
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/SubView.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-forone.hpp"

using namespace RAJA;

TEST(SubView, RangeSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[1:3]
    auto sv = SubView(view, RangeSlice<>{1,3});

    EXPECT_EQ(sv(0), 2);
    EXPECT_EQ(sv(1), 3);
    EXPECT_EQ(sv(2), 4);

}

TEST(SubView, RangeSubView2D)
{

    Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,1:2]
    auto sv = SubView(view, RangeSlice<>{1,2}, RangeSlice<>{1,2});

    EXPECT_EQ(sv(0,0), 5);
    EXPECT_EQ(sv(0,1), 6);
    EXPECT_EQ(sv(1,0), 8);
    EXPECT_EQ(sv(1,1), 9);

}

TEST(SubView, RangeFixedSubView2D)
{

    Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,1]
    auto sv = SubView(view, RangeSlice<>{1,2}, FixedSlice<>{1});

    EXPECT_EQ(sv(0), 5);
    EXPECT_EQ(sv(1), 8);

}

TEST(SubView, FixedFirstDimSubView2D)
{

    Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1,:]
    auto sv = SubView(view, FixedSlice<>{1}, NoSlice{});

    EXPECT_EQ(sv(0), 4);
    EXPECT_EQ(sv(1), 5);
    EXPECT_EQ(sv(2), 6);

}

TEST(SubView, RangeFirstDimSubView2D)
{

    Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,:]
    auto sv = SubView(view, RangeSlice<>{1,2}, NoSlice{});

    EXPECT_EQ(sv(0,0), 4);
    EXPECT_EQ(sv(0,1), 5);
    EXPECT_EQ(sv(0,2), 6);

    EXPECT_EQ(sv(1,0), 7);
    EXPECT_EQ(sv(1,1), 8);
    EXPECT_EQ(sv(1,2), 9);

}

// void test_subviewGPU() {
// #if defined(RAJA_ENABLE_HIP)
//     forone<test_hip>([=] __host__ __device__ () {
//         Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

//         View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

//         // sv = View[1:2,:]
//         auto sv = SubView(view, RangeSlice<>{1,2}, NoSlice{});

//         //printf("sv(0,0): %ld\n", sv(0,0));

//     });
// #endif
// }

// TEST(SubView, RangeFirstDimSubView2DGPU)
// {
//     test_subviewGPU();
// }