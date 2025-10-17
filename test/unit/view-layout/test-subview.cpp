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

template<typename ViewType, typename... Slices>
auto make_subview(ViewType view, Slices... slices) {
    using SubLayoutType = SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
    return View<Index_type, SubLayoutType>(view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

template<typename ViewType, typename... Slices>
auto make_submultiview(ViewType view, Slices... slices) {
    using SubLayoutType = SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
    return MultiView<Index_type, SubLayoutType>(view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

TEST(SubView, RangeSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[1:3]
    auto sv = make_subview(view, RangeSlice<>{1,3});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0), 2);
    EXPECT_EQ(sv(1), 3);
    EXPECT_EQ(sv(2), 4);

    EXPECT_EQ(sublayout(0), 1);
    EXPECT_EQ(sublayout(1), 2);
    EXPECT_EQ(sublayout(2), 3);

    EXPECT_EQ(sublayout.size(), 3);
}

TEST(SubView, RangeStartSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[2:]
    auto sv = make_subview(view, RangeStartSlice<>{2});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0), 3);
    EXPECT_EQ(sv(1), 4);
    EXPECT_EQ(sv(2), 5);

    EXPECT_EQ(sublayout.size(), 3);
}

TEST(SubView, StridedSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[0:3:2]
    auto sv = make_subview(view, StridedSlice<>{0,3,2});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0), 1);
    EXPECT_EQ(sv(1), 3);

    EXPECT_EQ(sublayout(0), 0);
    EXPECT_EQ(sublayout(1), 2);

    EXPECT_EQ(sublayout.size(), 2);
}

TEST(SubView, FixedSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[1]
    auto sv = make_subview(view, FixedSlice<>{1});
    auto sublayout = sv.get_layout();

    // 0D views don't work
    // EXPECT_EQ(sv(), 2);
    EXPECT_EQ(sublayout(), 1);

    EXPECT_EQ(sublayout.size(), 1);
    EXPECT_EQ(sublayout.size_noproj(), 1);
}

TEST(SubView, RangeSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,1:2]
    auto sv = make_subview(view, RangeSlice<>{1,2}, RangeSlice<>{1,2});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0,0), 5);
    EXPECT_EQ(sv(0,1), 6);
    EXPECT_EQ(sv(1,0), 8);
    EXPECT_EQ(sv(1,1), 9);

    EXPECT_EQ(sublayout.size(), 4);
    EXPECT_EQ(sublayout.get_dim_size<0>(), 2);
    EXPECT_EQ(sublayout.get_dim_size<1>(), 2);
    EXPECT_EQ(sublayout.get_dim_stride<0>(), 1);
    EXPECT_EQ(sublayout.get_dim_stride<1>(), 1);

}

TEST(SubView, RangeFixedSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,1]
    auto sv = make_subview(view, RangeSlice<>{1,2}, FixedSlice<>{1});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0), 5);
    EXPECT_EQ(sv(1), 8);

    EXPECT_EQ(sublayout.size(), 2);
    EXPECT_EQ(sublayout.get_dim_size<0>(), 2);
    EXPECT_EQ(sublayout.get_dim_stride<0>(), 1);

}

TEST(SubView, FixedFirstDimSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1,:]
    auto sv = make_subview(view, FixedSlice<>{1}, NoSlice{});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0), 4);
    EXPECT_EQ(sv(1), 5);
    EXPECT_EQ(sv(2), 6);

    EXPECT_EQ(sublayout.size(), 3);
    EXPECT_EQ(sublayout.get_dim_size<0>(), 3);
    EXPECT_EQ(sublayout.get_dim_stride<0>(), 1);
}

TEST(SubView, RangeFirstDimSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,:]
    auto sv = make_subview(view, RangeSlice<>{1,2}, NoSlice{});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0,0), 4);
    EXPECT_EQ(sv(0,1), 5);
    EXPECT_EQ(sv(0,2), 6);

    EXPECT_EQ(sv(1,0), 7);
    EXPECT_EQ(sv(1,1), 8);
    EXPECT_EQ(sv(1,2), 9);

    EXPECT_EQ(sublayout.size(), 6);
    EXPECT_EQ(sublayout.get_dim_size<0>(), 2);
    EXPECT_EQ(sublayout.get_dim_size<1>(), 3);
    EXPECT_EQ(sublayout.get_dim_stride<0>(), 1);
    EXPECT_EQ(sublayout.get_dim_stride<1>(), 1);

}

TEST(SubView, RangeFirstDimStridedSecondDimSubView2D)
{

    Index_type a[3][6] = {{1,2,3,4,5,6},
                          {7,8,9,10,11,12},
                          {13,14,15,16,17,18}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,6));

    // sv = View[1:2,1:5:2]
    auto sv = make_subview(view, RangeSlice<>{1,2}, StridedSlice<>{1,5,2});
    auto sublayout = sv.get_layout();

    EXPECT_EQ(sv(0,0), 8);
    EXPECT_EQ(sv(0,1), 10);
    EXPECT_EQ(sv(0,2), 12);

    EXPECT_EQ(sv(1,0), 14);
    EXPECT_EQ(sv(1,1), 16);
    EXPECT_EQ(sv(1,2), 18);

    EXPECT_EQ(sublayout.size(), 6);
    EXPECT_EQ(sublayout.get_dim_size<0>(), 2);
    EXPECT_EQ(sublayout.get_dim_size<1>(), 3);
    EXPECT_EQ(sublayout.get_dim_stride<0>(), 1);
    EXPECT_EQ(sublayout.get_dim_stride<1>(), 2);

}

TEST(SubView, SubViewOfSubView2D)
{

    Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,:]
    auto sv = make_subview(view, RangeSlice<>{1,2}, NoSlice{});


    EXPECT_EQ(sv.get_layout().size(), 6);

    // sv2 = sv[0:1,1:2]
    auto sv2 = make_subview(sv, RangeSlice<>{0,1}, RangeSlice<>{1,2});

    EXPECT_EQ(sv2(0,0), 5);
    EXPECT_EQ(sv2(0,1), 6);

    EXPECT_EQ(sv2(1,0), 8);
    EXPECT_EQ(sv2(1,1), 9);

    EXPECT_EQ(sv2.get_layout().size(), 4);

    // sv3 = sv2[:,1]
    auto sv3 = make_subview(sv2, NoSlice{}, FixedSlice<>{1});

    EXPECT_EQ(sv3(0), 6);
    EXPECT_EQ(sv3(1), 9);

    EXPECT_EQ(sv3.get_layout().size(), 2);

}

TEST(SubView, SubViewOfMultiView2D)
{

    Index_type data_squared[4];
    Index_type data_cubed[4];

    for (int i = 0; i < 4; i ++ ) {
        data_squared[i] = i*i;
    }
    
    for (int i = 0; i < 4; i ++ ) {
        data_cubed[i] = i*i*i;
    }

    Index_type* data_array[2];
    data_array[0] = data_squared;
    data_array[1] = data_cubed;

    Index_type index_list[4] = {3,1,2,0};

    auto index_tuple = make_index_tuple(IndexList<>{&index_list[0]});
    auto index_layout = make_index_layout(index_tuple, 4);

    auto view = MultiView<Index_type, IndexLayout<1, Index_type, IndexList<> > >(data_array, index_layout);

    // sv = MultiView[:,1:2]
    auto sv = make_submultiview(view, RangeSlice<>{1,2});

    EXPECT_EQ(sv.get_layout().size(), 2);
    EXPECT_EQ(sv.get_layout().get_dim_size<0>(), 2);

    EXPECT_EQ(sv(0,0), 1);
    EXPECT_EQ(sv(0,1), 4);

    EXPECT_EQ(sv(1,0), 1);
    EXPECT_EQ(sv(1,1), 8);

    // sv = MultiView[:,2]
    auto sv2 = make_submultiview(view, FixedSlice<>{2});

    // this size does not look right
    EXPECT_EQ(sv2.get_layout().size(), 1);

    EXPECT_EQ(sv2(0), 4);
    EXPECT_EQ(sv2(1), 8);

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