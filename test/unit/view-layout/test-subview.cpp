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

/* helper to create a RAJA View with a sliced SubLayout */
template<typename ViewType, typename... Slices>
auto make_view_with_sublayout(ViewType& view, Slices... slices) {
    using SubLayoutType = SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
    return View<Index_type, SubLayoutType>(view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

/* helper to create a sliced SubView without modifying underlying layout */
template<typename ViewType, typename... Slices>
auto make_subview_with_layout(ViewType& view, Slices... slices) {
    using SubViewType = SubView<ViewType, camp::list<Slices...>>;
    return SubViewType(view, slices...);
}

template<typename ViewType, typename... Slices>
auto make_multiview_with_sublayout(ViewType& view, Slices... slices) {
    using SubLayoutType = SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
    return MultiView<Index_type, SubLayoutType>(view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

struct UseViewWithSubLayout {

    template <typename ViewType, typename... Slices>
    auto operator()(ViewType& view, Slices... slices) const {
        return make_view_with_sublayout(view, slices...);
    }

    template<typename ViewType>
    static auto& get_subregion(ViewType& sv) { return sv.get_layout(); }

};

struct UseSubViewWithLayout {

    template <typename ViewType, typename... Slices>
    auto operator()(ViewType& view, Slices... slices) const {
        return make_subview_with_layout(view, slices...);
    }

    template<typename ViewType>
    static auto& get_subregion(ViewType& sv) { return sv; }
};

template <typename Factory>
class SubViewTest : public ::testing::Test {};

using FactoryTypes = ::testing::Types<UseViewWithSubLayout, UseSubViewWithLayout>;
TYPED_TEST_SUITE(SubViewTest, FactoryTypes);

TYPED_TEST(SubViewTest, RangeSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[1:3]
    auto sv = TypeParam{}(view, RangeSlice<>{1,3});

    EXPECT_EQ(sv(0), 2);
    EXPECT_EQ(sv(1), 3);
    EXPECT_EQ(sv(2), 4);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 3);

}

TYPED_TEST(SubViewTest, RangeStartSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[2:]
    auto sv = TypeParam{}(view, RangeStartSlice<>{2});

    EXPECT_EQ(sv(0), 3);
    EXPECT_EQ(sv(1), 4);
    EXPECT_EQ(sv(2), 5);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 3);
}

TYPED_TEST(SubViewTest, StridedSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[0:3:2]
    auto sv = make_view_with_sublayout(view, StridedSlice<>{0,3,2});

    // sv = View[3:0:2]
    auto sv_neg_stride = make_view_with_sublayout(view, StridedSlice<>{3,0,-2});

    EXPECT_EQ(sv(0), 1);
    EXPECT_EQ(sv(1), 3);

    EXPECT_EQ(sv_neg_stride(0), 4);
    EXPECT_EQ(sv_neg_stride(1), 2);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 2);

    auto& sr_neg_stride = TypeParam::get_subregion(sv_neg_stride);
    EXPECT_EQ(sr_neg_stride.size(), 2);

}

TYPED_TEST(SubViewTest, FixedSubView1D)
{

    Index_type a[] = {1,2,3,4,5};

    View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

    // sv = View[1]
    auto sv = TypeParam{}(view, FixedSlice<>{1});

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 1);
    EXPECT_EQ(sr.size_noproj(), 0);
}

TYPED_TEST(SubViewTest, RangeSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,1:2]
    auto sv = TypeParam{}(view, RangeSlice<>{1,2}, RangeSlice<>{1,2});

    EXPECT_EQ(sv(0,0), 5);
    EXPECT_EQ(sv(0,1), 6);
    EXPECT_EQ(sv(1,0), 8);
    EXPECT_EQ(sv(1,1), 9);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 4);
    EXPECT_EQ(sr.template get_dim_size<0>(), 2);
    EXPECT_EQ(sr.template get_dim_size<1>(), 2);
    EXPECT_EQ(sr.template get_dim_stride<0>(), 1);
    EXPECT_EQ(sr.template get_dim_stride<1>(), 1);

}

TYPED_TEST(SubViewTest, RangeFixedSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,1]
    auto sv = TypeParam{}(view, RangeSlice<>{1,2}, FixedSlice<>{1});

    EXPECT_EQ(sv(0), 5);
    EXPECT_EQ(sv(1), 8);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 2);
    EXPECT_EQ(sr.template get_dim_size<0>(), 2);
    EXPECT_EQ(sr.template get_dim_stride<0>(), 1);

}

TYPED_TEST(SubViewTest, FixedFirstDimSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1,:]
    auto sv = TypeParam{}(view, FixedSlice<>{1}, NoSlice{});

    EXPECT_EQ(sv(0), 4);
    EXPECT_EQ(sv(1), 5);
    EXPECT_EQ(sv(2), 6);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 3);
    EXPECT_EQ(sr.template get_dim_size<0>(), 3);
    EXPECT_EQ(sr.template get_dim_stride<0>(), 1);
}

TYPED_TEST(SubViewTest, RangeFirstDimSubView2D)
{

    Index_type a[3][3] = {{1,2,3},
                          {4,5,6},
                          {7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,:]
    auto sv = TypeParam{}(view, RangeSlice<>{1,2}, NoSlice{});

    EXPECT_EQ(sv(0,0), 4);
    EXPECT_EQ(sv(0,1), 5);
    EXPECT_EQ(sv(0,2), 6);

    EXPECT_EQ(sv(1,0), 7);
    EXPECT_EQ(sv(1,1), 8);
    EXPECT_EQ(sv(1,2), 9);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 6);
    EXPECT_EQ(sr.template get_dim_size<0>(), 2);
    EXPECT_EQ(sr.template get_dim_size<1>(), 3);
    EXPECT_EQ(sr.template get_dim_stride<0>(), 1);
    EXPECT_EQ(sr.template get_dim_stride<1>(), 1);

}

TYPED_TEST(SubViewTest, RangeFirstDimStridedSecondDimSubView2D)
{

    Index_type a[3][6] = {{1,2,3,4,5,6},
                          {7,8,9,10,11,12},
                          {13,14,15,16,17,18}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,6));

    // sv = View[1:2,1:5:2]
    auto sv = TypeParam{}(view, RangeSlice<>{1,2}, StridedSlice<>{1,5,2});

    EXPECT_EQ(sv(0,0), 8);
    EXPECT_EQ(sv(0,1), 10);
    EXPECT_EQ(sv(0,2), 12);

    EXPECT_EQ(sv(1,0), 14);
    EXPECT_EQ(sv(1,1), 16);
    EXPECT_EQ(sv(1,2), 18);

    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 6);
    EXPECT_EQ(sr.template get_dim_size<0>(), 2);
    EXPECT_EQ(sr.template get_dim_size<1>(), 3);
    EXPECT_EQ(sr.template get_dim_stride<0>(), 1);
    EXPECT_EQ(sr.template get_dim_stride<1>(), 2);

}

TYPED_TEST(SubViewTest, SubViewOfSubView2D)
{

    Index_type a[3][3] = {{1,2,3},{4,5,6},{7,8,9}};

    View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3,3));

    // sv = View[1:2,:]
    auto sv = TypeParam{}(view, RangeSlice<>{1,2}, NoSlice{});


    auto& sr = TypeParam::get_subregion(sv);
    EXPECT_EQ(sr.size(), 6);

    // sv2 = sv[0:1,1:2]
    auto sv2 = TypeParam{}(sv, RangeSlice<>{0,1}, RangeSlice<>{1,2});

    EXPECT_EQ(sv2(0,0), 5);
    EXPECT_EQ(sv2(0,1), 6);

    EXPECT_EQ(sv2(1,0), 8);
    EXPECT_EQ(sv2(1,1), 9);

    EXPECT_EQ(TypeParam::get_subregion(sv2).size(), 4);

    // sv3 = sv2[:,1]
    auto sv3 =TypeParam{}(sv2, NoSlice{}, FixedSlice<>{1});

    EXPECT_EQ(sv3(0), 6);
    EXPECT_EQ(sv3(1), 9);

    EXPECT_EQ(TypeParam::get_subregion(sv3).size(), 2);

}

TEST(SubViewMultiViewTest, SubViewOfMultiView2D)
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
    auto sv = make_multiview_with_sublayout(view, RangeSlice<>{1,2});

    EXPECT_EQ(sv.get_layout().size(), 2);
    EXPECT_EQ(sv.get_layout().get_dim_size<0>(), 2);

    EXPECT_EQ(sv(0,0), 1);
    EXPECT_EQ(sv(0,1), 4);

    EXPECT_EQ(sv(1,0), 1);
    EXPECT_EQ(sv(1,1), 8);

    // sv2 = MultiView[1,1:2]
    auto sv2 = make_subview_with_layout(view, FixedSlice<>{1}, RangeSlice<>{1,2});

    // the parent layout is a MultiView
    EXPECT_EQ(sv2.get_parent().get_layout().size(), 4);
    EXPECT_EQ(sv2.size(), 2);

    EXPECT_EQ(sv2(0), 1);
    EXPECT_EQ(sv2(1), 8);

    // sv3 = MultiView[:,2]
    auto sv3 = make_multiview_with_sublayout(view, FixedSlice<>{2});

    // this size corresponds to the sliced sublayout (0D)
    // which is sliced from the original MultiView's 1D layout
    EXPECT_EQ(sv3.get_layout().size(), 1);

    EXPECT_EQ(sv3(0), 4);
    EXPECT_EQ(sv3(1), 8);

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

// TYPED_TEST(SubViewTest, RangeFirstDimSubView2DGPU)
// {
//     test_subviewGPU();
// }
