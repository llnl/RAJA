//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <gtest/gtest.h>
#include <array>
#include <type_traits>
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/SubView.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA_test-base.hpp"

using namespace RAJA;

template<typename ViewType, typename... Slices>
RAJA_HOST_DEVICE auto make_multiview_with_sublayout(ViewType& view,
                                                    Slices... slices)
{
  using SubLayoutType =
      SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
  return MultiView<Index_type, SubLayoutType>(
      view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

struct UseMakeSubview
{

  template<typename ViewType, typename... Slices>
  auto operator()(ViewType& view, Slices... slices) const
  {
    return make_subview(view, slices...);
  }

  template<typename ViewType>
  static auto& get_subregion(ViewType& sv)
  {
    return sv.get_layout();
  }
};

struct UseSlicingAdapterOverView
{

  template<typename ViewType, typename... Slices>
  auto operator()(ViewType& view, Slices... slices) const
  {
    return SlicingAdapter(view, slices...);
  }

  template<typename ViewType>
  static auto& get_subregion(ViewType& sv)
  {
    return sv;
  }
};

template<typename Factory>
class SubViewTest : public ::testing::Test
{};

using FactoryTypes =
    ::testing::Types<UseMakeSubview, UseSlicingAdapterOverView>;
TYPED_TEST_SUITE(SubViewTest, FactoryTypes);

TEST(SlicingAdapterTest, GetLayoutParent)
{
  Index_type data[12] {};
  Layout<2> layout(3, 4);
  View<Index_type, Layout<2>> view(data, layout);

  auto subview =
      make_subview(view, RangeSlice<> {1, 3}, StridedSlice<> {0, 4, 2});
  auto const& parent = subview.get_layout().get_parent();

  static_assert(
      std::is_same<decltype(parent), Layout<2> const&>::value,
      "get_parent must return a const reference to the parent layout");
  EXPECT_EQ(parent.template get_dim_size<0>(),
            layout.template get_dim_size<0>());
  EXPECT_EQ(parent.template get_dim_size<1>(),
            layout.template get_dim_size<1>());
  EXPECT_EQ(parent.template get_dim_stride<0>(),
            layout.template get_dim_stride<0>());
  EXPECT_EQ(parent.template get_dim_stride<1>(),
            layout.template get_dim_stride<1>());
}

TEST(SlicingAdapterTest, GetViewParent)
{
  Index_type data[12] {};
  Layout<2> layout(3, 4);
  View<Index_type, Layout<2>> view(data, layout);

  auto adapter =
      SlicingAdapter(view, RangeSlice<> {1, 3}, StridedSlice<> {0, 4, 2});
  auto const& parent = adapter.get_parent();

  static_assert(std::is_same<decltype(parent), decltype(view) const&>::value,
                "get_parent must return a const reference to the parent view");
  EXPECT_EQ(parent.get_data(), view.get_data());
  EXPECT_EQ(parent.get_layout().template get_dim_size<0>(),
            layout.template get_dim_size<0>());
  EXPECT_EQ(parent.get_layout().template get_dim_size<1>(),
            layout.template get_dim_size<1>());
  EXPECT_EQ(parent.get_layout().template get_dim_stride<0>(),
            layout.template get_dim_stride<0>());
  EXPECT_EQ(parent.get_layout().template get_dim_stride<1>(),
            layout.template get_dim_stride<1>());
}

TYPED_TEST(SubViewTest, RangeSubView1D)
{

  Index_type a[] = {1, 2, 3, 4, 5};

  View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

  // sv = View[1:4]
  auto sv = TypeParam {}(view, RangeSlice<> {1, 4});

  EXPECT_EQ(sv(0), 2);
  EXPECT_EQ(sv(1), 3);
  EXPECT_EQ(sv(2), 4);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 3);
}

TYPED_TEST(SubViewTest, WriteThrough1D)
{
  Index_type a[] = {1, 2, 3, 4, 5};
  View<Index_type, Layout<1>> view(a, Layout<1>(5));

  auto sv = TypeParam {}(view, RangeSlice<> {1, 4});
  sv(1)   = 30;

  EXPECT_EQ(view(2), 30);
}

TYPED_TEST(SubViewTest, RangeStartSubView1D)
{

  Index_type a[] = {1, 2, 3, 4, 5};

  View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

  // sv = View[2:]
  auto sv = TypeParam {}(view, RangeStartSlice<> {2});

  EXPECT_EQ(sv(0), 3);
  EXPECT_EQ(sv(1), 4);
  EXPECT_EQ(sv(2), 5);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 3);
}

TYPED_TEST(SubViewTest, OffsetLayoutSubView1D)
{
  Index_type a[] = {1, 2, 3};
  auto layout    = make_offset_layout<1>(std::array<Index_type, 1> {{-1}},
                                         std::array<Index_type, 1> {{2}});
  View<Index_type, OffsetLayout<1>> view(a, layout);

  auto full_view = TypeParam {}(view, NoSlice {});
  EXPECT_EQ(full_view.size(), 3);
  EXPECT_EQ(full_view.template get_dim_begin<0>(), 0);
  EXPECT_EQ(full_view(0), 1);
  EXPECT_EQ(full_view(1), 2);
  EXPECT_EQ(full_view(2), 3);

  auto tail_view = TypeParam {}(view, RangeStartSlice<> {0});
  EXPECT_EQ(tail_view.size(), 2);
  EXPECT_EQ(tail_view(0), 2);
  EXPECT_EQ(tail_view(1), 3);

  auto range_view = TypeParam {}(view, RangeSlice<> {-1, 1});
  EXPECT_EQ(range_view.size(), 2);
  EXPECT_EQ(range_view(0), 1);
  EXPECT_EQ(range_view(1), 2);
}

TYPED_TEST(SubViewTest, IndexLayoutSubView1D)
{
  Index_type a[]       = {1, 2, 3};
  Index_type indices[] = {2, 0, 1};
  auto index_tuple     = make_index_tuple(IndexList<> {indices});
  auto layout          = make_index_layout(index_tuple, 3);
  auto view            = make_index_view(a, layout);

  EXPECT_EQ(view.template get_dim_begin<0>(), 0);

  auto full_view = TypeParam {}(view, NoSlice {});
  EXPECT_EQ(full_view.size(), 3);
  EXPECT_EQ(full_view(0), 3);
  EXPECT_EQ(full_view(1), 1);
  EXPECT_EQ(full_view(2), 2);
}

TYPED_TEST(SubViewTest, StridedSubView1D)
{

  Index_type a[] = {1, 2, 3, 4, 5};

  View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

  // sv = View[0:4:2]
  auto sv = TypeParam {}(view, StridedSlice<> {0, 4, 2});

  // sv_neg_stride = View[4:0:2]
  auto sv_neg_stride = TypeParam {}(view, StridedSlice<> {4, 0, -2});

  // sv_odd_stride = View[0:4:3]
  auto sv_odd_stride = TypeParam {}(view, StridedSlice<> {0, 4, 3});

  EXPECT_EQ(sv(0), 1);
  EXPECT_EQ(sv(1), 3);

  EXPECT_EQ(sv_neg_stride(0), 5);
  EXPECT_EQ(sv_neg_stride(1), 3);

  EXPECT_EQ(sv_odd_stride(0), 1);
  EXPECT_EQ(sv_odd_stride(1), 4);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 2);

  auto& sr_neg_stride = TypeParam::get_subregion(sv_neg_stride);
  EXPECT_EQ(sr_neg_stride.size(), 2);

  auto& sr_odd_stride = TypeParam::get_subregion(sv_odd_stride);
  EXPECT_EQ(sr_odd_stride.size(), 2);
}

TYPED_TEST(SubViewTest, EmptySlices1D)
{
  Index_type a[] = {1, 2, 3, 4, 5};
  View<Index_type, Layout<1>> view(a, Layout<1>(5));

  auto range          = TypeParam {}(view, RangeSlice<> {2, 2});
  auto& range_adapter = TypeParam::get_subregion(range);
  EXPECT_EQ(range_adapter.template get_dim_size<0>(), 0);
  EXPECT_EQ(range_adapter.size(), 1);
  EXPECT_EQ(range_adapter.size_noproj(), 0);

  auto positive_stride          = TypeParam {}(view, StridedSlice<> {3, 1, 1});
  auto& positive_stride_adapter = TypeParam::get_subregion(positive_stride);
  EXPECT_EQ(positive_stride_adapter.template get_dim_size<0>(), 0);
  EXPECT_EQ(positive_stride_adapter.size(), 1);
  EXPECT_EQ(positive_stride_adapter.size_noproj(), 0);

  auto negative_stride          = TypeParam {}(view, StridedSlice<> {1, 3, -1});
  auto& negative_stride_adapter = TypeParam::get_subregion(negative_stride);
  EXPECT_EQ(negative_stride_adapter.template get_dim_size<0>(), 0);
  EXPECT_EQ(negative_stride_adapter.size(), 1);
  EXPECT_EQ(negative_stride_adapter.size_noproj(), 0);
}

TYPED_TEST(SubViewTest, FixedSubView1D)
{

  Index_type a[] = {1, 2, 3, 4, 5};

  View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

  // sv = View[1]
  auto sv = TypeParam {}(view, FixedSlice<> {1});

  EXPECT_EQ(sv(), 2);
  sv() = 20;
  EXPECT_EQ(a[1], 20);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 1);
  EXPECT_EQ(sr.size_noproj(), 1);
}

TYPED_TEST(SubViewTest, RangeSubView2D)
{

  Index_type a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3, 3));

  // sv = View[1:3,1:3]
  auto sv = TypeParam {}(view, RangeSlice<> {1, 3}, RangeSlice<> {1, 3});

  EXPECT_EQ(sv(0, 0), 5);
  EXPECT_EQ(sv(0, 1), 6);
  EXPECT_EQ(sv(1, 0), 8);
  EXPECT_EQ(sv(1, 1), 9);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 4);
  EXPECT_EQ(sr.template get_dim_size<0>(), 2);
  EXPECT_EQ(sr.template get_dim_size<1>(), 2);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 3);
  EXPECT_EQ(sr.template get_dim_stride<1>(), 1);
}

TYPED_TEST(SubViewTest, ProjectedLayoutSizeDiff2D)
{

  Index_type a[3] = {1, 2, 3};

  // Projection in the second dimension (size 0)
  View<Index_type, Layout<2>> view(&a[0], Layout<2>(3, 0));

  // sv = View[:, :]
  auto sv = TypeParam {}(view, NoSlice {}, NoSlice {});

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 3);
  EXPECT_EQ(sr.size_noproj(), 0);
}

TYPED_TEST(SubViewTest, RangeFixedSubView2D)
{

  Index_type a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3, 3));

  // sv = View[1:3,1]
  auto sv = TypeParam {}(view, RangeSlice<> {1, 3}, FixedSlice<> {1});

  EXPECT_EQ(sv(0), 5);
  EXPECT_EQ(sv(1), 8);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 2);
  EXPECT_EQ(sr.template get_dim_size<0>(), 2);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 3);
}

TYPED_TEST(SubViewTest, FixedFirstDimSubView2D)
{

  Index_type a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3, 3));

  // sv = View[1,:]
  auto sv = TypeParam {}(view, FixedSlice<> {1}, NoSlice {});

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

  Index_type a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3, 3));

  // sv = View[1:3,:]
  auto sv = TypeParam {}(view, RangeSlice<> {1, 3}, NoSlice {});

  EXPECT_EQ(sv(0, 0), 4);
  EXPECT_EQ(sv(0, 1), 5);
  EXPECT_EQ(sv(0, 2), 6);

  EXPECT_EQ(sv(1, 0), 7);
  EXPECT_EQ(sv(1, 1), 8);
  EXPECT_EQ(sv(1, 2), 9);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 6);
  EXPECT_EQ(sr.template get_dim_size<0>(), 2);
  EXPECT_EQ(sr.template get_dim_size<1>(), 3);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 3);
  EXPECT_EQ(sr.template get_dim_stride<1>(), 1);
}

TYPED_TEST(SubViewTest, RangeFirstDimStridedSecondDimSubView2D)
{

  Index_type a[3][6] = {
      {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18}};

  View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3, 6));

  // sv = View[1:3,1:6:2]
  auto sv = TypeParam {}(view, RangeSlice<> {1, 3}, StridedSlice<> {1, 6, 2});

  EXPECT_EQ(sv(0, 0), 8);
  EXPECT_EQ(sv(0, 1), 10);
  EXPECT_EQ(sv(0, 2), 12);

  EXPECT_EQ(sv(1, 0), 14);
  EXPECT_EQ(sv(1, 1), 16);
  EXPECT_EQ(sv(1, 2), 18);

  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 6);
  EXPECT_EQ(sr.template get_dim_size<0>(), 2);
  EXPECT_EQ(sr.template get_dim_size<1>(), 3);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 6);
  EXPECT_EQ(sr.template get_dim_stride<1>(), 2);
}

TYPED_TEST(SubViewTest, SubViewOfSubView2D)
{

  Index_type a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  View<Index_type, Layout<2>> view(&a[0][0], Layout<2>(3, 3));

  // sv = View[1:3,:]
  auto sv = TypeParam {}(view, RangeSlice<> {1, 3}, NoSlice {});


  auto& sr = TypeParam::get_subregion(sv);
  EXPECT_EQ(sr.size(), 6);

  // sv2 = sv[0:2,1:3]
  auto sv2 = TypeParam {}(sv, RangeSlice<> {0, 2}, RangeSlice<> {1, 3});

  EXPECT_EQ(sv2(0, 0), 5);
  EXPECT_EQ(sv2(0, 1), 6);

  EXPECT_EQ(sv2(1, 0), 8);
  EXPECT_EQ(sv2(1, 1), 9);

  EXPECT_EQ(TypeParam::get_subregion(sv2).size(), 4);

  // sv3 = sv2[:,1]
  auto sv3 = TypeParam {}(sv2, NoSlice {}, FixedSlice<> {1});

  EXPECT_EQ(sv3(0), 6);
  EXPECT_EQ(sv3(1), 9);

  EXPECT_EQ(TypeParam::get_subregion(sv3).size(), 2);
}

TEST(SubLayoutMultiViewTest, MultiViewWithSubLayout2D)
{

  Index_type data_squared[4];
  Index_type data_cubed[4];

  for (int i = 0; i < 4; i++)
  {
    data_squared[i] = i * i;
  }

  for (int i = 0; i < 4; i++)
  {
    data_cubed[i] = i * i * i;
  }

  Index_type* data_array[2];
  data_array[0] = data_squared;
  data_array[1] = data_cubed;

  Index_type index_list[4] = {3, 1, 2, 0};

  auto index_tuple  = make_index_tuple(IndexList<> {&index_list[0]});
  auto index_layout = make_index_layout(index_tuple, 4);

  auto view = MultiView<Index_type, IndexLayout<1, Index_type, IndexList<>>>(
      data_array, index_layout);

  // sv = MultiView[:,1:3]
  auto sv = make_multiview_with_sublayout(view, RangeSlice<> {1, 3});

  EXPECT_EQ(sv.get_layout().size(), 2);
  EXPECT_EQ(sv.get_layout().get_dim_size<0>(), 2);

  EXPECT_EQ(sv(0, 0), 1);
  EXPECT_EQ(sv(0, 1), 4);

  EXPECT_EQ(sv(1, 0), 1);
  EXPECT_EQ(sv(1, 1), 8);

  // fixed_view = MultiView[:,2]
  auto fixed_view = make_multiview_with_sublayout(view, FixedSlice<> {2});

  // this size corresponds to the sliced sublayout (0D)
  // which is sliced from the original MultiView's 1D layout
  EXPECT_EQ(fixed_view.get_layout().size(), 1);

  EXPECT_EQ(fixed_view(0), 4);
  EXPECT_EQ(fixed_view(1), 8);
}
