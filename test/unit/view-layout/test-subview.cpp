//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <gtest/gtest.h>
#include <array>
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/SubView.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-forone.hpp"

using namespace RAJA;

/* helper to create a RAJA View with a sliced SubLayout */
template<typename ViewType, typename... Slices>
RAJA_HOST_DEVICE auto make_view_with_sublayout(ViewType& view, Slices... slices)
{
  using SubLayoutType =
      SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
  return View<Index_type, SubLayoutType>(
      view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

/* helper to create a sliced SubView without modifying underlying layout */
template<typename ViewType, typename... Slices>
RAJA_HOST_DEVICE auto make_subview_with_layout(ViewType& view, Slices... slices)
{
  using SubViewType = SubView<ViewType, camp::list<Slices...>>;
  return SubViewType(view, slices...);
}

template<typename ViewType, typename... Slices>
RAJA_HOST_DEVICE auto make_multiview_with_sublayout(ViewType& view,
                                                    Slices... slices)
{
  using SubLayoutType =
      SubLayout<typename ViewType::layout_type, camp::list<Slices...>>;
  return MultiView<Index_type, SubLayoutType>(
      view.get_data(), SubLayoutType(view.get_layout(), slices...));
}

struct UseViewWithSubLayout
{

  template<typename ViewType, typename... Slices>
  auto operator()(ViewType& view, Slices... slices) const
  {
    return make_view_with_sublayout(view, slices...);
  }

  template<typename ViewType>
  static auto& get_subregion(ViewType& sv)
  {
    return sv.get_layout();
  }
};

struct UseSubViewWithLayout
{

  template<typename ViewType, typename... Slices>
  auto operator()(ViewType& view, Slices... slices) const
  {
    return make_subview_with_layout(view, slices...);
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
    ::testing::Types<UseViewWithSubLayout, UseSubViewWithLayout>;
TYPED_TEST_SUITE(SubViewTest, FactoryTypes);

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

TYPED_TEST(SubViewTest, StridedSubView1D)
{

  Index_type a[] = {1, 2, 3, 4, 5};

  View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

  // sv = View[0:4:2]
  auto sv = make_view_with_sublayout(view, StridedSlice<> {0, 4, 2});

  // sv_neg_stride = View[4:0:2]
  auto sv_neg_stride =
      make_view_with_sublayout(view, StridedSlice<> {4, 0, -2});

  // sv_odd_stride = View[0:4:3]
  auto sv_odd_stride = make_view_with_sublayout(view, StridedSlice<> {0, 4, 3});

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

TYPED_TEST(SubViewTest, FixedSubView1D)
{

  Index_type a[] = {1, 2, 3, 4, 5};

  View<Index_type, Layout<1>> view(&a[0], Layout<1>(5));

  // sv = View[1]
  auto sv = TypeParam {}(view, FixedSlice<> {1});

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
  EXPECT_EQ(sr.template get_parent_dim_stride<0>(), 3);
  EXPECT_EQ(sr.template get_parent_dim_stride<1>(), 1);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 2);
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
  EXPECT_EQ(sr.template get_parent_dim_stride<0>(), 3);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 1);
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
  EXPECT_EQ(sr.template get_parent_dim_stride<0>(), 1);
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
  EXPECT_EQ(sr.template get_parent_dim_stride<0>(), 3);
  EXPECT_EQ(sr.template get_parent_dim_stride<1>(), 1);
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
  EXPECT_EQ(sr.template get_parent_dim_stride<0>(), 6);
  EXPECT_EQ(sr.template get_parent_dim_stride<1>(), 2);
  EXPECT_EQ(sr.template get_dim_stride<0>(), 3);
  EXPECT_EQ(sr.template get_dim_stride<1>(), 1);
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

TEST(SubViewMultiViewTest, SubViewOfMultiView2D)
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

  // sv2 = MultiView[1,1:3]
  auto sv2 =
      make_subview_with_layout(view, FixedSlice<> {1}, RangeSlice<> {1, 3});

  // the parent layout is a MultiView
  EXPECT_EQ(sv2.get_parent().get_layout().size(), 4);
  EXPECT_EQ(sv2.size(), 2);

  EXPECT_EQ(sv2(0), 1);
  EXPECT_EQ(sv2(1), 8);

  // sv3 = MultiView[:,2]
  auto sv3 = make_multiview_with_sublayout(view, FixedSlice<> {2});

  // this size corresponds to the sliced sublayout (0D)
  // which is sliced from the original MultiView's 1D layout
  EXPECT_EQ(sv3.get_layout().size(), 1);

  EXPECT_EQ(sv3(0), 4);
  EXPECT_EQ(sv3(1), 8);
}

#if defined(RAJA_ENABLE_HIP)
GPU_TEST(SubViewGPUTest, SubView2D_HIP)
{
  constexpr Index_type rows = 3;
  constexpr Index_type cols = 6;
  constexpr Index_type N    = rows * cols;

  std::array<Index_type, static_cast<size_t>(N)> host_data {};
  for (Index_type r = 0; r < rows; ++r)
  {
    for (Index_type c = 0; c < cols; ++c)
    {
      host_data[static_cast<size_t>(r * cols + c)] =
          Index_type(1) + r * cols + c;
    }
  }

  Index_type* data = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &data, sizeof(Index_type) * N);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, data, host_data.data(),
                                sizeof(Index_type) * N, hipMemcpyHostToDevice);

  std::array<Index_type, 16> host_out {};
  Index_type* out = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMalloc, &out, sizeof(Index_type) * 16);
  for (size_t i = 0; i < host_out.size(); ++i)
  {
    host_out[i] = Index_type(-1);
  }
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, out, host_out.data(),
                                sizeof(Index_type) * host_out.size(),
                                hipMemcpyHostToDevice);

  View<Index_type, Layout<2>> view(data, Layout<2>(rows, cols));

  forone<test_hip>([=] RAJA_DEVICE() {
    // sv = View[1:3,1:6:2]
    auto sv_with_sublayout = make_view_with_sublayout(view, RangeSlice<> {1, 3},
                                                      StridedSlice<> {1, 6, 2});
    auto sv_with_layout    = make_subview_with_layout(view, RangeSlice<> {1, 3},
                                                      StridedSlice<> {1, 6, 2});

    out[0] = sv_with_sublayout(0, 0);
    out[1] = sv_with_sublayout(0, 1);
    out[2] = sv_with_sublayout(0, 2);
    out[3] = sv_with_sublayout(1, 0);
    out[4] = sv_with_sublayout(1, 1);
    out[5] = sv_with_sublayout(1, 2);

    auto const& sr1 = sv_with_sublayout.get_layout();
    out[6]          = sr1.template get_parent_dim_stride<0>();
    out[7]          = sr1.template get_parent_dim_stride<1>();
    out[8]          = sr1.template get_dim_stride<0>();
    out[9]          = sr1.template get_dim_stride<1>();

    out[10] = sv_with_layout(0, 0);
    out[11] = sv_with_layout(0, 1);
    out[12] = sv_with_layout(0, 2);
    out[13] = sv_with_layout(1, 0);
    out[14] = sv_with_layout(1, 1);
    out[15] = sv_with_layout(1, 2);
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipMemcpy, host_out.data(), out,
                                sizeof(Index_type) * host_out.size(),
                                hipMemcpyDeviceToHost);

  EXPECT_EQ(host_out[0], 8);
  EXPECT_EQ(host_out[1], 10);
  EXPECT_EQ(host_out[2], 12);
  EXPECT_EQ(host_out[3], 14);
  EXPECT_EQ(host_out[4], 16);
  EXPECT_EQ(host_out[5], 18);
  EXPECT_EQ(host_out[6], 6);
  EXPECT_EQ(host_out[7], 2);
  EXPECT_EQ(host_out[8], 3);
  EXPECT_EQ(host_out[9], 1);
  EXPECT_EQ(host_out[10], 8);
  EXPECT_EQ(host_out[11], 10);
  EXPECT_EQ(host_out[12], 12);
  EXPECT_EQ(host_out[13], 14);
  EXPECT_EQ(host_out[14], 16);
  EXPECT_EQ(host_out[15], 18);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, out);
  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, data);
}
#endif
