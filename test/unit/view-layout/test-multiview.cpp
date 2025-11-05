//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

RAJA_INDEX_VALUE(TIX, "TIX");
RAJA_INDEX_VALUE(TIL, "TIL");

template<typename T>
class MultiViewUnitTest : public ::testing::Test {};

template<typename T>
class OffsetLayoutMultiViewUnitTest : public ::testing::Test {};

template<typename T>
class TypedIntegralMultiViewUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(MultiViewUnitTest, UnitIntFloatTypes);
TYPED_TEST_SUITE(OffsetLayoutMultiViewUnitTest, UnitIntFloatTypes);
TYPED_TEST_SUITE(TypedIntegralMultiViewUnitTest, UnitIntFloatTypes);

TYPED_TEST(MultiViewUnitTest, Constructors)
{

  using layout = RAJA::Layout<1>;

  TypeParam   a1[10];
  TypeParam   a2[10];
  TypeParam * data[2];

  data[0] = a1;
  data[1] = a2;

  constexpr int val = 8;
  a1[0] = val;
  a2[0] = val;

  TypeParam   b1[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  TypeParam   b2[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  TypeParam const * data2[2];

  data2[0] = b1;
  data2[1] = b2;

  b1[0] = val + 1;
  b2[0] = val + 1;

  RAJA::MultiView<TypeParam, layout> view(data, layout(10));
  ASSERT_EQ( val, view(0,0) );

  /*
   * Should be able to construct a const MultiView from const array of arrays
   */
  RAJA::MultiView<TypeParam const, layout> view_from_const(data2, layout(10));
  ASSERT_EQ( val + 1, view_from_const(0,0) );

  /*
   * TODO: Should be able to construct a const MultiView from non-const array of arrays. For now, this becomes an ambiguous call to constructor error.
   */
  //RAJA::MultiView<TypeParam const, layout> constview_from_nonconst(data, layout(10));
  //ASSERT_EQ( val, constview_from_nonconst(0,0) );

  /*
   * Should be able to construct an empty const MultiView
   */
  RAJA::MultiView<TypeParam const, layout> view_from_const2;
  ASSERT_EQ( nullptr, view_from_const2.get_data() );
  ASSERT_EQ( 1, (view_from_const2.get_layout()).size() );

  /*
   * Should be able to set data and layout in an empty const MultiView
   */
  view_from_const2.set_layout(layout(10));
  view_from_const2.set_data(data2);
  ASSERT_EQ( 10, (view_from_const2.get_layout()).size() );
  ASSERT_EQ( val + 1, view_from_const2(0,0) );

  /*
   * Should be able to set non-const data in a const MultiView
   */
  view_from_const2.set_data(data);
  ASSERT_EQ( val, view_from_const2(0,0) );

  /*
   * Should be able to construct an empty MultiView
   */
  RAJA::MultiView<TypeParam, layout> view_empty;
  ASSERT_EQ( nullptr, view_empty.get_data() );
  ASSERT_EQ( 1, (view_empty.get_layout()).size() );

  /*
   * Should be able to set data and layout in an empty MultiView
   */
  view_empty.set_layout(layout(10));
  view_empty.set_data(data);
  ASSERT_EQ( 10, (view_empty.get_layout()).size() );
  ASSERT_EQ( val, view_empty(0,0) );

  /*
   * Should be able to construct a non-const MultiView from a non-const MultiView
   */
  RAJA::MultiView<TypeParam, layout> view2(view);
  ASSERT_EQ( val, view2(0,0) );

  /*
   * Should be able to construct a const MultiView from a non-const MultiView
   */
  RAJA::MultiView<TypeParam const, layout> const_view(view);
  ASSERT_EQ( val, const_view(0,0) );

  /*
   * Should be able to construct a const MultiView from a const MultiView
   */
  RAJA::MultiView<TypeParam const, layout> const_view2(const_view);
  ASSERT_EQ( val, const_view2(0,0) );


  // non-default construction of MultiView with array-of-pointers index moved to 1st position
  RAJA::MultiView<TypeParam, layout, 1> view1p(data, layout(10));
  ASSERT_EQ( val, view1p(0,0) );

  // construct a non-const MultiView from a non-const MultiView
  RAJA::MultiView<TypeParam, layout, 1> view1p2(view1p);
  ASSERT_EQ( val, view1p2(0,0) );

  // construct a const MultiView from a non-const MultiView
  RAJA::MultiView<TypeParam const, layout, 1> const_view1p(view1p);
  ASSERT_EQ( val, const_view1p(0,0) );

  // construct a const MultiView from a const MultiView
  RAJA::MultiView<TypeParam const, layout, 1> const_view1p2(const_view1p);
  ASSERT_EQ( val, const_view1p2(0,0) );


  // non-default construction of MultiView with array-of-pointers index moved to 1st position
  // and non-const pointer type specification (used in CHAI)
  RAJA::MultiView<TypeParam, layout, 1, TypeParam **> view1pnc(data, layout(10));
  ASSERT_EQ( val, view1pnc(0,0) );

  // construct a non-const MultiView from a non-const MultiView
  RAJA::MultiView<TypeParam, layout, 1, TypeParam **> view1pnc2(view1pnc);
  ASSERT_EQ( val, view1pnc2(0,0) );

  // construct a const MultiView from a non-const MultiView
  RAJA::MultiView<TypeParam const, layout, 1, TypeParam **> const_view1pnc(view1pnc);
  ASSERT_EQ( val, const_view1pnc(0,0) );

  // construct a const MultiView from a const MultiView
  RAJA::MultiView<TypeParam const, layout, 1, TypeParam **> const_view1pnc2(const_view1pnc);
  ASSERT_EQ( val, const_view1pnc2(0,0) );
}

TYPED_TEST(MultiViewUnitTest, Accessor)
{

  const int Nx = 3;
  const int Ny = 5;
  const int Nz = 2;
  const int N  = Nx*Ny*Nz;
  TypeParam *b = new TypeParam[N];
  TypeParam *c = new TypeParam[N];
  TypeParam *a[2];

  a[0] = b;
  a[1] = c;

  int iter{0};
  for(TypeParam i=0; i<TypeParam{N}; ++i)
  {
    a[0][iter] = TypeParam{i};
    a[1][iter] = TypeParam{i}+1;
    ++iter;
  }

  /*
   * 1D Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<1>> view_1D(a,N);
  RAJA::MultiView<TypeParam, RAJA::Layout<1>,1> view_1D1p(a,N);
  TypeParam val{0};
  for(int i=0; i<N; ++i) {
    ASSERT_EQ(val, view_1D(0,i));
    ASSERT_EQ(val+1, view_1D(1,i));
    ASSERT_EQ(val, view_1D1p(i,0));
    ASSERT_EQ(val+1, view_1D1p(i,1));
    val++;
  }

  /*
   * 2D Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<2>> view_2D(a,Ny,Nx);
  RAJA::MultiView<TypeParam, RAJA::Layout<2>,1> view_2D1p(a,Ny,Nx);
  val = TypeParam{0};
  for(int j=0; j<Ny; ++j) {
    for(int i=0; i<Nx; ++i) {
      ASSERT_EQ(val, view_2D(0,j,i));
      ASSERT_EQ(val+1, view_2D(1,j,i));
      ASSERT_EQ(val, view_2D1p(j,0,i));
      ASSERT_EQ(val+1, view_2D1p(j,1,i));
      val++;
    }
  }

  /*
   * 3D Accessor
   */
  RAJA::MultiView<TypeParam, RAJA::Layout<3>> view_3D(a,Nz,Ny,Nx);
  RAJA::MultiView<TypeParam, RAJA::Layout<3>,2> view_3D1p(a,Nz,Ny,Nx);
  val = TypeParam{0};
  for(int k=0; k<Nz; ++k) {
    for(int j=0; j<Ny; ++j) {
      for(int i=0; i<Nx; ++i) {
        ASSERT_EQ(val, view_3D(0,k,j,i));
        ASSERT_EQ(val+1, view_3D(1,k,j,i));
        ASSERT_EQ(val, view_3D1p(k,j,0,i));
        ASSERT_EQ(val+1, view_3D1p(k,j,1,i));
        val++;
      }
    }
  }

  delete[] b;
  delete[] c;
}

TYPED_TEST(OffsetLayoutMultiViewUnitTest, View)
{
  TypeParam* d1 = new TypeParam[10];
  TypeParam* d2 = new TypeParam[10];
  TypeParam* data[2];

  data[0] = d1;
  data[1] = d2;

  using layout = RAJA::OffsetLayout<>;

  /*
   * MultiView is constructed by passing in the layout.
   */
  std::array<RAJA::Index_type, 1> lower{{1}};
  std::array<RAJA::Index_type, 1> upper{{11}};
  RAJA::MultiView<TypeParam, layout> view(data, RAJA::make_offset_layout<1>(lower, upper));
  RAJA::MultiView<TypeParam, layout,1> view1p(data, RAJA::make_offset_layout<1>(lower, upper));

  for (int i = 0; i < 10; i++) {
    data[0][i] = static_cast<TypeParam>(i);
    data[1][i] = static_cast<TypeParam>(i+1);
  }

  ASSERT_EQ(data[0][0], view(0,1));
  ASSERT_EQ(data[1][9], view(1,10));
  ASSERT_EQ(data[0][0], view1p(1,0));
  ASSERT_EQ(data[1][9], view1p(10,1));

  delete[] d1;
  delete[] d2;
}

TYPED_TEST(MultiViewUnitTest, Shift1D)
{

  int N = 10;
  TypeParam *reala = new TypeParam[N];
  TypeParam *realb = new TypeParam[N];
  TypeParam *a[2];
  a[0] = reala;
  a[1] = realb;

  //Create a view from a base view
  const int DIM = 1;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0}},{{N}});
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::MultiView<TypeParam, RAJA::Layout<DIM>> B(a,N);

  for(int i=0; i<N; ++i) {
    A(0,i) = static_cast<TypeParam>(i + 1);
    B(1,i) = static_cast<TypeParam>(i + 1);
  }

  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N}});
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N}});

  for(int i=N; i<2*N; ++i)
  {
    ASSERT_EQ(Ashift(0,i),A(0,i-N));
    ASSERT_EQ(Bshift(1,i),B(1,i-N));
  }

  // offset layout with MultiView with array-of-pointers index in 1st position
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>, 1> C(a,layout);
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>, 1> Cshift = C.shift({{N}});

  for(int i=N; i<2*N; ++i)
  {
    ASSERT_EQ(Cshift(i,0),C(i-N,0));
    ASSERT_EQ(Cshift(i,1),C(i-N,1));
    ASSERT_EQ(Ashift(0,i),C(i-N,0));
    ASSERT_EQ(Cshift(i,0),A(0,i-N));
  }


  //Create a shifted view from a view with a typed layout
  using TLayout = RAJA::TypedLayout<TIL, RAJA::tuple<TIX>>;
  using TOffsetLayout = RAJA::TypedOffsetLayout<TIL, RAJA::tuple<TIX>>;

  TLayout myLayout(10);

  RAJA::MultiView<TypeParam, TLayout> D(a, myLayout);
  RAJA::MultiView<TypeParam, TOffsetLayout> Dshift = D.shift({{N}});

  for(TIX i=TIX{N}; i<TIX{2*N}; ++i)
  {
    ASSERT_EQ(Dshift(0,i),D(0,i-N));
  };

  delete[] reala;
  delete[] realb;
}

TYPED_TEST(MultiViewUnitTest, Shift2D)
{

  int N = 10;
  TypeParam *a0 = new TypeParam[N*N];
  TypeParam *b0 = new TypeParam[N*N];
  TypeParam *a[2];
  a[0] = a0;
  a[1] = b0;

  const int DIM = 2;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0,0}},{{N,N}});
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::MultiView<TypeParam, RAJA::Layout<DIM>> B(a,N,N);

  for(int y=0; y<N; ++y) {
    for(int x=0; x<N; ++x) {
      A(0,y,x) = static_cast<TypeParam>(x + N*y);
      B(1,y,x) = static_cast<TypeParam>(x + N*y + 1);
    }
  }

  //Create a view from a base view with an offsetlayout
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N,N}});
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N,N}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Ashift(0,y,x),A(0,y-N,x-N));
      ASSERT_EQ(Bshift(1,y,x),B(1,y-N,x-N));
    }
  }

  //Create a view from a base view with permuted layout
  std::array< RAJA::idx_t, 2> perm {{1, 0}};
  RAJA::OffsetLayout<2> playout =
    RAJA::make_permuted_offset_layout<2>( {{0, 0}}, {{N, N}}, perm );

  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> C(a, playout);
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>> Cshift = C.shift({{N,N}});
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>,1> D(a, playout);
  RAJA::MultiView<TypeParam, RAJA::OffsetLayout<DIM>,1> Dshift1p = D.shift({{N,N}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Cshift(0,y,x),C(0,y-N,x-N));
      ASSERT_EQ(Cshift(1,y,x),C(1,y-N,x-N));
      ASSERT_EQ(Dshift1p(y,0,x),D(y-N,0,x-N));
      ASSERT_EQ(Dshift1p(y,1,x),D(y-N,1,x-N));
      ASSERT_EQ(Dshift1p(y,1,x),C(1,y-N,x-N));
      ASSERT_EQ(Cshift(0,y,x),D(y-N,0,x-N));
    }
  }

  delete[] a0;
  delete[] b0;
}
