//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"
#include "RAJA_unit-test-types.hpp"

RAJA_INDEX_VALUE(TX, "TX");
RAJA_INDEX_VALUE(TIX, "TIX");
RAJA_INDEX_VALUE(TIY, "TIY");
RAJA_INDEX_VALUE(TIL, "TIL");

template<typename T>
class TypedViewUnitTest : public ::testing::Test {};

template<typename T>
class OffsetLayoutViewUnitTest : public ::testing::Test {};

template<typename T>
class TypedIntegralViewUnitTest : public ::testing::Test {};

TYPED_TEST_SUITE(TypedViewUnitTest, UnitIntFloatTypes);
TYPED_TEST_SUITE(OffsetLayoutViewUnitTest, UnitIntFloatTypes);
TYPED_TEST_SUITE(TypedIntegralViewUnitTest, UnitIntFloatTypes);

TYPED_TEST(TypedViewUnitTest, Constructors)
{
  using layout = RAJA::Layout<1>;

  const TypeParam val = 2;

  TypeParam data[10];
  data[0] = val;

  const TypeParam data2[10] = {2, 0, 1, 3, 4, 5, 6, 7, 8, 9};

  RAJA::View<TypeParam, layout> view(data, layout(10));
  ASSERT_EQ(val, view(0));

  /*
   * Should be able to construct a View from const array
   */
  RAJA::View<const TypeParam, layout> viewc(data2, layout(10));
  ASSERT_EQ(val, viewc(0));

  /*
   * Should be able to construct a non-const View from a non-const View
   */
  RAJA::View<TypeParam, layout> view2(view);
  ASSERT_EQ(val, view2(0));

  /*
   * Should be able to construct a const View from a non-const View
   */
  RAJA::View<TypeParam const, layout> const_view(view);
  ASSERT_EQ(val, const_view(0));

  /*
   * Should be able to construct a const View from a const View
   */
  RAJA::View<TypeParam const, layout> const_view2(const_view);
  ASSERT_EQ(val, const_view2(0));
}

TYPED_TEST(TypedViewUnitTest, Accessor)
{

  const int Nx = 3;
  const int Ny = 5;
  const int Nz = 2;
  const int N  = Nx*Ny*Nz;
  TypeParam *a = new TypeParam[N];

  int iter{0};
  for(TypeParam i=0; i<TypeParam{N}; ++i)
  {
    a[iter] = TypeParam{i};
    ++iter;
  }

  /*
   * 1D Accessor
   */
  RAJA::View<TypeParam, RAJA::Layout<1>> view_1D(a,N);
  TypeParam val{0};
  for(int i=0; i<N; ++i) {
    ASSERT_EQ(val, view_1D(i));
    val++;
  }

  /*
   * 2D Accessor
   */
  RAJA::View<TypeParam, RAJA::Layout<2>> view_2D(a,Ny,Nx);
  val = TypeParam{0};
  for(int j=0; j<Ny; ++j) {
    for(int i=0; i<Nx; ++i) {
      ASSERT_EQ(val, view_2D(j,i));
      val++;
    }
  }

  /*
   * 3D Accessor
   */
  RAJA::View<TypeParam, RAJA::Layout<3>> view_3D(a,Nz,Ny,Nx);
  val = TypeParam{0};
  for(int k=0; k<Nz; ++k) {
    for(int j=0; j<Ny; ++j) {
      for(int i=0; i<Nx; ++i) {
        ASSERT_EQ(val, view_3D(k,j,i));
        val++;
      }
    }
  }

  delete[] a;
}
TYPED_TEST(TypedIntegralViewUnitTest, TypedAccessor)
{

  const int Nx = 3;
  const int Ny = 5;
  const int Nz = 2;
  const int N  = Nx*Ny*Nz;
  TypeParam *a = new TypeParam[N];

  int iter{0};
  for(TypeParam i=0; i<TypeParam{N}; ++i)
  {
    a[iter] = TypeParam{i};
    ++iter;
  }

  /*
   * 1D Typed Accessor
   */
  RAJA::TypedView<TypeParam, RAJA::Layout<1>, TypeParam> view_1D(a,N);
  TypeParam val{0};
  for(TypeParam i=0; i<N; ++i) {
    ASSERT_EQ(val, view_1D(i));
    val++;
  }

  /*
   * 2D Typed Accessor
   */
  RAJA::View<TypeParam, RAJA::Layout<2>> view_2D(a,Ny,Nx);
  val = TypeParam{0};
  for(TypeParam j=0; j<Ny; ++j) {
    for(TypeParam i=0; i<Nx; ++i) {
      ASSERT_EQ(val, view_2D(j,i));
      val++;
    }
  }

  /*
   * 3D Typed Accessor
   */
  RAJA::View<TypeParam, RAJA::Layout<3>> view_3D(a,Nz,Ny,Nx);
  val = TypeParam{0};
  for(TypeParam k=0; k<Nz; ++k) {
    for(TypeParam j=0; j<Ny; ++j) {
      for(TypeParam i=0; i<Nx; ++i) {
        ASSERT_EQ(val, view_3D(k,j,i));
        val++;
      }
    }
  }

  delete[] a;
}

TYPED_TEST(OffsetLayoutViewUnitTest, View)
{
  TypeParam* data = new TypeParam[10];

  using layout = RAJA::OffsetLayout<>;

  /*
   * View is constructed by passing in the layout.
   */
  std::array<RAJA::Index_type, 1> lower{{1}};
  std::array<RAJA::Index_type, 1> upper{{11}};
  RAJA::View<TypeParam, layout> view(data, RAJA::make_offset_layout<1>(lower, upper));

  for (int i = 0; i < 10; i++) {
    data[i] = static_cast<TypeParam>(i);
  }

  ASSERT_EQ(data[0], view(1));
  ASSERT_EQ(data[9], view(10));

  delete[] data;
}

TYPED_TEST(TypedViewUnitTest, Shift1D)
{

  int N = 10;
  TypeParam *a = new TypeParam[N];
  TypeParam *b = new TypeParam[N];

  /*
   * Create a view from a base view
   */
  const int DIM = 1;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0}},{{N}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::View<TypeParam, RAJA::Layout<DIM>> B(a,N);
  RAJA::TypedView<TypeParam, RAJA::Layout<DIM>,TX> C(a,N);

  for(int i=0; i<N; ++i) {
    A(i) = static_cast<TypeParam>(i + 1);
  }

  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N}});

  /*
   * Create a view from a base view with an offsetlayout
   */
  RAJA::TypedView<TypeParam, RAJA::OffsetLayout<DIM>, TX> Cshift = C.shift({{N}});

  for(int i=N; i<2*N; ++i)
  {
    ASSERT_EQ(Ashift(i),A(i-N));
    ASSERT_EQ(Bshift(i),B(i-N));
  }

  for(TX tx=TX{N}; tx<TX{2*N}; tx++)
  {
    ASSERT_EQ(Cshift(tx),C(tx-N));
  }

  /*
   * Create a shifted view from a view with a typed layout
   */
  using TLayout = RAJA::TypedLayout<TIL, RAJA::tuple<TIX>>;
  using TOffsetLayout = RAJA::TypedOffsetLayout<TIL, RAJA::tuple<TIX>>;

  TLayout myLayout(10);

  RAJA::View<TypeParam, TLayout> D(a, myLayout);
  RAJA::View<TypeParam, TOffsetLayout> Dshift = D.shift({{N}});

  for(TIX i=TIX{N}; i<TIX{2*N}; ++i)
  {
    ASSERT_EQ(Dshift(i),D(i-N));
  };

  delete[] a;
  delete[] b;

}


TYPED_TEST(TypedViewUnitTest, Shift2D)
{

  int N = 10;
  TypeParam *a = new TypeParam[N*N];
  TypeParam *b = new TypeParam[N*N];

  const int DIM = 2;
  RAJA::OffsetLayout<DIM> layout = RAJA::make_offset_layout<DIM>({{0,0}},{{N,N}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> A(a,layout);
  RAJA::View<TypeParam, RAJA::Layout<DIM>> B(a,N,N);

  for(int y=0; y<N; ++y) {
    for(int x=0; x<N; ++x) {
      A(y,x) = static_cast<TypeParam>(x + N*y);
    }
  }

  /*
   * Create a view from a base view with an offsetlayout
   */
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Ashift = A.shift({{N,N}});
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Bshift = B.shift({{N,N}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Ashift(y,x),A(y-N,x-N));
      ASSERT_EQ(Bshift(y,x),B(y-N,x-N));
    }
  }

  /*
   * Create a view from a base view with permuted layout
   */
  std::array< RAJA::idx_t, 2> perm {{1, 0}};
  RAJA::OffsetLayout<2> playout =
    RAJA::make_permuted_offset_layout<2>( {{0, 0}}, {{N, N}}, perm );

  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> C(a, playout);
  RAJA::View<TypeParam, RAJA::OffsetLayout<DIM>> Cshift = C.shift({{N,N}});

  for(int y=N; y<N+N; ++y) {
    for(int x=N; x<N+N; ++x) {
      ASSERT_EQ(Cshift(y,x),C(y-N,x-N));
    }
  }

  delete[] a;
  delete[] b;
}
