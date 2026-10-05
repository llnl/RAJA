//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

#include "RAJA/pattern/tensor.hpp"

namespace
{

struct MockTensorExpr
    : public RAJA::internal::expt::ET::TensorExpressionBase<MockTensorExpr>
{
  using element_type = int;
  using index_type   = int;
  using result_type  = MockTensorExpr;

  static constexpr camp::idx_t s_num_dims = 2;

  constexpr MockTensorExpr(index_type dim0, index_type dim1)
      : m_dim0(dim0),
        m_dim1(dim1)
  {}

  constexpr index_type getDimSize(index_type dim) const
  {
    return dim == 0 ? m_dim0 : m_dim1;
  }

  template<typename TILE_TYPE>
  RAJA_HOST_DEVICE RAJA_INLINE result_type eval(TILE_TYPE const&) const
  {
    return *this;
  }

  index_type m_dim0;
  index_type m_dim1;
};

}  // namespace

TEST(TensorBinaryOperatorTraits, BinaryOpsReportExpectedShape)
{
  constexpr MockTensorExpr lhs(5, 7);
  constexpr MockTensorExpr rhs(9, 11);

  auto lplusr = lhs + rhs;
  auto rplusl = rhs + lhs;

  auto lminusr = lhs - rhs;
  auto rminusl = rhs - lhs;

  // Binary expressions should report the dimensions of the left-hand operand.
  ASSERT_EQ(lhs.getDimSize(0), lplusr.getDimSize(0));  // 5
  ASSERT_EQ(lhs.getDimSize(1), lplusr.getDimSize(1));  // 7

  ASSERT_EQ(rhs.getDimSize(0), rplusl.getDimSize(0));  // 9
  ASSERT_EQ(rhs.getDimSize(1), rplusl.getDimSize(1));  // 11

  ASSERT_EQ(lhs.getDimSize(0), lminusr.getDimSize(0));  // 5
  ASSERT_EQ(lhs.getDimSize(1), lminusr.getDimSize(1));  // 7

  ASSERT_EQ(rhs.getDimSize(0), rminusl.getDimSize(0));  // 9
  ASSERT_EQ(rhs.getDimSize(1), rminusl.getDimSize(1));  // 11

  // Tensor dimensions should be reported when an operand is scalar.
  auto tpluss = lhs + 13;
  auto splust = 15 + lhs;

  auto tminuss = lhs - 17;
  auto sminust = 19 - lhs;

  ASSERT_EQ(lhs.getDimSize(0), tpluss.getDimSize(0));  // 5
  ASSERT_EQ(lhs.getDimSize(1), tpluss.getDimSize(1));  // 7

  ASSERT_EQ(lhs.getDimSize(0), splust.getDimSize(0));  // 5
  ASSERT_EQ(lhs.getDimSize(1), splust.getDimSize(1));  // 7

  ASSERT_EQ(lhs.getDimSize(0), tminuss.getDimSize(0));  // 5
  ASSERT_EQ(lhs.getDimSize(1), tminuss.getDimSize(1));  // 7

  ASSERT_EQ(lhs.getDimSize(0), sminust.getDimSize(0));  // 5
  ASSERT_EQ(lhs.getDimSize(1), sminust.getDimSize(1));  // 7
}
