//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_WORKGROUP_DISPATCHER__
#define __TEST_WORKGROUP_DISPATCHER__

#include "RAJA_test-workgroup.hpp"


template  < typename ForOnePol,
            typename Invoker,
            typename ... CallArgs >
typename  std::enable_if<
            !std::is_base_of<RunOnDevice, ForOnePol>::value
          >::type
call_dispatcher( Invoker invoker,
                 CallArgs... callArgs )
{
  forone<ForOnePol>( [=] () {
    invoker(callArgs...);
  });
}

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
template  < typename ForOnePol,
            typename Invoker,
            typename ... CallArgs >
typename  std::enable_if<
            std::is_base_of<RunOnDevice, ForOnePol>::value
          >::type
call_dispatcher( Invoker invoker,
                 CallArgs... callArgs )
{
  RAJA::tuple<CallArgs...> lambda_capturable_callArgs(callArgs...);
  forone<ForOnePol>( [=] RAJA_DEVICE () {
    camp::invoke(lambda_capturable_callArgs, invoker);
  });
}
#endif

template < typename IndexType,
           typename ... Args >
struct DispatcherTestCallable
{
  DispatcherTestCallable(IndexType* _ptr_call, IndexType _val_call,
                     IndexType* _ptr_dtor, IndexType _val_dtor)
    : ptr_call(_ptr_call)
    , val_call(_val_call)
    , ptr_dtor(_ptr_dtor)
    , val_dtor(_val_dtor)
  { }

  DispatcherTestCallable(DispatcherTestCallable const&) = delete;
  DispatcherTestCallable& operator=(DispatcherTestCallable const&) = delete;

  DispatcherTestCallable(DispatcherTestCallable&& o)
    : ptr_call(o.ptr_call)
    , val_call(o.val_call)
    , ptr_dtor(o.ptr_dtor)
    , val_dtor(o.val_dtor)
    , move_constructed(true)
  {
    o.moved_from = true;
  }
  DispatcherTestCallable& operator=(DispatcherTestCallable&& o)
  {
    ptr_call = o.ptr_call;
    val_call = o.val_call;
    ptr_dtor = o.ptr_dtor;
    val_dtor = o.val_dtor;
    o.moved_from = true;
    return *this;
  }

  ~DispatcherTestCallable()
  {
    *ptr_dtor = val_dtor;
  }

  RAJA_HOST_DEVICE void operator()(IndexType i, Args... args) const
  {
    RAJA_UNUSED_VAR(args...);
    ptr_call[i] = val_call;
  }

private:
  IndexType* ptr_call;
  IndexType  val_call;
  IndexType* ptr_dtor;
  IndexType  val_dtor;
public:
  bool move_constructed = false;
  bool moved_from = false;
};

template < typename ExecPolicy,
           typename DispatchTyper,
           typename IndexType,
           typename WORKING_RES,
           typename ForOnePol >
struct testWorkGroupDispatcherSingle {
template < typename ... Args >
void operator()(RAJA::xargs<Args...>) const
{
  using TestCallable = DispatcherTestCallable<IndexType, Args...>;

  camp::resources::Resource work_res{WORKING_RES()};
  camp::resources::Resource host_res{camp::resources::Host::get_default()};

  static constexpr auto platform = RAJA::platform_of<ExecPolicy>::value;
  using DispatchPolicy = typename DispatchTyper::template type<TestCallable>;
  using Dispatcher_type = RAJA::detail::Dispatcher<
      platform, DispatchPolicy, void, IndexType, Args...>;
  using Invoker_type = typename Dispatcher_type::invoker_type;
  using Dispatcher_cptr_type = typename Dispatcher_type::void_cptr_wrapper;
  const Dispatcher_type* dispatcher =
      RAJA::detail::get_Dispatcher<TestCallable, Dispatcher_type>(ExecPolicy{});

  TestCallable* old_obj = host_res.allocate<TestCallable>(1);
  TestCallable* new_obj = host_res.allocate<TestCallable>(1);
  TestCallable* wrk_obj = work_res.allocate<TestCallable>(1);

  IndexType* chckCall = host_res.allocate<IndexType>(3);
  IndexType* testCall = host_res.allocate<IndexType>(3);
  IndexType* workCall = work_res.allocate<IndexType>(3);

  IndexType* chckDtor = host_res.allocate<IndexType>(3);
  IndexType* testDtor = host_res.allocate<IndexType>(3);


  chckCall[0] = (IndexType)5;
  chckCall[1] = (IndexType)7;
  chckCall[2] = (IndexType)5;

  testCall[0] = (IndexType)5;
  testCall[1] = (IndexType)5;
  testCall[2] = (IndexType)5;

  work_res.memcpy(workCall, testCall, sizeof(IndexType) * 3);
  work_res.wait();

  testCall[0] = (IndexType)0;
  testCall[1] = (IndexType)0;
  testCall[2] = (IndexType)0;


  chckDtor[0] = (IndexType)15;
  chckDtor[1] = (IndexType)17;
  chckDtor[2] = (IndexType)15;

  testDtor[0] = (IndexType)15;
  testDtor[1] = (IndexType)15;
  testDtor[2] = (IndexType)15;


  new(old_obj) TestCallable(workCall, chckCall[1], testDtor+1, chckDtor[1]);

  ASSERT_FALSE(old_obj->move_constructed);
  ASSERT_FALSE(old_obj->moved_from);


  dispatcher->move_construct_destroy(new_obj, old_obj);

  ASSERT_TRUE(new_obj->move_constructed);
  ASSERT_FALSE(new_obj->moved_from);

  ASSERT_EQ(testDtor[0], chckDtor[0]);
  ASSERT_EQ(testDtor[1], chckDtor[1]);
  ASSERT_EQ(testDtor[2], chckDtor[2]);

  testDtor[0] = (IndexType)15;
  testDtor[1] = (IndexType)15;
  testDtor[2] = (IndexType)15;


  work_res.memcpy(wrk_obj, new_obj, sizeof(TestCallable) * 1);
  work_res.wait();

  // move a value onto device and fiddle
  call_dispatcher<ForOnePol, Invoker_type, Dispatcher_cptr_type, IndexType, Args...>(
      dispatcher->invoke, wrk_obj, (IndexType)1, Args{}...);

  work_res.memcpy(testCall, workCall, sizeof(IndexType) * 3);
  work_res.wait();

  ASSERT_EQ(testCall[0], chckCall[0]);
  ASSERT_EQ(testCall[1], chckCall[1]);
  ASSERT_EQ(testCall[2], chckCall[2]);


  dispatcher->destroy(new_obj);

  ASSERT_EQ(testDtor[0], chckDtor[0]);
  ASSERT_EQ(testDtor[1], chckDtor[1]);
  ASSERT_EQ(testDtor[2], chckDtor[2]);


  host_res.deallocate( old_obj );
  host_res.deallocate( new_obj );
  work_res.deallocate( wrk_obj );
  host_res.deallocate( chckCall );
  host_res.deallocate( testCall );
  work_res.deallocate( workCall );
  host_res.deallocate( chckDtor );
  host_res.deallocate( testDtor );
}
};


#if defined(RAJA_ENABLE_HIP) && !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

/// leave unsupported types untested
template <size_t BLOCK_SIZE, bool Async,
          typename IndexType,
           typename WORKING_RES,
          typename ForOnePol
          >
struct testWorkGroupDispatcherSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                     detail::indirect_function_call_dispatch_typer,
                                     IndexType,
                                     WORKING_RES,
                                     ForOnePol> {
template < typename ... Args >
void operator()(RAJA::xargs<Args...>) const
{ }
};
///
template <size_t BLOCK_SIZE, bool Async,
          typename IndexType,
           typename WORKING_RES,
          typename ForOnePol
          >
struct testWorkGroupDispatcherSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                     detail::indirect_virtual_function_dispatch_typer,
                                     IndexType,
                                     WORKING_RES,
                                     ForOnePol> {
template < typename ... Args >
void operator()(RAJA::xargs<Args...>) const
{ }
};

#endif


template <typename T>
class WorkGroupBasicDispatcherSingleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicDispatcherSingleUnitTest);

TYPED_TEST_P(WorkGroupBasicDispatcherSingleUnitTest, BasicWorkGroupDispatcherSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<1>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<2>>::type;
  using Args = typename camp::at<TypeParam, camp::num<3>>::type;
  using ResourceType = typename camp::at<TypeParam, camp::num<4>>::type;
  using ForOneType = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupDispatcherSingle< ExecPolicy, DispatchTyper, IndexType, ResourceType, ForOneType >{}(
      Args{});
}

#endif  //__TEST_WORKGROUP_DISPATCHER__
