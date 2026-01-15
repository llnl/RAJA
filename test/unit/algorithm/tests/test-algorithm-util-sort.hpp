//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing Sorter classes for util sort tests
///

#ifndef __TEST_ALGORITHM_UTIL_SORT_HPP__
#define __TEST_ALGORITHM_UTIL_SORT_HPP__

#include "test-algorithm-sort-utils.hpp"


template < typename test_policy >
using ForoneSynchronize = PolicySynchronize<test_equivalent_exec_policy<test_policy>>;


template < typename test_policy, typename platform = test_platform<test_policy> >
struct InsertionSort;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct InsertionSortPairs;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct ShellSort;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct ShellSortPairs;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct HeapSort;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct HeapSortPairs;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct IntroSort;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct IntroSortPairs;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct MergeSort;

template < typename test_policy, typename platform = test_platform<test_policy> >
struct MergeSortPairs;


template < typename test_policy >
struct InsertionSort<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::insertion_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::insertion_sort(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct InsertionSortPairs<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::insertion_sort[pairs]";
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    auto c = RAJA::zip_span(keys, vals);
    using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
    RAJA::insertion_sort(c, RAJA::compare_first<zip_ref>(comp));
  }
};

template < typename test_policy >
struct ShellSort<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::shell_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::shell_sort(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct ShellSortPairs<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::shell_sort[pairs]";
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    auto c = RAJA::zip_span(keys, vals);
    using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
    RAJA::shell_sort(c, RAJA::compare_first<zip_ref>(comp));
  }
};

template < typename test_policy >
struct HeapSort<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::heap_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::heap_sort(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct HeapSortPairs<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::heap_sort[pairs]";
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    auto c = RAJA::zip_span(keys, vals);
    using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
    RAJA::heap_sort(c, RAJA::compare_first<zip_ref>(comp));
  }
};

template < typename test_policy >
struct IntroSort<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::intro_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::intro_sort(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct IntroSortPairs<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::intro_sort[pairs]";
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    auto c = RAJA::zip_span(keys, vals);
    using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
    RAJA::intro_sort(c, RAJA::compare_first<zip_ref>(comp));
  }
};

template < typename test_policy >
struct MergeSort<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::merge_sort";
  }

  template < typename... Args >
  void operator()(Args&&... args)
  {
    RAJA::merge_sort(std::forward<Args>(args)...);
  }
};

template < typename test_policy >
struct MergeSortPairs<test_policy, RunOnHost>
  : ForoneSynchronize<test_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  const char* name()
  {
    return "RAJA::merge_sort[pairs]";
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    auto c = RAJA::zip_span(keys, vals);
    using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
    RAJA::merge_sort(c, RAJA::compare_first<zip_ref>(comp));
  }
};

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)

template < typename test_policy >
struct InsertionSort<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  InsertionSort()
    : m_name(std::string("RAJA::insertion_sort<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Container >
  void operator()(Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::insertion_sort(c);
    });
  }

  template < typename Container, typename Compare >
  void operator()(Container&& c, Compare comp)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::insertion_sort(c, comp);
    });
  }
};

template < typename test_policy >
struct InsertionSortPairs<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = stable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  InsertionSortPairs()
    : m_name(std::string("RAJA::insertion_sort<") + test_policy_info<test_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      auto c = RAJA::zip_span(keys, vals);
      using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
      RAJA::insertion_sort(c, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename test_policy >
struct ShellSort<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  ShellSort()
    : m_name(std::string("RAJA::shell_sort<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Container >
  void operator()(Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::shell_sort(c);
    });
  }

  template < typename Container, typename Compare >
  void operator()(Container&& c, Compare comp)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::shell_sort(c, comp);
    });
  }
};

template < typename test_policy >
struct ShellSortPairs<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  ShellSortPairs()
    : m_name(std::string("RAJA::shell_sort<") + test_policy_info<test_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      auto c = RAJA::zip_span(keys, vals);
      using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
      RAJA::shell_sort(c, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename test_policy >
struct HeapSort<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  HeapSort()
    : m_name(std::string("RAJA::heap_sort<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Container >
  void operator()(Container c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::heap_sort(c);
    });
  }

  template < typename Container, typename Compare >
  void operator()(Container c, Compare comp)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::heap_sort(c, comp);
    });
  }
};

template < typename test_policy >
struct HeapSortPairs<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  HeapSortPairs()
    : m_name(std::string("RAJA::heap_sort<") + test_policy_info<test_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      auto c = RAJA::zip_span(keys, vals);
      using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
      RAJA::heap_sort(c, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename test_policy >
struct IntroSort<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  IntroSort()
    : m_name(std::string("RAJA::intro_sort<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Container >
  void operator()(Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::intro_sort(c);
    });
  }

  template < typename Container, typename Compare >
  void operator()(Container&& c, Compare comp)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::intro_sort(c, comp);
    });
  }
};

template < typename test_policy >
struct IntroSortPairs<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  IntroSortPairs()
    : m_name(std::string("RAJA::intro_sort<") + test_policy_info<test_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      auto c = RAJA::zip_span(keys, vals);
      using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
      RAJA::intro_sort(c, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

template < typename test_policy >
struct MergeSort<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  MergeSort()
    : m_name(std::string("RAJA::merge_sort<") + test_policy_info<test_policy>::name() + std::string(">"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename Container >
  void operator()(Container&& c)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::merge_sort(c);
    });
  }

  template < typename Container, typename Compare >
  void operator()(Container&& c, Compare comp)
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      RAJA::merge_sort(c, comp);
    });
  }
};

template < typename test_policy >
struct MergeSortPairs<test_policy, RunOnDevice>
  : ForoneSynchronize<test_policy>
{
  using sort_category = unstable_sort_tag;
  using sort_interface = sort_pairs_interface_tag;
  using supports_resource = std::false_type;

  std::string m_name;

  MergeSortPairs()
    : m_name(std::string("RAJA::merge_sort<") + test_policy_info<test_policy>::name() + std::string(">[pairs]"))
  { }

  const char* name()
  {
    return m_name.c_str();
  }

  template < typename KeyContainer, typename ValContainer,
             typename Compare = RAJA::operators::less<RAJA::detail::ContainerRef<KeyContainer>>>
  void operator()(KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
  {
    forone<test_policy>( [=] RAJA_DEVICE() {
      auto c = RAJA::zip_span(keys, vals);
      using zip_ref = RAJA::detail::ContainerRef<camp::decay<decltype(c)>>;
      RAJA::merge_sort(c, RAJA::compare_first<zip_ref>(comp));
    });
  }
};

#endif


using SequentialInsertionSortSorters =
  camp::list<
              InsertionSort<test_seq>,
              InsertionSortPairs<test_seq>
            >;

using SequentialShellSortSorters =
  camp::list<
              ShellSort<test_seq>,
              ShellSortPairs<test_seq>
            >;

using SequentialHeapSortSorters =
  camp::list<
              HeapSort<test_seq>,
              HeapSortPairs<test_seq>
            >;

using SequentialIntroSortSorters =
  camp::list<
              IntroSort<test_seq>,
              IntroSortPairs<test_seq>
            >;

using SequentialMergeSortSorters =
  camp::list<
              MergeSort<test_seq>,
              MergeSortPairs<test_seq>
            >;

#if defined(RAJA_ENABLE_CUDA)

using CudaInsertionSortSorters =
  camp::list<
              InsertionSort<test_cuda>,
              InsertionSortPairs<test_cuda>
            >;

using CudaShellSortSorters =
  camp::list<
              ShellSort<test_cuda>,
              ShellSortPairs<test_cuda>
            >;

using CudaHeapSortSorters =
  camp::list<
              HeapSort<test_cuda>,
              HeapSortPairs<test_cuda>
            >;

using CudaIntroSortSorters =
  camp::list<
              IntroSort<test_cuda>,
              IntroSortPairs<test_cuda>
            >;

using CudaMergeSortSorters =
  camp::list<
              MergeSort<test_cuda>,
              MergeSortPairs<test_cuda>
            >;

#endif

#if defined(RAJA_ENABLE_HIP)

using HipInsertionSortSorters =
  camp::list<
              InsertionSort<test_hip>,
              InsertionSortPairs<test_hip>
            >;

using HipShellSortSorters =
  camp::list<
              ShellSort<test_hip>,
              ShellSortPairs<test_hip>
            >;

using HipHeapSortSorters =
  camp::list<
              HeapSort<test_hip>,
              HeapSortPairs<test_hip>
            >;

using HipIntroSortSorters =
  camp::list<
              IntroSort<test_hip>,
              IntroSortPairs<test_hip>
            >;

using HipMergeSortSorters =
  camp::list<
              MergeSort<test_hip>,
              MergeSortPairs<test_hip>
            >;

#endif

#endif //__TEST_ALGORITHM_UTIL_SORT_HPP__

