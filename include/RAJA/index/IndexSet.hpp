/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining index set classes.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_IndexSet_HPP
#define RAJA_IndexSet_HPP

#include "RAJA/config.hpp"

#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/RAJAVec.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/concepts.hpp"

namespace RAJA
{

enum PushEnd
{
  PUSH_FRONT,
  PUSH_BACK
};

enum PushCopy
{
  PUSH_COPY,
  PUSH_NOCOPY
};

template<typename... TALL>
class TypedIndexSet;

namespace policy
{
namespace indexset
{

///
/// Class representing index set execution policy.
///
/// The first template parameter describes the policy for iterating
/// over segments.  The second describes the policy for executing
/// each segment.
///
template<typename SEG_ITER_POLICY_T, typename SEG_EXEC_POLICY_T = void>
struct ExecPolicy
    : public RAJA::make_policy_pattern_t<SEG_EXEC_POLICY_T::policy,
                                         RAJA::Pattern::forall>
{
  using seg_it   = SEG_ITER_POLICY_T;
  using seg_exec = SEG_EXEC_POLICY_T;
};

}  // end namespace indexset
}  // end namespace policy

using policy::indexset::ExecPolicy;

/*!
 ******************************************************************************
 *
 * \brief  Class representing an index set which is a collection
 *         of segment objects.
 *
 ******************************************************************************
 */
template<typename T0, typename... TREST>
class TypedIndexSet<T0, TREST...> : public TypedIndexSet<TREST...>
{
  using PARENT               = TypedIndexSet<TREST...>;
  static const int T0_TypeId = sizeof...(TREST);

public:
  // Adopt the value type of the first segment type
  using value_type = typename T0::value_type;

  // Ensure that all value types in all segments are the same
  static_assert(std::is_same<value_type, typename PARENT::value_type>::value ||
                    T0_TypeId == 0,
                "All segments must have the same value_type");

  //! Construct empty index set
#if _MSC_VER < 1910
  // this one instance of constexpr does not work on VS2012 or VS2015
  RAJA_INLINE TypedIndexSet() : PARENT() {}
#else
  RAJA_INLINE constexpr TypedIndexSet() : PARENT() {}
#endif

  //! Copy-constructor for index set
  RAJA_INLINE
  TypedIndexSet(TypedIndexSet<T0, TREST...> const& c) : PARENT((PARENT const&)c)
  {
    size_t num = c.data.size();
    data.resize(num);
    for (size_t i = 0; i < num; ++i)
    {
      data[i] = c.data[i];
    }
    // mark all as not owned by us
    owner.resize(num, 0);
  }

  //! Copy-assignment operator for index set
  TypedIndexSet<T0, TREST...>& operator=(const TypedIndexSet<T0, TREST...>& rhs)
  {
    if (&rhs != this)
    {
      TypedIndexSet<T0, TREST...> copy(rhs);
      this->swap(copy);
    }
    return *this;
  }

  //! Destroy index set including all index set segments.
  RAJA_INLINE ~TypedIndexSet()
  {
    size_t num_seg = data.size();
    for (size_t i = 0; i < num_seg; ++i)
    {
      // Only free segment of we allocated it
      if (owner[i])
      {
        delete data[i];
      }
    }
  }

  //! Swap function for copy-and-swap idiom.
  void swap(TypedIndexSet<T0, TREST...>& other)
  {
    // Swap parents data
    PARENT::swap((PARENT&)other);
    // Swap our data
    using std::swap;
    swap(data, other.data);
    swap(owner, other.owner);
  }

  ///
  /// Equality operator for given segment index.
  ///
  /// This is used to implement the == and != operators
  ///
  template<typename P0, typename... PREST>
  RAJA_INLINE bool compareSegmentById(
      size_t segid,
      const TypedIndexSet<P0, PREST...>& other) const
  {
    // drill down our types until we have the right type
    if (getSegmentTypes()[segid] != T0_TypeId)
    {
      // peel off T0
      return PARENT::compareSegmentById(segid, other);
    }

    // Check that other's segid is of type T0
    if (!other.template checkSegmentType<T0>(segid))
    {
      return false;
    }

    // Compare to others segid
    Index_type offset = getSegmentOffsets()[segid];
    return *data[offset] == other.template getSegment<T0>(segid);
  }

  template<typename P0>
  RAJA_INLINE bool checkSegmentType(size_t segid) const
  {
    if (getSegmentTypes()[segid] == T0_TypeId)
    {
      return std::is_same<T0, P0>::value;
    }
    return PARENT::template checkSegmentType<P0>(segid);
  }

  //! get specified segment by ID
  template<typename P0>
  RAJA_INLINE P0& getSegment(size_t segid)
  {
    if (getSegmentTypes()[segid] == T0_TypeId)
    {
      Index_type offset = getSegmentOffsets()[segid];
      return *reinterpret_cast<P0 const*>(data[offset]);
    }
    return PARENT::template getSegment<P0>(segid);
  }

  //! get specified segment by ID
  template<typename P0>
  RAJA_INLINE P0 const& getSegment(size_t segid) const
  {
    if (getSegmentTypes()[segid] == T0_TypeId)
    {
      Index_type offset = getSegmentOffsets()[segid];
      return *reinterpret_cast<P0 const*>(data[offset]);
    }
    return PARENT::template getSegment<P0>(segid);
  }

  //! Returns the number of types this TypedIndexSet can store.
  RAJA_INLINE
  constexpr size_t getNumTypes() const { return 1 + PARENT::getNumTypes(); }

  /*
   * IMPORTANT: Some methods to add a segment to an index set
   *            make a copy of the segment object passed in. Others do not.
   *
   *            The no-copy method names indicate the choice.
   *            The copy/no-copy methods are further distinguished
   *            by taking a const reference (copy) or non-const
   *            pointer (no-copy).
   *
   *            Each method returns true if segment is added successfully;
   *            false otherwise.
   */

  ///
  /// Append an TypedIndexSet of another type to this one.
  ///
  /// This requires the set of types supported by this TypedIndexSet to be a
  /// superset of the one copied in.
  ///
  /// Order is preserved, and all segments are appended to the end of this one
  ///

private:
  template<typename... CALL>
  RAJA_INLINE void push_into(TypedIndexSet<CALL...>& c,
                             PushEnd pend   = PUSH_BACK,
                             PushCopy pcopy = PUSH_COPY)
  {
    Index_type num = getNumSegments();

    if (pend == PUSH_BACK)
    {
      for (Index_type i = 0; i < num; ++i)
      {
        segment_push_into(i, c, pend, pcopy);
      }
    }
    else
    {
      for (Index_type i = num - 1; i > -1; --i)
      {
        segment_push_into(i, c, pend, pcopy);
      }
    }
  }

  static constexpr int value_for(PushEnd end, PushCopy copy)
  {
    return (end == PUSH_BACK) << 1 | (copy == PUSH_COPY);
  }

public:
  template<typename... CALL>
  RAJA_INLINE void segment_push_into(size_t segid,
                                     TypedIndexSet<CALL...>& c,
                                     PushEnd pend   = PUSH_BACK,
                                     PushCopy pcopy = PUSH_COPY)
  {
    if (getSegmentTypes()[segid] != T0_TypeId)
    {
      PARENT::segment_push_into(segid, c, pend, pcopy);
      return;
    }
    Index_type offset = getSegmentOffsets()[segid];
    switch (value_for(pend, pcopy))
    {
      case value_for(PUSH_BACK, PUSH_COPY):
        c.push_back(*data[offset]);
        break;
      case value_for(PUSH_BACK, PUSH_NOCOPY):
        c.push_back_nocopy(data[offset]);
        break;
      case value_for(PUSH_FRONT, PUSH_COPY):
        c.push_front(*data[offset]);
        break;
      case value_for(PUSH_FRONT, PUSH_NOCOPY):
        c.push_front_nocopy(data[offset]);
        break;
    }
  }

  //! Add segment to back end of index set without making a copy.
  template<typename Tnew>
  RAJA_INLINE void push_back_nocopy(Tnew* val)
  {
    push_internal(val, PUSH_BACK, PUSH_NOCOPY);
  }

  //! Add segment to front end of index set without making a copy.
  template<typename Tnew>
  RAJA_INLINE void push_front_nocopy(Tnew* val)
  {
    push_internal(val, PUSH_FRONT, PUSH_NOCOPY);
  }

  //! Add copy of segment to back end of index set.
  template<typename Tnew>
  RAJA_INLINE void push_back(Tnew&& val)
  {
    push_internal(new typename std::decay<Tnew>::type(std::forward<Tnew>(val)),
                  PUSH_BACK, PUSH_COPY);
  }

  //! Add copy of segment to front end of index set.
  template<typename Tnew>
  RAJA_INLINE void push_front(Tnew&& val)
  {
    push_internal(new typename std::decay<Tnew>::type(std::forward<Tnew>(val)),
                  PUSH_FRONT, PUSH_COPY);
  }

  //! Return total length -- sum of lengths of all segments
  RAJA_INLINE size_t getLength() const
  {
    size_t total = PARENT::getLength();
    size_t num   = data.size();
    for (size_t i = 0; i < num; ++i)
    {
      total += data[i]->size();
    }
    return total;
  }

  //! Return total number of segments in index set.
  RAJA_INLINE constexpr size_t getNumSegments() const
  {
    return data.size() + PARENT::getNumSegments();
  }

  ///
  /// Calls the operator "body" with the segment stored at segid.
  ///
  /// This requires that "body" be templated, as the segment will be passed
  /// in as a properly typed object.
  ///
  /// The "args..." are passed-thru to the body as arguments AFTER the segment.
  ///
  RAJA_SUPPRESS_HD_WARN
  template<typename BODY, typename... ARGS>
  RAJA_HOST_DEVICE void segmentCall(size_t segid,
                                    BODY&& body,
                                    ARGS&&... args) const
  {
    if (getSegmentTypes()[segid] != T0_TypeId)
    {
      PARENT::segmentCall(segid, std::forward<BODY>(body),
                          std::forward<ARGS>(args)...);
      return;
    }
    Index_type offset = getSegmentOffsets()[segid];
    body(*data[offset], std::forward<ARGS>(args)...);
  }

protected:
  //! Internal logic to add a new segment -- catch invalid type insertion
  template<typename Tnew>
  RAJA_INLINE void push_internal(Tnew* val,
                                 PushEnd pend   = PUSH_BACK,
                                 PushCopy pcopy = PUSH_COPY)
  {
    static_assert(sizeof...(TREST) > 0, "Invalid type for this TypedIndexSet");
    PARENT::push_internal(val, pend, pcopy);
  }

  //! Internal logic to add a new segment
  RAJA_INLINE void push_internal(T0* val,
                                 PushEnd pend   = PUSH_BACK,
                                 PushCopy pcopy = PUSH_COPY)
  {
    data.push_back(val);
    owner.push_back(pcopy == PUSH_COPY);

    // Determine if we push at the front or back of the segment list
    if (pend == PUSH_BACK)
    {
      // Store the segment type
      getSegmentTypes().push_back(T0_TypeId);

      // Store the segment offset in data[]
      getSegmentOffsets().push_back(data.size() - 1);

      // Store the segment icount
      size_t icount = val->size();
      getSegmentIcounts().push_back(getTotalLength());
      increaseTotalLength(icount);
    }
    else
    {
      // Store the segment type
      getSegmentTypes().push_front(T0_TypeId);

      // Store the segment offset in data[]
      getSegmentOffsets().push_front(data.size() - 1);

      // Store the segment icount
      getSegmentIcounts().push_front(0);
      size_t icount = val->size();
      for (size_t i = 1; i < getSegmentIcounts().size(); ++i)
      {
        getSegmentIcounts()[i] += icount;
      }
      increaseTotalLength(icount);
    }
  }

  //! Returns the number of indices (the total icount of segments
  RAJA_INLINE Index_type& getTotalLength() { return PARENT::getTotalLength(); }

  //! set total length of the indexset
  RAJA_INLINE void setTotalLength(int n) { return PARENT::setTotalLength(n); }

  //! increase the total stored size of the indexset
  RAJA_INLINE void increaseTotalLength(int n)
  {
    return PARENT::increaseTotalLength(n);
  }

public:
  using iterator = Iterators::numeric_iterator<Index_type>;

  //! Get an iterator to the end.
  iterator end() const { return iterator(getNumSegments()); }

  //! Get an iterator to the beginning.
  iterator begin() const { return iterator(0); }

  //! Return the number of elements in the range.
  Index_type size() const { return getNumSegments(); }

  //!  @name TypedIndexSet segment subsetting methods (slices ranges)
  ///
  /// Return a new TypedIndexSet object that contains the subset of
  /// segments in this TypedIndexSet with ids in the interval [begin, end).
  ///
  /// This TypedIndexSet will not change and the created "slice" into it
  /// will not own any of its segments.
  ///
  TypedIndexSet<T0, TREST...> createSlice(int begin, int end)
  {
    TypedIndexSet<T0, TREST...> retVal;

    int minSeg = RAJA::operators::maximum<int> {}(0, begin);
    int maxSeg = RAJA::operators::minimum<int> {}(end, getNumSegments());
    for (int i = minSeg; i < maxSeg; ++i)
    {
      segment_push_into(i, retVal, PUSH_BACK, PUSH_NOCOPY);
    }
    return retVal;
  }

  ///
  /// Return a new TypedIndexSet object that contains the subset of
  /// segments in this TypedIndexSet with ids in the given int array.
  ///
  /// This TypedIndexSet will not change and the created "slice" into it
  /// will not own any of its segments.
  ///
  TypedIndexSet<T0, TREST...> createSlice(const int* segIds, int len)
  {
    TypedIndexSet<T0, TREST...> retVal;

    int numSeg = getNumSegments();
    for (int i = 0; i < len; ++i)
    {
      if (segIds[i] >= 0 && segIds[i] < numSeg)
      {
        segment_push_into(segIds[i], retVal, PUSH_BACK, PUSH_NOCOPY);
      }
    }
    return retVal;
  }

  ///
  /// Return a new TypedIndexSet object that contains the subset of
  /// segments in this TypedIndexSet with ids in the argument object.
  ///
  /// This TypedIndexSet will not change and the created "slice" into it
  /// will not own any of its segments.
  ///
  /// The object must provide methods begin(), end(), and its
  /// iterator type must de-reference to an integral value.
  ///
  template<typename T>
  TypedIndexSet<T0, TREST...> createSlice(const T& segIds)
  {
    TypedIndexSet<T0, TREST...> retVal;
    int numSeg = getNumSegments();
    for (auto& seg : segIds)
    {
      if (seg >= 0 && seg < numSeg)
      {
        segment_push_into(seg, retVal, PUSH_BACK, PUSH_NOCOPY);
      }
    }
    return retVal;
  }

  //! Set [begin, end) interval of segments identified by interval_id
  void setSegmentInterval(size_t interval_id, int begin, int end)
  {
    m_seg_interval_begin[interval_id] = begin;
    m_seg_interval_end[interval_id]   = end;
  }

  //! get lower bound of segment identified with interval_id
  int getSegmentIntervalBegin(size_t interval_id) const
  {
    return m_seg_interval_begin[interval_id];
  }

  //! get upper bound of segment identified with interval_id
  int getSegmentIntervalEnd(size_t interval_id) const
  {
    return m_seg_interval_end[interval_id];
  }

protected:
  //! Returns the mapping of  segment_index -> segment_type
  RAJA_INLINE RAJA::RAJAVec<Index_type>& getSegmentTypes()
  {
    return PARENT::getSegmentTypes();
  }

  //! Returns the mapping of  segment_index -> segment_type
  RAJA_INLINE RAJA::RAJAVec<Index_type> const& getSegmentTypes() const
  {
    return PARENT::getSegmentTypes();
  }

  //! Returns the mapping of  segment_index -> segment_offset
  RAJA_INLINE RAJA::RAJAVec<Index_type>& getSegmentOffsets()
  {
    return PARENT::getSegmentOffsets();
  }

  //! Returns the mapping of  segment_index -> segment_offset
  RAJA_INLINE RAJA::RAJAVec<Index_type> const& getSegmentOffsets() const
  {
    return PARENT::getSegmentOffsets();
  }

  //! Returns the icount of segments
  RAJA_INLINE RAJA::RAJAVec<Index_type>& getSegmentIcounts()
  {
    return PARENT::getSegmentIcounts();
  }

  //! Returns the icount of segments
  RAJA_INLINE RAJA::RAJAVec<Index_type> const& getSegmentIcounts() const
  {
    return PARENT::getSegmentIcounts();
  }

public:
  ///
  /// Equality operator returns true if all segments are equal; else false.
  ///
  /// Note: method does not check equality of anything other than segment
  ///       types and indices; e.g., dependency info not checked.
  ///
  template<typename P0, typename... PREST>
  RAJA_INLINE bool operator==(const TypedIndexSet<P0, PREST...>& other) const
  {
    size_t num_seg = getNumSegments();
    if (num_seg != other.getNumSegments()) return false;

    for (size_t segid = 0; segid < num_seg; ++segid)
    {
      if (!compareSegmentById(segid, other))
      {
        return false;
      }
    }
    return true;
  }

  //! Inequality operator returns true if any segment is not equal, else false.
  template<typename P0, typename... PREST>
  RAJA_INLINE bool operator!=(const TypedIndexSet<P0, PREST...>& other) const
  {
    return (!(*this == other));
  }

private:
  //! vector of TypedIndexSet data objects of type T0
  RAJA::RAJAVec<T0*> data;

  //! vector indicating which segments are owned by the TypedIndexSet
  RAJA::RAJAVec<Index_type> owner;

  //! vector holding user defined begin segment intervals
  RAJA::RAJAVec<Index_type> m_seg_interval_begin;

  //! vector holding user defined end segment intervals
  RAJA::RAJAVec<Index_type> m_seg_interval_end;
};

template<>
class TypedIndexSet<>
{
public:
  // termination case, just to make static_assert work
  using value_type = RAJA::Index_type;

  //! create empty TypedIndexSet
  RAJA_INLINE TypedIndexSet() : m_len(0) {}

  //! dtor cleans up segements that we own (none)
  RAJA_INLINE
  ~TypedIndexSet() {}

  //! Copy-constructor.
  RAJA_INLINE
  TypedIndexSet(TypedIndexSet const& c)
  {
    segment_types   = c.segment_types;
    segment_offsets = c.segment_offsets;
    segment_icounts = c.segment_icounts;
    m_len           = c.m_len;
  }

  //! Swap function for copy-and-swap idiom (deep copy).
  void swap(TypedIndexSet& other)
  {
    using std::swap;
    swap(segment_types, other.segment_types);
    swap(segment_offsets, other.segment_offsets);
    swap(segment_icounts, other.segment_icounts);
    swap(m_len, other.m_len);
  }

protected:
  RAJA_INLINE static size_t getNumTypes() { return 0; }

  template<typename T>
  RAJA_INLINE constexpr bool isValidSegmentType(T const&) const
  {
    // Segment type wasn't found
    return false;
  }

  RAJA_INLINE static int getNumSegments() { return 0; }

  RAJA_INLINE static size_t getLength() { return 0; }

  template<typename BODY, typename... ARGS>
  RAJA_INLINE void segmentCall(size_t, BODY, ARGS...) const
  {}

  RAJA_INLINE RAJA::RAJAVec<Index_type>& getSegmentTypes()
  {
    return segment_types;
  }

  RAJA_INLINE RAJA::RAJAVec<Index_type> const& getSegmentTypes() const
  {
    return segment_types;
  }

  RAJA_INLINE RAJA::RAJAVec<Index_type>& getSegmentOffsets()
  {
    return segment_offsets;
  }

  RAJA_INLINE RAJA::RAJAVec<Index_type> const& getSegmentOffsets() const
  {
    return segment_offsets;
  }

  RAJA_INLINE RAJA::RAJAVec<Index_type>& getSegmentIcounts()
  {
    return segment_icounts;
  }

  RAJA_INLINE RAJA::RAJAVec<Index_type> const& getSegmentIcounts() const
  {
    return segment_icounts;
  }

  RAJA_INLINE Index_type& getTotalLength() { return m_len; }

  RAJA_INLINE void setTotalLength(int n) { m_len = n; }

  RAJA_INLINE void increaseTotalLength(int n) { m_len += n; }

  template<typename P0, typename... PREST>
  RAJA_INLINE bool compareSegmentById(size_t,
                                      const TypedIndexSet<P0, PREST...>&) const
  {
    return false;
  }

  template<typename P0>
  RAJA_INLINE bool checkSegmentType(size_t) const
  {
    return false;
  }

  template<typename P0>
  RAJA_INLINE P0& getSegment(size_t)
  {
    return *((P0*)(this - this));
  }

  template<typename P0>
  RAJA_INLINE P0 const& getSegment(size_t) const
  {
    return *((P0*)(this - this));
  }

  template<typename... CALL>
  RAJA_INLINE void push_into(TypedIndexSet<CALL...>&, PushEnd, PushCopy) const
  {}

  template<typename... CALL>
  RAJA_INLINE void segment_push_into(size_t,
                                     TypedIndexSet<CALL...>&,
                                     PushEnd,
                                     PushCopy) const
  {}

  template<typename Tnew>
  RAJA_INLINE void push(Tnew const&, PushEnd, PushCopy)
  {}

public:
  using iterator = Iterators::numeric_iterator<Index_type>;

  RAJA_INLINE int getStartingIcount(int segid)
  {
    return segment_icounts[segid];
  }

  RAJA_INLINE int getStartingIcount(int segid) const
  {
    return segment_icounts[segid];
  }

  //! Get an iterator to the end.
  iterator end() const { return iterator(getNumSegments()); }

  //! Get an iterator to the beginning.
  iterator begin() const { return iterator(0); }

  //! Return the number of elements in the range.
  Index_type size() const { return getNumSegments(); }

private:
  //! Vector of segment types:    seg_index -> seg_type
  RAJA::RAJAVec<Index_type> segment_types;

  //! offsets into each segment vector:    seg_index -> seg_offset
  //! used as segment_data[seg_type][seg_offset]
  RAJA::RAJAVec<Index_type> segment_offsets;

  //! the icount of each segment
  RAJA::RAJAVec<Index_type> segment_icounts;

  //! Total length of all TypedIndexSet segments.
  Index_type m_len;
};

namespace type_traits
{

template<typename T>
struct is_index_set
    : ::RAJA::type_traits::SpecializationOf<RAJA::TypedIndexSet,
                                            typename std::decay<T>::type>
{};

template<typename T>
struct is_indexset_policy
    : ::RAJA::type_traits::SpecializationOf<RAJA::ExecPolicy,
                                            typename std::decay<T>::type>
{};
}  // namespace type_traits

}  // namespace RAJA

#endif  // closing endif for header file include guard
