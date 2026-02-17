/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implmentation file for aligned range index set builder methods.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>

#include "RAJA/index/IndexSetBuilders.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "camp/resource.hpp"

namespace RAJA
{

/*
 ******************************************************************************
 *
 * Generate an index set with aligned Range segments and List segments,
 * as needed, from given array of indices.
 *
 ******************************************************************************
 */
void buildIndexSetAligned(
    RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment>& iset,
    camp::resources::Resource work_res,
    const RAJA::Index_type* const indices_in,
    RAJA::Index_type length,
    RAJA::Index_type range_min_length,
    RAJA::Index_type range_align)
{
  if (length == 0) return;

  /* only transform relatively large */
  if (length > range_min_length)
  {
    /* build a rindex array from an index array */
    RAJA::Index_type docount = 0;
    RAJA::Index_type inrange = -1;

    /****************************/
    /* first, gather statistics */
    /****************************/

    RAJA::Index_type scanVal    = indices_in[0];
    RAJA::Index_type sliceCount = 0;
    for (RAJA::Index_type ii = 1; ii < length; ++ii)
    {
      RAJA::Index_type lookAhead = indices_in[ii];

      if (inrange == -1)
      {
        if ((lookAhead == scanVal + 1) && ((scanVal % range_align) == 0))
        {
          inrange = 1;
        }
        else
        {
          inrange = 0;
        }
      }

      if (lookAhead == scanVal + 1)
      {
        if ((inrange == 0) && ((scanVal % range_align) == 0))
        {
          if (sliceCount != 0)
          {
            docount += 1 + sliceCount; /* length + singletons */
          }
          inrange    = 1;
          sliceCount = 0;
        }
        ++sliceCount; /* account for scanVal */
      }
      else
      {
        if (inrange == 1)
        {
          /* we can tighten this up by schleping any trailing */
          /* sigletons off into the subsequent singleton */
          /* array.  We would then also need to recheck the */
          /* final length of the range to make sure it meets */
          /* our minimum length crietria.  If it doesnt, */
          /* we need to emit a random array instead of */
          /* a range array */
          ++sliceCount;
          docount += 2; /* length + begin */
          inrange    = 0;
          sliceCount = 0;
        }
        else
        {
          ++sliceCount; /* account for scanVal */
        }
      }

      scanVal = lookAhead;
    }  // end loop to gather statistics

    if (inrange != -1)
    {
      if (inrange)
      {
        ++sliceCount;
        docount += 2; /* length + begin */
      }
      else
      {
        ++sliceCount;
        docount += 1 + sliceCount; /* length + singletons */
      }
    }
    else if (scanVal != -1)
    {
      ++sliceCount;
      docount += 2;
    }
    ++docount; /* zero length termination */

    /* What is the cutoff criteria for generating the rindex array? */
    if (docount < (length * (range_align - 1)) / range_align)
    {
      /* The rindex array can either contain a pointer into the */
      /* original index array, *or* it can repack the data from the */
      /* original index array.  Benefits of repacking could include */
      /* better use of hardware prefetch streams, or guaranteeing */
      /* alignment of index array segments. */

      /*******************************/
      /* now, build the rindex array */
      /*******************************/

      RAJA::Index_type dobegin;
      inrange = -1;

      scanVal    = indices_in[0];
      sliceCount = 0;
      dobegin    = scanVal;
      for (RAJA::Index_type ii = 1; ii < length; ++ii)
      {
        RAJA::Index_type lookAhead = indices_in[ii];

        if (inrange == -1)
        {
          if ((lookAhead == scanVal + 1) && ((scanVal % range_align) == 0))
          {
            inrange = 1;
          }
          else
          {
            inrange = 0;
            dobegin = ii - 1;
          }
        }
        if (lookAhead == scanVal + 1)
        {
          if ((inrange == 0) && ((scanVal % range_align) == 0))
          {
            if (sliceCount != 0)
            {
              iset.push_back(
                  ListSegment(&indices_in[dobegin], sliceCount, work_res));
            }
            inrange    = 1;
            dobegin    = scanVal;
            sliceCount = 0;
          }
          ++sliceCount; /* account for scanVal */
        }
        else
        {
          if (inrange == 1)
          {
            /* we can tighten this up by schleping any trailing */
            /* sigletons off into the subsequent singleton */
            /* array.  We would then also need to recheck the */
            /* final length of the range to make sure it meets */
            /* our minimum length crietria.  If it doesnt, */
            /* we need to emit a random array instead of */
            /* a range array */
            ++sliceCount;
            iset.push_back(RangeSegment(dobegin, dobegin + sliceCount));
            inrange    = 0;
            sliceCount = 0;
            dobegin    = ii;
          }
          else
          {
            ++sliceCount; /* account for scanVal */
          }
        }

        scanVal = lookAhead;
      }  // for (RAJA::Index_type ii ...

      if (inrange != -1)
      {
        if (inrange)
        {
          ++sliceCount;
          iset.push_back(RangeSegment(dobegin, dobegin + sliceCount));
        }
        else
        {
          ++sliceCount;
          iset.push_back(
              ListSegment(&indices_in[dobegin], sliceCount, work_res));
        }
      }
      else if (scanVal != -1)
      {
        iset.push_back(ListSegment(&scanVal, 1, work_res));
      }
    }
    else
    {  // !(docount < (length*range_align-1))/range_align)
      iset.push_back(ListSegment(indices_in, length, work_res));
    }
  }
  else
  {  // else !(length > range_min_length)
    iset.push_back(ListSegment(indices_in, length, work_res));
  }
}

}  // namespace RAJA
