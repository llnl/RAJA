#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#------------------------------------------------------------------------------
# This script installs client-side hooks
#------------------------------------------------------------------------------
basedir=`git rev-parse --show-toplevel`
hooksdir="$basedir/.git/hooks/"
cp -v $basedir/scripts/githooks/pre-commit $hooksdir
