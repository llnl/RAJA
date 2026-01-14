#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#=============================================================================
# This script is used to update the copyright year in RAJA files.
#
# To use, change the 'sed' commands below as needed to modify the content that
# is changed in each file and what it is changed to.
#
#=============================================================================

echo LICENSE
cp LICENSE LICENSE.sed.bak
sed "s/Copyright (c) 2016-2025/Copyright (c) 2016-2026/" LICENSE.sed.bak > LICENSE
rm LICENSE.sed.bak

echo docs/conf.py
cp docs/conf.py docs/conf.py.sed.bak
sed "s/2016-25/2016-26/" docs/conf.py.sed.bak > docs/conf.py
rm docs/conf.py.sed.bak
