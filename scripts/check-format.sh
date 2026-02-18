#!/bin/bash

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
#  and RAJA project contributors. See the RAJA/LICENSE file for details.
#
#  SPDX-License-Identifier: (BSD-3-Clause)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List of file extensions to format
file_extensions=("cpp" "h" "c" "hpp")
# List of directories where style is enforced
file_directories=("src" "include")

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

# Function to format files
function format_file() {
  local loc_file="$1"
  "$__clang_format_executable" -i "$loc_file"
}

function iterate_files() {
  local dir="$1"
  for entry in "$dir"/*; do
      if [ -d "$entry" ]; then
          iterate_files "$entry"
      elif [ -f "$entry" ]; then
          format_file "$entry"
      fi
  done
}

find . -type d -exec process_directory {} \;

__clang_format_executable=""
# Check if clang-format exists
if command -v clang-format &> /dev/null; then
  __clang_format_executable=clang-format
else
  echo "Clang-format is not available.  This is an internal CI failure.  Please file a "
       "bug report under RAJA on github"
  exit 1
fi

# Check the major version of the provided clang-format
VERSION_STRING=$($__clang_format_executable --version)
MAJOR_VERSION=$(echo "$VERSION_STRING" | grep -oP '\d+' | head -1)

if [ "$MAJOR_VERSION" != "14" ]; then
  echo "An incorrect version of clang-format was found in the available CI docker image."
       "This is an internal CI failure.  Please file a bug report."
  exit 1
fi

# Find and format staged files with the specified extensions
iterate_files $RAJA_DIR


function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

or_die cd RAJA
git submodule init 
git submodule update 

echo "~~~~ helpful info ~~~~"
echo "USER="`id -u -n`
echo "PWD="`pwd`
echo "HOST_CONFIG=$HOST_CONFIG"
echo "CMAKE_EXTRA_FLAGS=$CMAKE_EXTRA_FLAGS"
echo "~~~~~~~~~~~~~~~~~~~~~~"

echo "~~~~~~~~~ls -al /~~~~~~~~~~"
ls -al /
echo "~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~ls -al /home~~~~~~~~~~"
ls -al /home
echo "~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~ls -al /home/raja~~~~~~~~~~"
ls -al /home/raja
echo "~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~~~~~ls -al /home/raja/RAJA~~~~~~~~~~"
ls -al /home/raja/RAJA
echo "~~~~~~~~~~~~~~~~~~~~~~"


echo "~~~~~~ RUNNING CMAKE ~~~~~~~~"
mkdir build
cmake ../ -DENABLE_CLANGFORMAT=On
cd build
echo "~~~~~~ RUNNING make check ~~~~~~~~"
or_die make VERBOSE=1 check

exit 0
