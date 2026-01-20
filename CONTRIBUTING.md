
[comment]: # (#################################################################)
[comment]: # (Copyright Lawrence Livermore National Security, LLC and other)
[comment]: # (RAJA Project Developers. See top-level LICENSE and COPYRIGHT)
[comment]: # (files for dates and other details. No copyright assignment is)
[comment]: # (required to contribute to RAJA.)
[comment]: #
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# Contributing to RAJA

This document is intended for developers who want to add new features or
bug fixes to RAJA. It assumes you have some familiarity with git and Github. It
will discuss what a good pull request (PR) looks like, and the tests that your
PR must pass before it can be merged into RAJA.

## Forking RAJA

If you aren't a RAJA developer at LLNL, then you won't have permission to push
new branches to the repository. First, you should create a
[fork](https://github.com/LLNL/RAJA#fork-destination-box). This will create a
copy of the RAJA repository that you own, and will ensure you can push your
changes up to Github and create pull requests.

## Developing a New Feature

New features should be based on the `develop` branch. When you want to create a
new feature, first ensure you have an up-to-date copy of the `develop` branch:

    $ git checkout develop
    $ git pull origin develop

You can now create a new branch to develop your feature on:

    $ git checkout -b feature/<name-of-feature>

Proceed to develop your feature on this branch, and add tests that will exercise
your new code. If you are creating new methods or classes, please add Doxygen
documentation.

Once your feature is complete and your tests are passing, you can push your
branch to Github and create a PR.

## Developing a Bug Fix

First, check if the change you want to make has been fixed in `develop`. If so,
we suggest you either start using the `develop` branch, or temporarily apply the
fix to whichever version of RAJA you are using.

Assuming there is an unsolved bug, first make sure you have an up-to-date copy
of the develop branch:

    $ git checkout develop
    $ git pull origin develop

Then create a new branch for your bug fix:

    $ git checkout -b bugfix/<name-of-bug>

First, add a test that reproduces the bug you have found. Then develop your
bugfix as normal, and ensure to `make test` to check your changes actually fix
the bug.

Once you are finished, you can push your branch to Github, then create a PR.

## Creating a Pull Request

You can create a new PR [here](https://github.com/LLNL/RAJA/compare). Github
has a good [guide](https://help.github.com/articles/about-pull-requests/) to PR
basics if you want some more information. Ensure that your PR base is the
`develop` branch of RAJA.

Add a descriptive title explaining the bug you fixed or the feature you have
added, and put a longer description of the changes you have made in the comment
box.

Once your PR has been created, it will be run through our automated tests and
also be reviewed by RAJA team members. Providing the branch passes both the
tests and reviews, it will be merged into RAJA.

## Tests

RAJA uses Travis CI for continuous integration tests. Our tests are
automatically run against every new pull request, and passing all tests is a
requirement for merging your PR. If you are developing a bugfix or a new
feature, please add a test that checks the correctness of your new code. RAJA
is used on a wide variety of systems with a number of configurations, and adding
new tests helps ensure that all features work as expected across these
environments.

All RAJA tests are in the `test` directory and are split up by backend and feature.
