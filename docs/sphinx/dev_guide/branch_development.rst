.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _branching-label:

*******************************************
RAJA Branch Development
*******************************************

========================
Gitflow Branching Model
========================

The RAJA project uses a simple branch development model based on 
`Gitflow <https://datasift.github.io/gitflow/IntroducingGitFlow.html>`_.
The Gitflow model is centered around software releases. It is a simple
workflow that makes clear which branches correspond to which phases of
development. Those phases are represented explicitly in the branch names and
structure of the repository. As in other branching models, developers develop 
code on a local branch and push their work to a central repository.

---------------------------------
Persistent, Protected Branches
---------------------------------

The **main** and **develop** branches are the two primary branches we use.
They always exist and are protected in the RAJA GitHub project, meaning that
changes to them can only occur as a result of approved pull requests. The 
distinction between the main and develop branches is an important part of 
Gitflow.

  * The *main* branch records the release history of the project. Each time 
    the main branch is changed, a new tag for a new code version is made. 
    See :ref:`version-label` for a description of the version labeling scheme 
    we use.

  * The *develop* branch is used to integrate and test new features and most
    bug fixes before they are merged into the main branch for a release.

.. important:: **Development never occurs directly on the main branch or 
               develop branch.**

All other branches are temporary and are used to perform specific development 
tasks. When such a branch is no longer needed (e.g., after it is merged), the 
branch is deleted typically.

----------------
Feature Branches
----------------

A *feature* branch is created from another branch (usually develop) and is 
used to develop new features, bug fixes, etc. before they are merged to develop
and eventually main. *Feature branches are temporary*, living only as long as 
they are needed to complete development tasks they contain.

Each new feature, or other well-defined portion of work, is developed on its
own feature branch. We typically include a label, such as  "feature" or 
"bugfix", in a feature branch name to make it clear what type of work is being 
done on the branch. For example, **feature/<name-of-feature>** for a new 
feature, **bugfix/<issue>** for a bugfix, etc.

.. important:: When doing development on a feature branch, it is good practice
               to regularly push your changes to the GitHub repository 
               as a backup mechanism. Also, regularly merge the RAJA develop 
               branch into your feature branch so that it does not diverge 
               too much from other development on the project. This will help 
               reduce merge conflicts that you must resolve when your work is 
               ready to be merged into the RAJA develop branch.

When a portion of development is complete and ready to be merged into the
develop branch, submit a *pull request* (PR) for review by other team members. 
When all issues and comments arising in PR review discussion have been 
addressed, the PR has been approved, and all continuous integration checks 
have passed, the pull request can be merged.

.. important:: **Feature branches almost never interact directly with the main
               branch.** One exception is when a bug fix is needed in
               the main branch to tag a patch release.

---------------------------
Other Important Branches
---------------------------

**Release candidate** and **hotfix** branches are two other important 
temporary branch types in Gitflow. They will be explained in the
:ref:`release-label` section.

----------------------
Gitflow Illustrated
----------------------

The figure below shows the basics of how branches interact in Gitflow.

.. figure:: ./figures/git-workflow-gitflow2.png

   This figure shows typical interactions between key branches in the Gitflow
   workflow. Here, development is shown following the v0.1.0 release. While
   development was ongoing, a bug was found and a fix was needed to be made 
   available to users. A *hotfix* branch was made off of main and merged back 
   to main after the issue was addressed. Release v0.1.1 was tagged and main 
   was merged into develop so that it would not recur. Work started on a 
   feature branch before the v0.1.1 release and was merged into develop after 
   the v0.1.1 release. Then, a release candidate branch was made from develop. 
   The release was finalized on that branch after which it was merged into 
   main, and the v0.2.0 release was tagged. Finally, main was merged into 
   develop.
