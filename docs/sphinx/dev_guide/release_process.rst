.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

*******************************************
RAJA Portability Suite Coordinated Releases
*******************************************

RAJA is part of the **RAJA Portability Suite** set of projects.
Currently, the Suite includes `Umpire <https://github.com/LLNL/Umpire>`_, 
`CHAI <https://github.com/LLNL/CHAI>`_, and 
`camp <https://github.com/LLNL/camp>`_, in addition to RAJA. 

.. important:: The RAJA Portability Suite does coordinated releases, meaning
               that when a non-patch release is done for one, a new version 
               release is done for all Suite projects at the same time. When
               a coordinated release is done, common dependencies, such as
               BLT and camp, are set to the same versions in all Suite projects.

               Individual projects in the RAJA Portability Suite may do
               patch releases (to fix bugs, etc.) independently of other
               Suite projects.

.. _rcbranch-label:

===========================
Release Candidate Branch
===========================

A *release candidate* branch is a temporary branch used to finalize a release.
When the features, documentation, bug fixes, etc. to include in a release are 
complete and merged into the develop branch, a release candidate branch is made
from the develop branch. Typically, a release candidate branch is named 
``rc-<release name>``, or similar. Please see :ref:`release-label` for a 
description of how a release candidate branch is used in the release process. 

Finalizing a release on a release candidate branch involves the following steps:

  #. **Complete the release notes describing the release** in the 
     ``RELEASE_NOTES.md`` file. Describe all API changes, notable new features,
     bug fixes, improvements, build changes, etc. included in the release in 
     appropriately labeled sections of the file. 

     .. important:: Please follow the pattern established in the release notes
                    file used for previous releases. 

     All changes that users need to be aware of should be documented in the
     release notes. Hopefully, the release notes file has been updated along
     with the corresponding changes in PRs that are merged into the develop
     branch. Regardless, it is good practice to look over the commit history
     since the last release to ensure all important changes are documented
     in the release notes.
  #. **Update the version number entries in the code**. The top-level 
     ``CMakeLists.txt`` file must be changed, where the entries: 
     ``RAJA_VERSION_MAJOR``, ``RAJA_VERSION_MINOR``, and 
     ``RAJA_VERSION_PATCHLEVEL`` are defined. These items are used to define 
     corresponding macro values in the ``include/RAJA/config.hpp`` file when 
     the code is configured so that users can access and check the RAJA 
     version in their code by including that header file. The ``version`` and 
     ``release`` fields in the ``RAJA/docs/conf.py`` file must also be changed
     to the new release number. This information is used in the online
     RAJA documentation.

.. important:: **No feature development is done on a release branch. Only bug 
               fixes, release documentation, and other release-oriented changes
               are made on a release candidate branch.**

.. _release-label:

*******************************************
RAJA Release Process
*******************************************

The RAJA release process includes the following sequence of steps:

  #. **Identify all work to be to be included in the release**, such as 
     new features, improvements, and bug fixes.
  #. **Merge each PR** containing work to be included in the release into the 
     develop branch after it is reviewed and approved by the team, and passes
     all the CI checks.
  #. **Make a release candidate branch** from the develop branch. 

     .. important:: Creation of the release candidate branch begins the next 
                    release cycle. While the release candidate branch is being
                    finalized, work can continue on the develop branch.
 
  #. **Finalize the release on the release candidate branch**  by completing 
     remaining release tasks on it. See :ref:`rcbranch-label` for typical 
     tasks to complete.
  #. **Make a PR to merge the release candidate branch into the main branch** 
     when the release candidate branch is ready for the release.

     .. note:: Since the main branch only changes when a release is made, the
               release candidate PR will likely contain many modifications.
               Fortunately, the vast majority of those changes will have been
               reviewed, approved, and merged into the develop branch. 
  
               While the release candidate PR targets main, opening a companion
               draft PR that targets develop will make it easier for team 
               members to review the changes not yet merged into develop.
               Cross-reference the two PRs (release candidate and draft) in 
               their descriptions, and tell reviewers to review the draft PR 
               into develop, but approve the release candidate PR to merge 
               into main. We will merge main into develop after the release 
               to get the missing changes from the release.

  #. **Merge the release candidate branch into the RAJA main branch** when it 
     is approved and all CI checks pass.
  #. **Create the release on GitHub**.

     #. Choose the RAJA main branch as the release target and the option to 
        create the release tag when the release is published. The release tag 
        is the name of the release.

        .. important:: Set the release name (and associated git tag name) 
                       following the convention established for prior releases.
                       Specifically, the tag label should have the format 
                       ``vYYYY.mm.pp``. See :ref:`version-label` for more
                       description of the version naming scheme we use. 

     #. Fill in the release description. Note key features, bugfixes, etc.
        included in the release. The description should summarize the relevant 
        items in the ``RELEASE_NOTES.md`` file in the release candidate 
        branch that was merged. Also, add a note to the release description to 
        remind users to download the gzipped tarfile named for the release 
        (see below) instead of the assets GitHub creates for a release. The 
        assets created by GitHub do not contain the RAJA submodules and may 
        cause issues for users as a result.

        .. important:: For consistency, please follow a similar release 
                       description pattern for all RAJA releases.

     #. Publish the release when it is ready by clicking the button on GitHub.

     #. Generate a release tarfile. Check out the main branch locally and 
        make sure it is up-to-date. Then, run the script::
 
          ./scripts/make_release_tarball.sh 

        from the top-level RAJA directory. The script strips out the Git files
        from the code and generates a tarfile whose name contains the release
        tag name in the top-level RAJA directory of your local repository. If 
        this is successful, the name of the generated gzipped tarfile **will 
        not contain extraneous SHA-1 hash information**. If it does, you need
        to make sure that your local repo checkout is at the same commit as
        the release tag. To do this, run the command::

          git checkout <release tag name>

        in your local clone of the repository.

     #. Edit the release in GitHub and upload the tarfile to the release.

  #. Lastly, **make a PR to merge the main branch into the develop branch**. 
     After it passes all CI checks and is approved, merge the PR. This will 
     ensure that all changes done to finalize the release are included 
     in the develop branch.

After a RAJA release is done, there are other tasks that typically need to be 
performed to update content in other projects. These tasks are described in
:ref:`post_release-label`.

.. _hotfixbranch-label:

===========================
Hotfix Branch
===========================

A *Hotfix* branch is used in the (hopefully!) rare event that a bug is found
shortly after a release that may negatively impact RAJA users. A hotfix branch 
will address the issue in both the develop and main branches.

A hotfix branch is treated like a release candidate branch and it is used to 
generate a *patch release* following the same basic process that is described 
in :ref:`release-label`.

For completeness, the key steps for performing a hotfix (patch) release are:

  #. Make a **hotfix** branch from main at the buggy release tag 
     (hotfix/<issue>), fix the issue on the branch and verify, testing against 
     user code if necessary. Update the release notes and RAJA patch version 
     number as described
     in :ref:`rcbranch-label`.
  #. When the hotfix branch is ready, make a PR for it to be merged
     into the **main branch.** When that is approved and all CI checks pass,
     merge it into the RAJA main branch.
  #. On GitHub, make a new release with a tag for the release. Following our
     convention, the tag label should have the format ``YYYY.mm.pp``, where
     only the **patch** portion of the release tag should differ from the
     last release. In the GitHub release description, note that the release 
     is a bugfix release and describe the issue(s) that it resolves. Also, add 
     a note to the release description to download the gzipped tarfile for the 
     release rather than the assets GitHub creates as part of the release.
  #. Check out the main branch locally and make sure it is up-to-date.     
     Then, generate the tarfile for the release by running the script 
     ``./scripts/make_release_tarball.sh`` from the top-level RAJA directory. 
     If this is successful, a gzipped tarfile whose name includes the release 
     tag **with no extraneous SHA-1 hash information** will be in the top-level
     RAJA directory.
  #. Make a PR to merge the main branch back into the develop branch. After it 
     passes all CI checks and is approved, merge the PR. This will ensure that
     changes for the bugfix will be included in future development on develop.

.. _post_release-label:

=========================
Post-release Activities
=========================

After a RAJA release is complete, other tasks are performed to update content 
in other repositories, typically. These tasks include:

  * Update the `RAJAProxies <https://github.com/LLNL/RAJAProxies>`_ project
    to the new RAJA Portability Suite project release. This typically consists 
    of updating the submodules to the new RAJA Portability Suite project 
    versions, making sure the proxy-apps build and run correctly. When this
    is done, tag a release for proxy-app project.
  * Update the 
    `RAJA Template Project <https://github.com/LLNL/RAJA-project-template>`_ 
    project to the new RAJA release.
  * Update the RAJA Spack package in the 
    `Spack repository <https://github.com/spack/spack>`_. This requires some
    knowledge of Spack and attention to details and Spack conventions. Please
    see :ref:`spack_package-label` for details.

Typically, we also do a new release of the 
`RAJA Performance Suite project <https://github.com/LLNL/RAJAPerf>`_ after
completing a RAJA release. This involves updating the RAJA and BLT submodules
to match the RAJA release and follows the same process as :ref:`release-label`.

.. _spack_package-label:

=========================
Spack Package Update
=========================

After each RAJA release, we update the **RAJA Spack Package** and make a PR to
push it upstream to the `RADIUSS Spack Configs project <https://github.com/LLNL/radiuss-spack-configs>`_, where it will eventually be upstreamed to the 
`Spack project <https://github.com/spack/spack>`_. 

The Spack package is used in RAJA GitLab CI testing and also by RAJA users who 
use Spack to manage their third party library installations. The RAJA Spack
package that we use in our GitLab CI resides in the RADIUSS Spack Configs
submodule. Typically, users will use the Spack package in the Spack repo.
We maintain the RAJA Spack package in the Spack project to be as close 
as possible to the one in the RADIUSS Spack Configs project, which may contain
minor modifications specific to our GitLab CI testing.

Like all Spack packages, the RAJA package is a file containing a Python class. 
The following list contains a description of items to update.

  * **Add a new RAJA version when a release is made.** Near the beginning of
    the ``Raja class`` definition, you will find a list of versions that 
    identify RAJA releases as well as items for the ``develop`` and ``main``
    branches. Adding a new RAJA version is done by adding a line, such as::

     version("2022.10.3", tag="v2022.10.3", submodules=False)

    The last entry indicates whether Spack will use RAJA's submodules when it
    builds RAJA. Currently, we do not use the submodules by default and allow
    Spack to manage the installation of RAJA dependencies.

  * **Add new (build) variants as needed.** The ``variant`` items identify
    how to specify RAJA build variations in a ``Spack spec``. For example,
    the RAJA build variant to enable desul atomics is defined by the line::

     variant("desul", default=False, description="Build Desul Atomics backend") 

    For each variant, there may be an entry in the file to enable the
    corresponding CMake option in the CMake cache, such as::

     entries.append(cmake_cache_option("RAJA_ENABLE_DESUL_ATOMICS", "+desul" in spec))

    There may also be additional options needed. For example, desul also 
    requires that C++ 14 (at least) is enabled for the build. Such information
    may appear as::

     if "+desul" in spec:
         entries.append(cmake_cache_string("BLT_CXX_STD","c++14"))
         if "+cuda" in spec:
             entries.append(cmake_cache_string("CMAKE_CUDA_STANDARD", "14")) 

    When a variant is defined properly, it can be enabled in a Spack spec
    using the shorthand name in the ``variant`` line. For example, to
    enable desul atomics in a Spack build of RAJA, one can include::

     +desul

    in the Spack spec. 

  * **Add new TPL version constraints and package entries as needed.** For 
    example, RAJA depends on BLT to configure a build and the 0.5.2 version 
    of BLT is used for all RAJA versions greater than 2022.10.0. This 
    dependency and version constraint is expressed in the package file as::

     depends_on("blt@0.5.2:", type="build", when="@2022.10.0:")

    In the Spack package file, you will see similar version constraint 
    specifications for RAJA camp and CMake dependencies as well as others.

  * **Add or update configuration package entries as needed.** In addition to 
    the TPL version constraints, there are lines in the package file
    that specify which CMake variables are used to pass options to a CMake
    configuration. For example, the CMake variables that indicate the location
    of BLT and camp to use for a RAJA build are specified on the lines::

      entries.append(cmake_cache_path("BLT_SOURCE_DIR", spec["blt"].prefix))

    and::

      if "camp" in self.spec:
         entries.append(cmake_cache_path("camp_DIR", spec["camp"].prefix)) 

    respectively.

    .. note:: Information that applies to specific build variants, CMake
              variables, etc. should be specified in the appropriate
              Python class function implementation in the package file.
              Specifically,

                * the ``initconfig_compiler_entries`` function contains
                  compiler options
                * the ``initconfig_hardware_entries`` function contains
                  options hardware-based RAJA back-end support
                * the ``initconfig_package_entries`` function contains
                  options for TPLs and build variants that are not
                  specific to a compiler or hardware

One final point is worth noting. We try to add known conflicts to our Spack
package as early as we can. For example, enabling OpenMP in a ROCm compiler
build for HIP is only allowed in recent ROCm releases. So we include this
conflict in our Spack package::

  depends_on("rocprim", when="+rocm")
    with when("+rocm @0.12.0:"):
        ....
        conflicts("+openmp", when="@:2022.03")

This helps users avoid unknown conflicts and potential build or runtime 
failures.

.. important:: It is good practice to add known conflicts to the Spack 
               package as soon as we know about them.

