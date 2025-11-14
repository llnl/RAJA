.. ##
.. ## Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _ci_tasks-label:

*****************************************************
Continuous Integration (CI) Testing Maintenance Tasks
*****************************************************

In :ref:`ci-label`, we described RAJA CI testing workflows. This section
describes how to perform common RAJA CI testing maintenance tasks.

.. _gitlab_ci_tasks-label:

===============
GitLab CI Tasks
===============

The tasks in this section apply to GitLab CI testing on Livermore 
Computing (LC) platforms. LC folks and others maintain Confluence pages
with a lot of useful information for setting up and maintaining GitLab CI
for a project, mirroring a GitHub to GitLab, etc. Please refer to `LC GitLab CI <https://lc.llnl.gov/confluence/display/GITLAB/GitLab+CI>`_ for such information.

Changing build and test configurations
--------------------------------------

The configurations that are tested in RAJA are defined by a Spack spec in one 
of two places, depending on whether it is *shared* with other projects or
it is specific to RAJA. The details are described
in :ref:`gitlab_ci_pipelines-label`. Each spec contains information (compiler
and version, build variants, etc.) that must be consistent with the 
build specs defined in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project, which also includes
the RAJA Spack package. The RADIUSS Spack Configs project is included as a
RAJA submodule in the ``RAJA/scripts`` directory.

Removing a configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

To remove a RAJA-specific test configuration, simply delete the entry for it in
the ``RAJA/.gitlab/jobs/<MACHINE>.yml`` file where it is defined. Here,
``MACHINE`` is the name of an LC platform where GitLab CI is run.

To remove a shared configuration, it must be removed from the appropriate
``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project.  Create a branch
there, remove the job entry, and create a pull request.

.. important:: The RADIUSS Spack Configs project is used by several other
   projects.  When changing a shared configuration file, please make sure the
   change is OK with those other projects. Typically, shared configurations
   are only changed when it makes sense to update compilers for all projects,
   such as when system default compiler versions change.

Adding a configuration
^^^^^^^^^^^^^^^^^^^^^^^^

To add a RAJA-specific test configuration, add an entry for it to the
``RAJA/.gitlab/jobs/<MACHINE>.yml`` file, where ``MACHINE`` is the name of the
LC platform where it will be run. When adding a test configuration, it is
important to note two items that must be specified properly:

  * Each jobs must have a unique **job label**, which identifies it in the 
    machine configuration file and also on a web page for a GitLab CI pipeline
  * The **Spack spec** name identifies the compiler and version,
    compiler flags, build options, etc. must match an existing spec in
    the `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-configs>`_
    project. Also, the build options must be consistent with the variants
    defined in the RAJA package in that project.

For example, an entry for a build using the clang 12.0.1 compiler with CUDA 
11.5.0 on the LC lassen machine would be something like this:

.. code-block:: bash

  clang_12_0_1_cuda_11_5_0:
    variables:
      SPEC: " ~shared +openmp +tests +cuda cuda_arch=70 %clang@12.0.1 ^cuda@11.5.0"
    extends: .job_on_lassen

Here, we enable OpenMP and CUDA, both of which must be enabled to test those
RAJA back-ends, and specify the CUDA target architecture 'sm_70'.

To add a shared configuration, it must be added to the appropriate
``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project. Create a branch
there, add the job entry, and create a pull request.

.. important:: The RADIUSS Spack Configs project is used by several other
   projects. When changing a shared configuration file, please make sure the
   change is OK with those other projects. Typically, shared configurations
   are only changed when it makes sense to update compilers for all projects,
   such as when system default compiler versions change.

Modifying a configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

To change an existing configuration, change the relevant information in the
configuration in the appropriate ``RAJA/.gitlab/jobs/<MACHINE>.yml`` file. Make
sure to also modify the job label as needed, so it is descriptive of the
configuration is unique with respect to the others that are being run.

To modify a shared configuration, it must be changed in the appropriate
``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the `RADIUSS Spack Configs
<https://github.com/LLNL/radiuss-spack-configs>`_ project. Create a branch
there, modify the job entry, and create a pull request.

.. important:: Build spec information used in RAJA GitLab CI pipelines must
   exist in the ``compilers.yaml`` file and/or ``packages.yaml`` file for the
   appropriate system type in the `RADIUSS Spack Configs
   <https://github.com/LLNL/radiuss-spack-configs>`_ repo.

   If the desired entry is not there, but exists in a newer version of the 
   RADIUSS Spack Configs project, update the RAJA submodule to use the newer
   version. If the information does not exist in any version of the RADIUSS
   Spack Configs project, create a branch there, add the needed spec info,
   and create a pull request. Then, when that PR is merged, update the RAJA
   submodule for the RADIUSS Spack Configs project to the new version.

Changing run parameters
^^^^^^^^^^^^^^^^^^^^^^^

The parameters for each system/scheduler on which we run GitLab CI for
RAJA, such as job time limits, resource allocations, etc. are defined in the 
``RAJA/.gitlab/custom-jobs-and-variables.yml`` file. This information can
remain as is, for the most part, and should not be changed unless absolutely 
necessary.

For example, sometimes a particular job will take longer to build and run than
the default allotted time for jobs on a machine. In this case, the time for the
job can be adjusted in the job entry in the associated
``RAJA/.gitlab/jobs/<MACHINE>.yml`` file. For example:

.. code-block:: bash

  gcc_8_1_0:
  variables:
    SPEC: " ${PROJECT_DANE_VARIANTS} %gcc@8.1.0 ${PROJECT_DANE_DEPS}"
    DANE_BUILD_AND_TEST_JOB_ALLOC: "--time=60 --nodes=1"
  extends: .job_on_dane

This example sets the build and test allocation time to 60 minutes and the
the run resource to one node.

Allowing failures
^^^^^^^^^^^^^^^^^

Sometimes a shared job configuration is known to fail for RAJA. To allow
the job to fail without the CI check associated with it failing, we can
annotate the job for this. For example:

.. code-block:: bash

  ibm_clang_9_0_0:
    variables:
      SPEC: " ${PROJECT_LASSEN_VARIANTS} %clang@ibm.9.0.0 ${PROJECT_LASSEN_DEPS}"
    extends: .job_on_lassen
    allow_failure: true

.. important:: When a shared job needs to be modified for RAJA specifically, we
   call that "overriding". The job label must be kept the same as for the 
   shared job in the ``gitlab/radiuss-jobs/<MACHINE>.yml`` file in the 
   `RADIUSS Spack Configs <https://github.com/LLNL/radiuss-spack-confgs>`_,
   and the RAJA-specific job can be adapted. If you override a shared job,
   please add a comment to describe the change in the
   ``RAJA/.gitlab/jobs/<MACHINE>.yml`` file where the job is overridden.


Building the Intel clang + SYCL HIP compiler for use in CI
----------------------------------------------------------

To run CI tests for the RAJA SYCL back-end on GitLab, we use the corona 
system and a custom Intel Clang compiler that we build ourselves to support
SYCL for AMD GPUs. This compiler lives in the ``/usr/workspace/raja-dev/``
folder so that it can be accessed by the service user account that we use to
run our GitLab CI. Since the Intel compiler does not
do releases in the typical sense (they simply update their repo *every night*), 
it may become necessary to periodically build a new version of the compiler to 
ensure that we are using the most up-to-date version available. The steps for 
building, installing, and running are shown here.

Building the Compiler
^^^^^^^^^^^^^^^^^^^^^

.. important:: Because Intel updates their compiler repo daily, it is possible
   that the head of the SYCL branch will fail to build. In the event that it
   does not build, try checking out an earlier commit. On the Intel/LLVM GitHub
   page, one can see which of their commits builds by checking the status
   badge next to each commit. Look for a commit that passes. 

#. Load the module of the version of GCC headers that you want to use. For example, we typically use the system default, which on corona is gcc/10.3.1-magic::

    module load gcc/10.3.1-magic

#. Load the module of the version of ROCm that you want to use. For example::

    module load rocm/5.7.1 

#. Clone the SYCL branch of Intel's LLVM compiler::

    git clone https://github.com/intel/llvm -b sycl

#. cd into the LLVM folder:: 
    
    cd llvm

   In the event that the head of the sycl branch does not build, run
   ``git checkout <git sha>`` to checkout a version that does build.

#. Build the compiler. 

   Note that in this example, we are using rocm5.7.1, but one can change the
   version they wish to use simply by changing the paths in the configure step

   a. Configure

     .. code-block:: bash 

        srun -n1 /usr/bin/python3 buildbot/configure.py --hip -o buildrocm5.7.1 \
        --cmake-gen "Unix Makefiles" \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/rocm-5.7.1 \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_INCLUDE_DIR=/opt/rocm-5.7.1/include \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_LIB_DIR=/opt/rocm-5.7.1/lib \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_INCLUDE_DIR=/opt/rocm-5.7.1/include \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR=/opt/rocm-5.7.1/hsa/include/hsa \
        --cmake-opt=-DSYCL_BUILD_PI_HIP_LIB_DIR=/opt/rocm-5.7.1/lib \
        --cmake-opt=-DUR_HIP_ROCM_DIR=/opt/rocm-5.7.1 \
        --cmake-opt=-DUR_HIP_INCLUDE_DIR=/opt/rocm-5.7.1/include \
        --cmake-opt=-DUR_HIP_HSA_INCLUDE_DIR=/opt/rocm-5.7.1/hsa/include/hsa \
        --cmake-opt=-DUR_HIP_LIB_DIR=/opt/rocm-5.7.1/lib

   b. Build

     .. code-block:: bash

      srun -n1 /usr/bin/python3 buildbot/compile.py -o buildrocm5.7.1

#. Test the compiler

   Follow the steps in the `Using the compiler`_ section to test the installation

#. Install

  a. The build step will install the compiler in the folder ``buildrocm<version>/install``. Copy this folder to the ``/usr/workspace/raja-dev/`` directory using the naming scheme ``clang_sycl_<git sha>_hip_gcc<version>_rocm<version>``

  #. Set the permissions of the folder, and everything in it to 750::

      chmod 750 /usr/workspace/raja-dev/<foldername>/ -R  

  #. Change the group of the folder and everything in it to raja-dev::

      chgrp raja-dev /usr/workspace/raja-dev/<foldername>/ -R  


Using the compiler
^^^^^^^^^^^^^^^^^^

#. Load the version of rocm that you used when building the compiler, for example::

    module load rocm/5.7.1

#. Navigate to the root of your local RAJA checkout space::

    cd /path/to/raja

#. Run the test config script::

    ./scripts/lc-builds/corona_sycl.sh /usr/workspace/raja-dev/clang_sycl_2f03ef85fee5_hip_gcc10.3.1_rocm5.7.1

   Note that at the time of writing, the newest compiler we had built was at ``clang_sycl_2f03ef85fee5_hip_gcc10.3.1_rocm5.7.1``

#. cd into the generated build directory::

    cd {build directory}

#. Build the code and run the RAJA tests::

    make -j
    make test


========================
GitHub Actions CI Tasks
========================

The tasks in this section apply to RAJA GitHub Actions CI testing that was 
described in :ref:`github_actions_ci-label`

Changing Builds/Container Images
--------------------------------

The builds we run in GitHub Actions are defined in the 
`RAJA/.github/workflows/build.yml <https://github.com/LLNL/RAJA/blob/develop/.github/workflows/build.yml>`_ file.
  
Linux/Docker
^^^^^^^^^^^^

To update or add a new compiler / job to GitHub Actions CI, we need to edit 
the ``RAJA/.github/workflows/build.yml`` file and the ``RAJA/Dockerfile``, if
changes are needed there.

For GitHub Actions, we add the name of the job to the job list in the
``RAJA/.github/workflows/build.yml`` file::

  jobs:
  build_docker:
    strategy:
      matrix:
        target: [..., compilerX]

In the ``RAJA/Dockerfile`` file, we add a section that defines the commands for the ``compilerX`` job, such as::

  FROM ghcr.io/llnl/radiuss:compilerX-ubuntu-22.04 AS compilerX
  ENV GTEST_COLOR=1
  COPY . /home/raja/workspace
  WORKDIR /home/raja/workspace/build
  RUN cmake -DCMAKE_CXX_COMPILER=compilerX ... && \
      make -j 6 && \
      ctest -T test --output-on-failure && \
      make clean

Each of our docker builds is built up on a base image maintained in the
`RADIUSS Docker Project <https://github.com/LLNL/radiuss-docker>`_.

The base container images are shared by multiple projects and are rebuilt
regularly. If bugs are fixed in the base images, the changes will be
automatically propagated to all projects using them in their Docker builds.

Check `RADIUSS Docker Project <https://github.com/LLNL/radiuss-docker>`_ for a
list of currently available images.

Windows / MacOS
^^^^^^^^^^^^^^^

In GitHub Actions, we run our Windows and MacOS builds directly on the 
provided machine instances. To change the versions, change the appropriate 
lines in the ``RAJA/.github/workflows/build.yml`` file::

  build_mac:
    runs-on: macos-latest
  ...

  build_windows:
    runs-on: windows-latest
  ...
   

Changing Build/Run Parameters
-----------------------------

Linux/Docker
^^^^^^^^^^^^

We can edit the build and run configurations of each Docker build, by editing 
the appropriate line containing the ``RUN`` command in the ``RAJA/Dockerfile``
file. For example, we can change CMake options or change the parallel build
value of ``make -j N`` for adjusting throughput.

Each base image is built using `spack <https://github.com/spack/spack>`_.
For the most part the container environments are set up to run our CMake and
build commands out of the box. However, there are a few exceptions where we
may need to load compiler specific environment variables, such as for
the Intel LLVM compiler. For example, this may appear as::

  RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake ..."

In these cases, it is important to include the double quotes in the correct
locations.

Windows / MacOS
^^^^^^^^^^^^^^^

Windows and MacOS build / run parameters can be configured directly in the
``RAJA/.github/workflows/build.yml`` file. CMake options can be configured 
in the workflow file for each job. The parallel build value can also be 
edited directly in the workflow file for each job.

.. _rajaperf_ci_tasks-label:

===============================
RAJA Performance Suite CI Tasks
===============================

The `RAJA Performance Suite <https://github.com/LLNL/RAJAPerf>`_ project CI
testing processes, directory/file structure, and dependencies are nearly
identical to that for RAJA, which is described in :ref:`ci-label`.
Specifically,

  * The RAJA Performance Suite GitLab CI process is driven by the
    `RAJAPerf/.gitlab-ci.yml
    <https://github.com/LLNL/RAJAPerf/blob/develop/.gitlab-ci.yml>`_ file.
  * The ``custom-jobs-and-variables.yml`` and ``subscribed-pipelines.yml``
    files reside in the `RAJAPerf/.gitlab
    <https://github.com/LLNL/RAJAPerf/tree/develop/.gitlab>`_ directory.
  * The ``build_and_test.sh`` script resides in the `RAJAPerf/scripts/gitlab
    <https://github.com/LLNL/RAJAPerf/tree/develop/scripts/gitlab>`_ directory.
  * The `RAJAPerf/Dockerfile
    <https://github.com/LLNL/RAJAPerf/blob/develop/Dockerfile>`_ drives the
    GitHub Actions testing pipelines.

The Performance Suite GitLab CI uses the ``uberenv`` and
``radiuss-spack-configs`` versions located in the RAJA submodule to make the
testing consistent across projects and avoid redundancy. This is reflected in
the `RAJAPerf/.uberenv_config.json
<https://github.com/LLNL/RAJAPerf/blob/develop/.uberenv_config.json>`_ file
which point at the relevant RAJA submodule locations. That is the paths contain
``tpl/RAJA/...``.

Apart from these minor differences, all CI maintenance and development tasks for
the RAJA Performance Suite follow the same pattern that is described in 
:ref:`ci_tasks-label`.
