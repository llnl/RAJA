#!/usr/bin/env bash

# Initialize modules for users not using bash as a default shell
if test -e /usr/share/lmod/lmod/init/bash
then
  . /usr/share/lmod/lmod/init/bash
fi

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
truehostname=${hostname//[0-9]/}
project_dir="$(pwd)"

hostconfig=${HOST_CONFIG:-""}
spec=${SPEC:-""}
module_list=${MODULE_LIST:-""}
job_unique_id=${CI_JOB_ID:-""}
use_dev_shm=${USE_DEV_SHM:-true}
spack_debug=${SPACK_DEBUG:-false}
debug_mode=${DEBUG_MODE:-false}
push_to_registry=${PUSH_TO_REGISTRY:-true}

# REGISTRY_TOKEN allows you to provide your own personal access token to the CI
# registry. Be sure to set the token with at least read access to the registry.
registry_token=${REGISTRY_TOKEN:-""}
ci_registry_image=${CI_REGISTRY_IMAGE:-"czregistry.llnl.gov:5050/radiuss/raja"}
export ci_registry_user=${CI_REGISTRY_USER:-"${USER}"}
export ci_registry_token=${CI_JOB_TOKEN:-"${registry_token}"}

timed_message ()
{
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~ $(date --rfc-3339=seconds) ~ ${1}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
}

if [[ ${debug_mode} == true ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Debug mode:"
    echo "~~~~~ - Spack debug mode."
    echo "~~~~~ - Deactivated shared memory."
    echo "~~~~~ - Do not push to buildcache."
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    use_dev_shm=false
    spack_debug=true
    push_to_registry=false
fi

if [[ -n ${module_list} ]]
then
    timed_message "Modules to load: ${module_list}"
    module load ${module_list}
fi

prefix=""

if [[ -d /dev/shm && ${use_dev_shm} == true ]]
then
    prefix="/dev/shm/${hostname}"
    if [[ -z ${job_unique_id} ]]; then
      job_unique_id=manual_job_$(date +%s)
      while [[ -d ${prefix}-${job_unique_id} ]] ; do
          sleep 1
          job_unique_id=manual_job_$(date +%s)
      done
    fi

    prefix="${prefix}-${job_unique_id}"
else
    # We set the prefix in the parent directory so that spack dependencies are not installed inside the source tree.
    prefix="${project_dir}/../spack-and-build-root"
fi

echo "Creating directory ${prefix}"
echo "project_dir: ${project_dir}"

mkdir -p ${prefix}

spack_cmd="${prefix}/spack/bin/spack"
spack_env_path="${prefix}/spack_env"
uberenv_cmd="./scripts/uberenv/uberenv.py"
if [[ ${spack_debug} == true ]]
then
    spack_cmd="${spack_cmd} --debug --stacktrace"
    uberenv_cmd="${uberenv_cmd} --spack-debug"
fi

# Dependencies
if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    timed_message "Building dependencies"

    if [[ -z ${spec} ]]
    then
        echo "[Error]: SPEC is undefined, aborting..."
        exit 1
    fi

    prefix_opt="--prefix=${prefix}"

    # We force Spack to put all generated files (cache and configuration of
    # all sorts) in a unique location so that there can be no collision
    # with existing or concurrent Spack.
    spack_user_cache="${prefix}/spack-user-cache"
    export SPACK_DISABLE_LOCAL_CONFIG=""
    export SPACK_USER_CACHE_PATH="${spack_user_cache}"
    mkdir -p ${spack_user_cache}

    # generate cmake cache file with uberenv and radiuss spack package
    timed_message "Spack setup and environment"
    ${uberenv_cmd} --setup-and-env-only --spec="${spec}" ${prefix_opt}

    if [[ -n ${ci_registry_token} ]]
    then
        timed_message "GitLab registry as Spack Buildcache"
        ${spack_cmd} -D ${spack_env_path} mirror add --unsigned --oci-username-variable ci_registry_user --oci-password-variable ci_registry_token gitlab_ci oci://${ci_registry_image}
    fi

    timed_message "Spack build of dependencies"
    ${uberenv_cmd} --skip-setup-and-env --spec="${spec}" ${prefix_opt}

    if [[ -n ${ci_registry_token} && ${push_to_registry} == true ]]
    then
        timed_message "Push dependencies to buildcache"
        ${spack_cmd} -D ${spack_env_path} buildcache push --only dependencies gitlab_ci
    fi

    timed_message "Dependencies built"
fi

# Find cmake cache file (hostconfig)
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "[Error]: No result for: ${project_dir}/*.cmake"
        echo "[Error]: Spack generated host-config not found."
        exit 1
    else
        echo "[Error]: More than one result for: ${project_dir}/*.cmake"
        echo "[Error]: ${hostconfigs[@]}"
        echo "[Error]: Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/${hostconfig}"
fi

hostconfig=$(basename ${hostconfig_path})
echo "[Information]: Found hostconfig ${hostconfig_path}"

# Build Directory
# When using /dev/shm, we use prefix for both spack builds and source build, unless BUILD_ROOT was defined
build_root=${BUILD_ROOT:-"${prefix}"}

build_dir="${build_root}/build_${hostconfig//.cmake/}"
install_dir="${build_root}/install_${hostconfig//.cmake/}"

cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Prefix: ${prefix}"
    echo "~~~~~ Host-config: ${hostconfig_path}"
    echo "~~~~~ Build Dir:   ${build_dir}"
    echo "~~~~~ Project Dir: ${project_dir}"
    echo "~~~~~ Install Dir: ${install_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    timed_message "Cleaning working directory"

    # Map CPU core allocations
    declare -A core_counts=(["lassen"]=40 ["poodle"]=28 ["dane"]=28 ["matrix"]=28 ["corona"]=32 ["rzansel"]=48 ["tioga"]=32 ["tuolumne"]=48)

    # If building, then delete everything first
    # NOTE: 'cmake --build . -j core_counts' attempts to reduce individual build resources.
    #       If core_counts does not contain hostname, then will default to '-j ', which should
    #       use max cores.
    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    timed_message "Building RAJA"
    # We set the MPI tests command to allow overlapping.
    # Shared allocation: Allows build_and_test.sh to run within a sub-allocation (see CI config).
    # Use /dev/shm: Prevent MPI tests from running on a node where the build dir doesn't exist.
    cmake_options=""
    if [[ "${truehostname}" == "poodle" || "${truehostname}" == "dane" || "${truehostname}" == "matrix" ]]
    then
        cmake_options="-DBLT_MPI_COMMAND_APPEND:STRING=--overlap"
    fi

    $cmake_exe \
      -C ${hostconfig_path} \
      ${cmake_options} \
      -DCMAKE_INSTALL_PREFIX=${install_dir} \
      ${project_dir}
    if ! $cmake_exe --build . -j ${core_counts[$truehostname]}
    then
        echo "[Error]: Compilation failed, building with verbose output..."
        timed_message "Re-building with --verbose"
        $cmake_exe --build . --verbose -j 1
    else
        timed_message "Installing"
        $cmake_exe --install .
    fi

    timed_message "RAJA built and installed"
fi

# Test
if [[ "${option}" != "--build-only" ]] && grep -q -i "ENABLE_TESTS.*ON" ${hostconfig_path}
then

    if [[ ! -d ${build_dir} ]]
    then
        echo "[Error]: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    timed_message "Testing RAJA"
    ctest --output-on-failure --no-compress-output -T test -VV 2>&1 | tee tests_output.txt

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "[Error]: No tests were found" && exit 1
    fi

    timed_message "Preparing tests xml reports for export"
    tree Testing
    xsltproc -o junit.xml ${project_dir}/scripts/radiuss-spack-configs/utilities/ctest-to-junit.xsl Testing/*/Test.xml
    mv junit.xml ${project_dir}/junit.xml

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "[Error]: Failure(s) while running CTest" && exit 1
    fi

    if [[ ! -d ${install_dir} ]]
    then
        echo "[Error]: Install directory not found : ${install_dir}" && exit 1
    fi

    cd ${install_dir}/examples/RAJA/using-with-cmake
    mkdir build && cd build
    if ! $cmake_exe -C ../host-config.cmake ..; then
        echo "[Error]: Running $cmake_exe for using-with-cmake test" && exit 1
    fi

    if ! make; then
        echo "[Error]: Running make for using-with-cmake test" && exit 1
    fi

    timed_message "RAJA tests completed"
fi

timed_message "Cleaning up"
make clean

timed_message "Build and test completed"
