# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import re
import shutil
import socket
from textwrap import dedent

from spack_repo.builtin.build_systems.cached_cmake import (
    CachedCMakePackage,
    cmake_cache_option,
    cmake_cache_path,
    cmake_cache_string,
)
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage
from spack_repo.builtin.packages.blt.package import llnl_link_helpers

from spack.package import *


# Starting with 2022.03.0, the only submodule we want to fetch is tpl/desul
# since there is no package for it. Other RAJA submodules are defined as
# dependencies.
def submodules(package):
    submodules = []
    submodules.append("tpl/desul")
    return submodules


class Raja(CachedCMakePackage, CudaPackage, ROCmPackage):
    """RAJA Parallel Framework."""

    homepage = "https://github.com/LLNL/RAJA"
    git = "https://github.com/LLNL/RAJA.git"
    tags = ["radiuss", "e4s"]

    maintainers("adrienbernede", "davidbeckingsale", "kab163")

    license("BSD-3-Clause")

    version("develop", branch="develop", submodules=submodules)
    version("main", branch="main", submodules=submodules)
    version(
        "2025.12.2",
        tag="v2025.12.2",
        commit="eca7c5015a5cf8bf7cc8ad1829fd36d3276ab274",
        submodules=submodules,
    )
    version(
        "2025.12.1",
        tag="v2025.12.1",
        commit="3b8b59a1e9be2e1066c0d77372b3bf5956e6d6e2",
        submodules=submodules,
    )
    version(
        "2025.12.0",
        tag="v2025.12.0",
        commit="e827035c630e71a9358e2f21c2f3cf6fd5fb6605",
        submodules=submodules,
    )
    version(
        "2025.09.1",
        tag="v2025.09.1",
        commit="1e0756eda3c344da362e483afb9100ebd8137a2c",
        submodules=submodules,
    )
    version(
        "2025.09.0",
        tag="v2025.09.0",
        commit="ca756788dbdd43fec2a3840389126ae94a905d5f",
        submodules=submodules,
    )
    version(
        "2025.03.2",
        tag="v2025.03.2",
        commit="6e36a94380adbe88fed11a3213fc08461428ece0",
        submodules=submodules,
    )
    version(
        "2025.03.1",
        tag="v2025.03.1",
        commit="ffa7b92377705aff855b4bf602e197ae4f8e8cc3",
        submodules=submodules,
    )
    version(
        "2025.03.0",
        tag="v2025.03.0",
        commit="1d70abf171474d331f1409908bdf1b1c3fe19222",
        submodules=submodules,
    )
    version(
        "2024.07.0",
        tag="v2024.07.0",
        commit="4d7fcba55ebc7cb972b7cc9f6778b48e43792ea1",
        submodules=submodules,
    )
    version(
        "2024.02.2",
        tag="v2024.02.2",
        commit="593f756b14ac57ded33ee61d8d2292d4beb840e6",
        submodules=submodules,
    )
    version(
        "2024.02.1",
        tag="v2024.02.1",
        commit="3ada0950b0774ec907d30a9eceaf6af7478b833b",
        submodules=submodules,
    )
    version(
        "2024.02.0",
        tag="v2024.02.0",
        commit="82d1b926ada0fbb15a4a6e0adadc30c715cfda7b",
        submodules=submodules,
    )
    version(
        "2023.06.1",
        tag="v2023.06.1",
        commit="9b5f61edf3aa1e6fdbc9a4b30828c81504639963",
        submodules=submodules,
    )
    version(
        "2023.06.0",
        tag="v2023.06.0",
        commit="e330b2560747d5417cd7bd265fab3fb91d32ecbd",
        submodules=submodules,
    )
    version(
        "2022.10.5",
        tag="v2022.10.5",
        commit="3774f51339459bbbdb77055aa23f82919b6335b6",
        submodules=submodules,
    )
    version(
        "2022.10.4",
        tag="v2022.10.4",
        commit="c2a6b1740759ae3ae7c85b35e20dbffbe235355d",
        submodules=submodules,
    )
    version(
        "2022.03.0",
        tag="v2022.03.0",
        commit="4351fe6a50bd579511a625b017c9e054885e7fd2",
        submodules=submodules,
    )
    version(
        "0.14.0", tag="v0.14.0", commit="357933a42842dd91de5c1034204d937fce0a2a44", submodules=True
    )
    version(
        "0.13.0", tag="v0.13.0", commit="3047fa720132d19ee143b1fcdacaa72971f5988c", submodules=True
    )
    version(
        "0.12.1", tag="v0.12.1", commit="9cb6370bb2868e35ebba23cdce927f5f7f9da530", submodules=True
    )
    version(
        "0.12.0", tag="v0.12.0", commit="32d92e38da41cc8d4db25ec79b9884a73a0cb3a1", submodules=True
    )
    version(
        "0.11.0", tag="v0.11.0", commit="0502b9b69c4cb60aa0afbdf699b555c76cb18f22", submodules=True
    )
    version(
        "0.10.1", tag="v0.10.1", commit="be91e040130678b1350dbda56cc352433db758bd", submodules=True
    )
    version(
        "0.10.0", tag="v0.10.0", commit="53cb89cf788d28bc4ed2b4e6f75483fdd26024aa", submodules=True
    )
    version(
        "0.9.0", tag="v0.9.0", commit="df7ca1fa892b6ac4147c614d2d739d5022f63fc7", submodules=True
    )
    version(
        "0.8.0", tag="v0.8.0", commit="8d19a8c2cbac611de6f92ad8852b9f3454b27e63", submodules=True
    )
    version(
        "0.7.0", tag="v0.7.0", commit="caa33b371b586dfae3d8569caee91c5eddfd7b31", submodules=True
    )
    version(
        "0.6.0", tag="v0.6.0", commit="cc7a97e8b4e52c3de820c9dfacd358822a147871", submodules=True
    )
    version(
        "0.5.3", tag="v0.5.3", commit="1ca35c0ed2a43a3fa9c6cd70c5d25f16d88ecd8c", submodules=True
    )
    version(
        "0.5.2", tag="v0.5.2", commit="4d5c3d5d7f311838855f7010810610349e729f64", submodules=True
    )
    version(
        "0.5.1", tag="v0.5.1", commit="bf340abe5199d7e051520913c9a7a5de336b5820", submodules=True
    )
    version(
        "0.5.0", tag="v0.5.0", commit="9b539d84fdad049f65caeba836f41031f5baf4cc", submodules=True
    )
    version(
        "0.4.1", tag="v0.4.1", commit="3618cfe95d6a442fa50fbe7bfbcf654cf9f800b9", submodules=True
    )
    version(
        "0.4.0", tag="v0.4.0", commit="31b2a48192542c2da426885baa5af0ed57606b78", submodules=True
    )

    # export targets when building pre-2.4.0 release with BLT 0.4.0+
    patch(
        "https://github.com/LLNL/RAJA/commit/eca1124ee4af380d6613adc6012c307d1fd4176b.patch?full_index=1",
        sha256="12bb78c00b6683ad3e7fd4e3f87f9776bae074b722431b79696bc862816735ef",
        when="@:0.13.0 ^blt@0.4:",
    )

    # Backward compatibility is stopped from ROCm 6.0
    # Future relase will have the change from PR https://github.com/LLNL/RAJA/pull/1568
    patch(
        "https://github.com/LLNL/RAJA/commit/406eb8dee05a41eb32c421c375688a4863b60642.patch?full_index=1",
        sha256="d9ce5ef038555cbccb330a9016b7be77e56ae0660583cba955dab9d0297a4b07",
        when="^hip@6.0",
    )

    # Fix compilation issue reported by Intel from their new compiler version
    patch(
        "https://github.com/LLNL/RAJA/commit/3e831e034bd92daacf49f40b66459aefd6ea3972.patch?full_index=1",
        sha256="c0548fc5220f24082fb2592d5b4e8b7c8c783b87906d5f0950d53953d25161f6",
        when="@2024.02.1:2024.02.99 %oneapi@2025:",
    )

    patch("tile-iterator-comparison-fix-2024.02.patch", when="@2024.02.0:2024.02.2")

    variant("openmp", default=False, description="Build OpenMP backend")
    variant("shared", default=False, description="Build shared libs")
    variant("desul", default=False, description="Build desul atomics backend")
    variant("vectorization", default=True, description="Build SIMD/SIMT intrinsics support")
    variant(
        "omptask", default=False, description="Build OpenMP task variants of internal algorithms"
    )
    variant("omptarget", default=False, description="Build OpenMP on target device support")
    variant("sycl", default=False, description="Build sycl backend")
    variant("gpu-profiling", default=False, description="Enable GPU profiling")

    variant("plugins", default=False, description="Enable runtime plugins")
    variant("caliper", default=False, description="Enable caliper support")
    variant("examples", default=True, description="Build examples.")
    variant("exercises", default=True, description="Build exercises.")
    # TODO: figure out gtest dependency and then set this default True
    # and remove the +tests conflict below.
    variant("tests", default=False, description="Build tests")

    # we don't use variants to express the failing test, we only add a variant to
    # define whether we want to run all the tests (including those known to fail)
    # or only the passing ones.
    variant(
        "run-all-tests",
        default=False,
        description="Run all the tests, including those known to fail.",
    )

    variant(
        "lowopttest",
        default=False,
        description="For developers, lowers optimization level to pass tests with some compilers",
    )

    variant(
        "cxxstd",
        default="20",
        values=("11", "14", "17", "20"),
        description="C++ standard to build with",
    )
    conflicts("cxxstd=11", when="@0.14.0:")
    conflicts("cxxstd=14", when="@2025.09.0:")
    conflicts("cxxstd=17", when="@2026.03.0:")
    conflicts("+sycl cxxstd=14", when="@2024.07.0:")

    depends_on("cxx", type="build")
    depends_on("c", type="build")

    depends_on("blt", type="build")
    # TODO(smith84): Edit the following line after the June 2026 RAJA suite release
    depends_on("blt@0.7.2:", type="build", when="@develop")
    depends_on("blt@0.7.1:", type="build", when="@2025.09.0:")
    depends_on("blt@0.7.0:", type="build", when="@2025.03.0:")
    depends_on("blt@0.6.2:", type="build", when="@2024.02.1:")
    depends_on("blt@0.6.1", type="build", when="@2024.02.0")
    depends_on("blt@0.5.3", type="build", when="@2023.06.0:2023.06.1")
    depends_on("blt@0.5.2:0.5.3", type="build", when="@2022.10.5")
    depends_on("blt@0.5.0:0.5.3", type="build", when="@0.14.1:2022.10.4")
    depends_on("blt@0.4.1", type="build", when="@0.14.0")
    depends_on("blt@0.4.0:0.4.1", type="build", when="@0.13.0")
    depends_on("blt@0.3.6:0.4.1", type="build", when="@:0.12.0")
    conflicts("^blt@:0.3.6", when="+rocm")
    conflicts("^blt@:0.7.1", when="+cuda ^cuda@13:", msg="CUDA 13+ requires BLT 0.7.2 or newer")

    depends_on("camp")
    depends_on("camp+openmp", when="+openmp")
    depends_on("camp+omptarget", when="+omptarget")
    depends_on("camp+sycl", when="+sycl")
    # TODO(johnbowen42): Remove the following line after the June 2026 RAJA suite release
    depends_on("camp@2026.07", when="@develop")
    depends_on("camp@2025.12", when="@2025.12.0:2025.12.2")
    depends_on("camp@2025.09", when="@2025.09")
    depends_on("camp@2025.03", when="@2025.03")
    depends_on("camp@2024.07", when="@2024.07")
    depends_on("camp@2024.02.1", when="@2024.02.1")
    depends_on("camp@2024.02.0", when="@2024.02.0")
    depends_on("camp@2023.06.0", when="@2023.06.0:2023.06.1")
    depends_on("camp@2022.10.1:2023.06.0", when="@2022.10.3:2022.10.5")
    depends_on("camp@2022.10.0:2023.06.0", when="@2022.10.0:2022.10.2")
    depends_on("camp@2022.03.2", when="@2022.03.0:2022.03.1")
    depends_on("camp@0.2.2:0.2.3", when="@0.14.0")
    depends_on("camp@0.1.0", when="@0.10.0:0.13.0")

    depends_on("cmake@3.24:", when="@2025.09.0:", type="build")
    depends_on("cmake@3.23:", when="@2024.07.0:2025.03.2", type="build")
    depends_on("cmake@3.23:", when="@2022.10.0:2024.02.2+rocm", type="build")
    depends_on("cmake@3.20:", when="@2022.10.0:2024.02.2", type="build")
    depends_on("cmake@3.20:", when="@:2022.03+rocm", type="build")
    depends_on("cmake@3.14:", when="@:2022.03", type="build")

    depends_on("llvm-openmp", when="+openmp %apple-clang")

    depends_on("caliper", when="+caliper")

    depends_on("rocprim", when="+rocm")
    with when("+rocm @0.12.0:"):
        depends_on("camp+rocm")
        for arch in ROCmPackage.amdgpu_targets:
            depends_on(
                "camp+rocm amdgpu_target={0}".format(arch), when="amdgpu_target={0}".format(arch)
            )
        conflicts("+openmp", when="@:2022.03")

    with when("+cuda @0.12.0:"):
        depends_on("camp+cuda")
        for sm_ in CudaPackage.cuda_arch_values:
            depends_on("camp +cuda cuda_arch={0}".format(sm_), when="cuda_arch={0}".format(sm_))

    conflicts("+gpu-profiling", when="~cuda~rocm", msg="GPU profiling requires CUDA or ROCm")
    conflicts("+gpu-profiling +cuda", when="@:2022.02.99")
    conflicts("+gpu-profiling +rocm", when="@:2022.02.99")

    conflicts("+omptarget +rocm")
    conflicts("+sycl +omptarget")
    conflicts("+sycl +rocm")
    conflicts(
        "+sycl",
        when="@:2024.02.99",
        msg="Support for SYCL was introduced in RAJA after 2024.02 release, "
        "please use a newer release.",
    )

    depends_on("cuda@12:", when="+cuda")
    conflicts(
        "^cuda@13:",
        when="@:2025.12.2 +cuda",
        msg="RAJA versions up to and including 2025.12.2 do not support CUDA 13+",
    )

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    @property
    def libs(self):
        shared = "+shared" in self.spec
        return find_libraries("libRAJA", root=self.prefix, shared=shared, recursive=True)

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            hostname = hostname.rstrip("1234567890")
        return "{0}-{1}-{2}@{3}-{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            self.spec.dag_hash(8),
        )

    def initconfig_compiler_entries(self):
        spec = self.spec
        compiler = self.compiler
        # Default entries are already defined in CachedCMakePackage, inherit them:
        entries = super().initconfig_compiler_entries()

        if spec.satisfies("+rocm ^blt@:0.6"):
            entries.insert(0, cmake_cache_path("CMAKE_CXX_COMPILER", spec["hip"].hipcc))

        llnl_link_helpers(entries, spec, compiler)

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super().initconfig_hardware_entries()

        entries.append("#------------------{0}".format("-" * 30))
        entries.append("# Package custom hardware settings")
        entries.append("#------------------{0}\n".format("-" * 30))

        entries.append(cmake_cache_option("ENABLE_OPENMP", spec.satisfies("+openmp")))
        entries.append(cmake_cache_option("ENABLE_CUDA", spec.satisfies("+cuda")))

        if spec.satisfies("+cuda"):
            # CUDA configuration from cuda_for_radiuss_projects
            cuda_flags = []
            if not spec.satisfies("cuda_arch=none"):
                cuda_archs = ";".join(spec.variants["cuda_arch"].value)
                entries.append(cmake_cache_string("CMAKE_CUDA_ARCHITECTURES", cuda_archs))

            # gcc-toolchain support
            gcc_toolchain_regex = re.compile(".*gcc-toolchain.*")
            using_toolchain = list(
                filter(gcc_toolchain_regex.match, spec.compiler_flags["cxxflags"])
            )
            if using_toolchain:
                cuda_flags.append("-Xcompiler {}".format(using_toolchain[0]))

            if cuda_flags:
                entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS", " ".join(cuda_flags)))

        if spec.satisfies("+rocm"):
            entries.append(cmake_cache_option("ENABLE_HIP", True))

            # HIP configuration from hip_for_radiuss_projects
            rocm_root = spec["llvm-amdgpu"].prefix
            gcc_toolchain_regex = re.compile(".*gcc-toolchain.*")
            using_toolchain = list(
                filter(gcc_toolchain_regex.match, spec.compiler_flags["cxxflags"])
            )
            hip_link_flags = ""

            if using_toolchain:
                gcc_prefix = using_toolchain[0]
                entries.append(
                    cmake_cache_string("HIP_CLANG_FLAGS", "--gcc-toolchain={0}".format(gcc_prefix))
                )
                entries.append(
                    cmake_cache_string(
                        "CMAKE_EXE_LINKER_FLAGS",
                        hip_link_flags + " -Wl,-rpath={0}/lib64".format(gcc_prefix),
                    )
                )
            else:
                entries.append(
                    cmake_cache_string(
                        "CMAKE_EXE_LINKER_FLAGS", "-Wl,-rpath={0}/llvm/lib/".format(rocm_root)
                    )
                )

            hipcc_flags = []
            if self.spec.satisfies("^rocprim@7.0"):
                hipcc_flags.append("-std=c++17")
            if self.spec.satisfies("@2025.09.0:"):
                hipcc_flags.append("-std=c++17")
            elif self.spec.satisfies("@0.14.0:2025.09.0"):
                hipcc_flags.append("-std=c++14")
            entries.append(cmake_cache_string("HIP_HIPCC_FLAGS", " ".join(hipcc_flags)))
        else:
            entries.append(cmake_cache_option("ENABLE_HIP", False))

        return entries

    @property
    def cxx_std(self):
        return self.spec.variants.get("cxxstd").value

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        option_prefix = "RAJA_" if spec.satisfies("@0.14.0:") else ""

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        entries.append(cmake_cache_path("BLT_SOURCE_DIR", spec["blt"].prefix))
        if "camp" in self.spec:
            entries.append(cmake_cache_path("camp_DIR", spec["camp"].prefix))

        # Build options
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Build Options")
        entries.append("#------------------{0}\n".format("-" * 60))

        entries.append(cmake_cache_string("CMAKE_BUILD_TYPE", spec.variants["build_type"].value))
        entries.append(cmake_cache_option("BUILD_SHARED_LIBS", spec.satisfies("+shared")))

        entries.append(cmake_cache_option("RAJA_ENABLE_DESUL_ATOMICS", spec.satisfies("+desul")))

        entries.append(
            cmake_cache_option("RAJA_ENABLE_VECTORIZATION", spec.satisfies("+vectorization"))
        )

        entries.append(cmake_cache_option("RAJA_ENABLE_OPENMP_TASK", spec.satisfies("+omptask")))

        entries.append(
            cmake_cache_option("RAJA_ENABLE_TARGET_OPENMP", spec.satisfies("+omptarget"))
        )

        entries.append(cmake_cache_option("RAJA_ENABLE_SYCL", spec.satisfies("+sycl")))
        entries.append(
            cmake_cache_option("RAJA_ENABLE_NV_TOOLS_EXT", spec.satisfies("+gpu-profiling +cuda"))
        )
        entries.append(
            cmake_cache_option("RAJA_ENABLE_ROCTX", spec.satisfies("+gpu-profiling +rocm"))
        )

        if spec.satisfies("+lowopttest"):
            entries.append(cmake_cache_string("CMAKE_CXX_FLAGS_RELEASE", "-O1"))

        # C++ standard
        entries.append(cmake_cache_string("BLT_CXX_STD", f"c++{self.cxx_std}"))

        entries.append(
            cmake_cache_option("RAJA_ENABLE_RUNTIME_PLUGINS", spec.satisfies("+plugins"))
        )

        if spec.satisfies("+omptarget"):
            entries.append(
                cmake_cache_string(
                    "BLT_OPENMP_COMPILE_FLAGS", "-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda"
                )
            )
            entries.append(
                cmake_cache_string(
                    "BLT_OPENMP_LINK_FLAGS", "-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda"
                )
            )

        entries.append(
            cmake_cache_option(
                "{}ENABLE_EXAMPLES".format(option_prefix), spec.satisfies("+examples")
            )
        )
        if spec.satisfies("@0.14.0:"):
            entries.append(
                cmake_cache_option(
                    "{}ENABLE_EXERCISES".format(option_prefix), spec.satisfies("+exercises")
                )
            )
        else:
            entries.append(cmake_cache_option("ENABLE_EXERCISES", spec.satisfies("+exercises")))

        # TODO: Treat the workaround when building tests with spack wrapper
        #       For now, removing it to test CI, which builds tests outside of wrapper.
        # Work around spack adding -march=ppc64le to SPACK_TARGET_ARGS which
        # is used by the spack compiler wrapper.  This can go away when BLT
        # removes -Werror from GTest flags
        #
        # if self.spec.satisfies("%clang target=ppc64le:")
        #   or (not self.run_tests and not spec.satisfies("+tests")):
        if not self.run_tests and not spec.satisfies("+tests"):
            entries.append(cmake_cache_option("ENABLE_TESTS", False))
        else:
            entries.append(cmake_cache_option("ENABLE_TESTS", True))
            if not spec.satisfies("+run-all-tests"):
                if spec.satisfies("%clang@12.0.0:13.9.999"):
                    entries.append(
                        cmake_cache_string(
                            "CTEST_CUSTOM_TESTS_IGNORE",
                            "test-algorithm-sort-OpenMP.exe;test-algorithm-stable-sort-OpenMP.exe",
                        )
                    )
                excluded_tests = [
                    "test-algorithm-sort-Cuda.exe",
                    "test-algorithm-stable-sort-Cuda.exe",
                    "test-algorithm-sort-OpenMP.exe",
                    "test-algorithm-stable-sort-OpenMP.exe",
                ]
                if spec.satisfies("+cuda %clang@12.0.0:13.9.999"):
                    entries.append(
                        cmake_cache_string("CTEST_CUSTOM_TESTS_IGNORE", ";".join(excluded_tests))
                    )
                if spec.satisfies("+cuda %xl@16.1.1.12"):
                    entries.append(
                        cmake_cache_string(
                            "CTEST_CUSTOM_TESTS_IGNORE",
                            "test-algorithm-sort-Cuda.exe;test-algorithm-stable-sort-Cuda.exe",
                        )
                    )

        entries.append(cmake_cache_option("RAJA_HOST_CONFIG_LOADED", True))

        return entries

    def cmake_args(self):
        return []

    @run_after("build")
    @on_package_attributes(run_tests=True)
    def check_build(self):
        """Run RAJA's unit test target after build when tests are enabled."""
        with working_dir(self.build_directory):
            print("Running RAJA Unit Tests...")
            make("test")

    examples_src_dir = "examples"
    using_with_cmake_dir = join_path("examples", "using-with-cmake")

    def _rewrite_host_config(self, path):
        """Replace compiler wrappers in cached install-test files."""
        kwargs = {"backup": False, "ignore_absent": True}
        compiler_paths = {
            "CMAKE_C_COMPILER": getattr(self.compiler, "cc", None),
            "CMAKE_CXX_COMPILER": getattr(self.compiler, "cxx", None),
            "CMAKE_Fortran_COMPILER": getattr(self.compiler, "fc", None),
            "CMAKE_CUDA_HOST_COMPILER": getattr(self.compiler, "cxx", None),
        }

        for key, value in compiler_paths.items():
            if value:
                filter_file(
                    rf"set\({key}.*\)", f'set({key} "{value}" CACHE PATH "")', path, **kwargs
                )

    @run_after("install")
    def setup_install_tests(self):
        """Install and cache standalone test sources, using staged or build outputs
        when available."""

        cache_extra_test_sources(self, [self.examples_src_dir])

        src_dir = join_path(self.stage.source_path, "test", "install", "using-with-cmake")
        dst_dir = join_path(install_test_root(self), self.using_with_cmake_dir)

        if os.path.exists(src_dir):
            shutil.rmtree(dst_dir, ignore_errors=True)
            install_tree(src_dir, dst_dir)
            self._rewrite_host_config(join_path(dst_dir, "host-config.cmake"))

        src_dir = join_path(self.build_directory, "examples", "using-with-cmake")
        dst_dir = join_path(install_test_root(self), self.using_with_cmake_dir)

        if os.path.exists(src_dir):
            install_tree(src_dir, dst_dir)
            self._rewrite_host_config(join_path(dst_dir, "host-config.cmake"))
        else:
            tty.msg("Can't install host-config.cmake\n")

    def _run_common_check_install(self, test_dir):
        """Verify that the using-with-cmake example can build against the installed
        RAJA package."""

        example_stage_dir = join_path(test_dir, "examples", "using-with-cmake")
        with working_dir(join_path(example_stage_dir, "build"), create=True):
            host_config = join_path("../", "host-config.cmake")
            if not os.path.exists(host_config):
                raise SkipTest(f"{os.path.abspath(host_config)} not found, cannot build example")
            cmake_args = ["-C", host_config, "../"]
            cmake = self.spec["cmake"].command
            make_exe = which("make", required=True)
            cmake(*cmake_args)
            make_exe()
            example = Executable("./using-with-cmake")
            example()
            make_exe("clean")

    @run_after("install")
    @on_package_attributes(run_tests=True)
    def check_install(self):
        """Installation-time verification that the using-with-cmake example can build against the
        installed RAJA package."""

        src_dir = join_path(install_test_root(self))
        dst_dir = join_path(self.stage.path, "spack-test")

        if os.path.exists(src_dir):
            install_tree(src_dir, dst_dir)
            self._run_common_check_install(dst_dir)
        else:
            raise SkipTest("examples directory not found, cannot build example")

    def test_check_install(self):
        """Stand-alone verification that the using-with-cmake example can build against the
        installed RAJA package."""

        self._run_common_check_install(self.test_suite.current_test_cache_dir)

    def _write_example_cmakelists(self, path, exe, source):
        cmake_contents = dedent(f"""\
        cmake_minimum_required(VERSION 3.23)
        project(raja_package_test LANGUAGES CXX)

        if(NOT DEFINED RAJA_DIR OR NOT EXISTS
           ${{RAJA_DIR}}/lib/cmake/raja/raja-config.cmake)
          message(FATAL_ERROR "Missing required 'RAJA_DIR' variable pointing to
          an installed RAJA")
        endif()

        find_package(RAJA REQUIRED
                     NO_DEFAULT_PATH
                     PATHS ${{RAJA_DIR}}/lib/cmake/raja)

        add_executable({exe} ../{source})
        target_link_libraries({exe} RAJA)
        """)

        with open(path, "w", encoding="utf-8") as f:
            f.write(cmake_contents)

    def build_and_run_example(self, exe, expected):
        """Build an example from the cached test sources and verify its output."""

        examples_dir = join_path(self.test_suite.current_test_cache_dir, self.examples_src_dir)
        build_dir = join_path(examples_dir, f"build-{exe}")
        with working_dir(build_dir, create=True):
            cmake = self.spec["cmake"].command
            make_exe = which("make", required=True)
            host_config = join_path("../using-with-cmake", "host-config.cmake")
            if not os.path.exists(host_config):
                raise SkipTest("host-config.cmake not found, cannot build example")
            self._write_example_cmakelists("CMakeLists.txt", exe, f"{exe}.cpp")
            cmake_args = ["-C", host_config, "."]
            cmake(*cmake_args)
            make_exe()
            exe_path = join_path(".", exe)
            if not os.path.exists(exe_path):
                raise SkipTest(f"{exe} was not built")
            example = Executable(exe_path)
            out = example(output=str, error=str)
            check_outputs(expected, out)
            make_exe("clean")

    def test_daxpy(self):
        """Check daxpy tutorial"""
        self.build_and_run_example("tut_daxpy", [r"daxpy", r"result -- PASS"])

    # TODO: this test seems to hang or take a long time?
    # SGS 2026-05-22: Did not see hangs/long execution times on LC systems or Redhat workstation
    #                 clarify with Cody where this was occuring.
    # def test_matrix_multiply(self):
    #     """check batched matrix multiple tutorial"""
    #     self.build_and_run_example(
    #         "tut_matrix-multiply", [r"matrix multiplication", r"result -- PASS"]
    #     )

    def test_launch_basic(self):
        """Check basic raja::launch tutorial."""
        if "+cuda" in self.spec or "+rocm" in self.spec:
            self.build_and_run_example(
                "tut_launch_basic", [r"Running RAJA-Teams", r"result -- PASS"]
            )
        else:
            raise SkipTest("CUDA or ROCm support is required to run this example")

    def test_halo_exchange(self):
        """Check halo exchange tutorial."""
        self.build_and_run_example(
            "tut_halo-exchange", [r"RAJA halo exchange example", r"result -- PASS"]
        )

    def test_wave_equation(self):
        """Check wave equation."""
        self.build_and_run_example("wave-eqn", [r"Max Error = 2", r"Evolved solution to time"])
