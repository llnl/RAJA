#!/usr/bin/env python3

# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""Configure, build, install, and test RAJA from a host-config."""

import argparse
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RSC_ROOT = PROJECT_ROOT / "scripts" / "radiuss-spack-configs"
sys.path.insert(0, str(RSC_ROOT))

from rad_bat import logs
from rad_bat.runner import scheduler_pass_fds


class BuildError(Exception):
    pass


class CommandError(BuildError):
    def __init__(self, command, returncode):
        super().__init__("command failed with exit code {}: {}".format(returncode, " ".join(command)))
        self.returncode = returncode


def run(command, cwd, check=True):
    logs.command(command)
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            check=False,
            pass_fds=scheduler_pass_fds(),
        )
    except FileNotFoundError:
        raise BuildError("command not found: {}".format(command[0]))
    if check and result.returncode:
        raise CommandError(command, result.returncode)
    return result.returncode


def cmake_from_host_config(path):
    match = re.search(
        r"^# CMake executable path:\s*(.+?)\s*$",
        path.read_text(encoding="utf-8", errors="replace"),
        re.MULTILINE,
    )
    if not match:
        raise BuildError("CMake executable path is missing from {}".format(path))
    cmake = match.group(1)
    if not Path(cmake).is_file():
        raise BuildError("CMake executable from host-config does not exist: {}".format(cmake))
    return cmake


def enabled(path, name):
    text = path.read_text(encoding="utf-8", errors="replace")
    return re.search(r"{}[^\n]*\bON\b".format(re.escape(name)), text, re.IGNORECASE) is not None


def configure(cmake, source, build_dir, install, host_config, extra_arguments):
    command = [cmake, "-C", str(host_config)]
    if re.sub(r"\d", "", socket.gethostname()) in {"dane", "matrix"}:
        command.append("-DBLT_MPI_COMMAND_APPEND:STRING=--overlap")
    command.extend(extra_arguments)
    command.extend(["-DCMAKE_INSTALL_PREFIX={}".format(install), str(source)])
    try:
        run(command, build_dir)
    except CommandError as error:
        logs.warning("CMake configuration failed; rerunning with trace output.")
        run(command + ["--debug-output", "--trace-expand"], build_dir, check=False)
        raise error


def build(cmake, build_dir, jobs):
    command = [cmake, "--build", ".", "--parallel", str(jobs)]
    try:
        run(command, build_dir)
    except CommandError as error:
        logs.warning("Compilation failed; rerunning serially with verbose output.")
        with logs.sections.section("build_verbose", "Verbose rebuild", collapsed=False):
            run([cmake, "--build", ".", "--verbose", "--parallel", "1"], build_dir, check=False)
        raise error


def run_ctest(build_dir):
    command = ["ctest", "--output-on-failure", "--no-compress-output", "-T", "test", "-VV"]
    regex = os.environ.get("CTEST_REGEX")
    parallel = os.environ.get("CTEST_JOBS")
    if regex:
        command.extend(["-R", regex])
    if parallel:
        command.extend(["-j", parallel])
    logs.command(command)
    try:
        process = subprocess.Popen(
            command,
            cwd=str(build_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=scheduler_pass_fds(),
            universal_newlines=True,
        )
    except FileNotFoundError:
        raise BuildError("command not found: ctest")
    lines = []
    with (build_dir / "tests_output.txt").open("w", encoding="utf-8") as output:
        for line in process.stdout:
            sys.stdout.write(line)
            output.write(line)
            lines.append(line)
    status = process.wait()
    text = "".join(lines)
    if "No tests were found!!!" in text:
        raise BuildError("CTest did not find any tests")
    if status or "Errors while running CTest" in text:
        raise CommandError(command, status or 1)


def write_junit(source, build_dir):
    results = sorted((build_dir / "Testing").glob("*/Test.xml"))
    if not results:
        raise BuildError("CTest did not produce Testing/*/Test.xml")
    stylesheet = source / "scripts/radiuss-spack-configs/utilities/ctest-to-junit.xsl"
    run(["xsltproc", "-o", str(source / "junit.xml"), str(stylesheet), str(results[-1])], build_dir)


def test_installed_example(cmake, install):
    example = install / "examples" / "RAJA" / "using-with-cmake"
    if not example.is_dir():
        raise BuildError("installed example was not found: {}".format(example))
    build_dir = example / "build"
    if build_dir.exists():
        shutil.rmtree(str(build_dir))
    build_dir.mkdir()
    run([cmake, "-C", str(example / "host-config.cmake"), str(example)], build_dir)
    run([cmake, "--build", "."], build_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--source", default=".")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--build-only", action="store_true")
    phase.add_argument("--test-only", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be greater than zero")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        source = Path(args.source).expanduser().resolve()
        work_dir = Path(args.work_dir).expanduser().resolve()
        host_config = Path(args.host_config).expanduser().resolve()
        if not source.is_dir() or not host_config.is_file():
            raise BuildError("source or host-config does not exist")
        work_dir.mkdir(parents=True, exist_ok=True)
        build_dir = work_dir / "build_{}".format(host_config.stem)
        install_dir = work_dir / "install_{}".format(host_config.stem)
        cmake = cmake_from_host_config(host_config)
        extra_arguments = shlex.split(os.environ.get("CMAKE_EXTRA_ARGS", ""))
        if not args.test_only:
            with logs.sections.section("clean", "Cleaning working directories"):
                for path in (build_dir, install_dir):
                    if path.exists():
                        shutil.rmtree(str(path))
                build_dir.mkdir(parents=True)
            with logs.sections.section("cmake_config", "CMake configuration"):
                configure(cmake, source, build_dir, install_dir, host_config, extra_arguments)
            with logs.sections.section("build", "Building RAJA"):
                build(cmake, build_dir, args.jobs)
            with logs.sections.section("install", "Installing RAJA"):
                run([cmake, "--install", "."], build_dir)
        if not args.build_only and enabled(host_config, "ENABLE_TESTS"):
            if not build_dir.is_dir():
                raise BuildError("build directory does not exist: {}".format(build_dir))
            with logs.sections.section("tests", "Running tests"):
                run_ctest(build_dir)
                write_junit(source, build_dir)
            with logs.sections.section("install_test", "Testing installed examples"):
                test_installed_example(cmake, install_dir)
        if not args.test_only and build_dir.is_dir():
            with logs.sections.section("cleanup", "Cleaning build directory"):
                run([cmake, "--build", ".", "--target", "clean"], build_dir)
        logs.info("Build and test completed")
        return 0
    except (BuildError, ValueError) as error:
        logs.error("raja-build-and-test: {}".format(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
