# `launch_nd` Benchmark Workflow

This workflow uses the updated `examples/launch_nd.cpp` benchmark driver to
compare the `RAJA::launch_nd` mappings across a size sweep using Caliper only.
Each kernel launch is labeled with `RAJA::Name`, so Caliper reports are grouped
by mapping and problem size.

## 1. Build Caliper

```bash
export REPO_DIR=$(pwd)
export CALIPER_SOURCE_DIR=/path/to/caliper
export CALIPER_INSTALL_PREFIX="${REPO_DIR}/../install-caliper"

mkdir -p "${REPO_DIR}/../build-caliper"
cd "${REPO_DIR}/../build-caliper"
/usr/tce/packages/cmake/cmake-3.29.2/bin/cmake "${CALIPER_SOURCE_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${CALIPER_INSTALL_PREFIX}"
/usr/tce/packages/cmake/cmake-3.29.2/bin/cmake --build . --target install -j
```

## 2. Build RAJA with Caliper support

```bash
cd "${REPO_DIR}"
export CALIPER_DIR="${CALIPER_INSTALL_PREFIX}/share/cmake/caliper"

mkdir -p build-raja-clang
cd build-raja-clang
/usr/tce/packages/cmake/cmake-3.29.2/bin/cmake "${REPO_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/cray/pe/craype/2.7.35/bin/cc \
  -DCMAKE_CXX_COMPILER=/opt/cray/pe/craype/2.7.35/bin/CC \
  -DRAJA_ENABLE_RUNTIME_PLUGINS=ON \
  -DRAJA_ENABLE_CALIPER=ON \
  -Dcaliper_DIR="${CALIPER_DIR}"
/usr/tce/packages/cmake/cmake-3.29.2/bin/cmake --build . --target launch_nd -j
```

The LC helper script now enforces the same Caliper-only configuration:

```bash
cd "${REPO_DIR}"
CALIPER_DIR="${CALIPER_DIR}" ./scripts/lc-builds/toss4_amdclang.sh 6.4.3 gfx90a
```

## 3. Generate a Caliper text report

```bash
cd "${REPO_DIR}/build-raja-clang"
RAJA_CALIPER=1 CALI_CONFIG=runtime-report ./bin/launch_nd \
  --mapping all \
  --sizes 65536x8,262144x8,1048576x8 \
  --warmup 5 \
  --repetitions 50
```

Look for kernels like:

- `launch_nd_flat_cells65536_comp8`
- `launch_nd_global_cells65536_comp8`
- `launch_nd_block_cells262144_comp8`
- `launch_nd_thread_local_cells1048576_comp8`

Those names come from `RAJA::Name` and let Caliper separate each policy/size
combination in a single run.

## 4. Generate a `.cali` profile for offline analysis

```bash
cd "${REPO_DIR}/build-raja-clang"
RAJA_CALIPER=1 CALI_CONFIG=runtime-profile(output=launch_nd.cali,output.format=cali) \
./bin/launch_nd \
  --mapping all \
  --sizes 65536x8,262144x8,1048576x8 \
  --warmup 5 \
  --repetitions 50
```

## 5. Convert the profile into a throughput CSV

The command below computes average time per named kernel and converts it into
logical-iteration throughput:

```bash
"${CALIPER_INSTALL_PREFIX}/bin/cali-query" -e launch_nd.cali | awk -F, '
BEGIN {
  print "mapping,cells,components,total_iterations,avg_seconds,throughput_iterations_per_second"
}
{
  kernel_name = ""
  total_sec = ""

  for (i = 1; i <= NF; ++i) {
    split($i, kv, "=")
    key = kv[1]
    value = kv[2]

    if ((key == "loop" || key == "region" || key == "path") &&
        value ~ /launch_nd_/) {
      kernel_name = value
    } else if (key == "sum#sum#time.duration") {
      total_sec = value + 0
    }
  }

  if (kernel_name ~ /launch_nd_/ &&
      match(kernel_name, /launch_nd_(.+)_cells([0-9]+)_comp([0-9]+)/, m)) {
    mapping = m[1]
    cells = m[2] + 0
    comp = m[3] + 0
    total_iterations = cells * comp
    avg_seconds = total_sec / 50
    throughput = total_iterations / avg_seconds
    print mapping "," cells "," comp "," total_iterations "," avg_seconds "," throughput
  }
}' > launch_nd_throughput.csv
```

`launch_nd_throughput.csv` is the file to use for plotting throughput curves.

## 6. Plot throughput curves with `gnuplot`

```bash
gnuplot -e "
set datafile separator ',';
set key left top;
set logscale x 2;
set xlabel 'Total logical iterations';
set ylabel 'Throughput (iterations/s)';
plot \
  '< awk -F, \"NR==1 || \\$1==\\\"flat\\\"\" launch_nd_throughput.csv' using 4:6 with linespoints title 'flat', \
  '< awk -F, \"NR==1 || \\$1==\\\"global\\\"\" launch_nd_throughput.csv' using 4:6 with linespoints title 'global', \
  '< awk -F, \"NR==1 || \\$1==\\\"block\\\"\" launch_nd_throughput.csv' using 4:6 with linespoints title 'block', \
  '< awk -F, \"NR==1 || \\$1==\\\"thread_local\\\"\" launch_nd_throughput.csv' using 4:6 with linespoints title 'thread_local'
"
```

## 7. Suggested study design

- Keep `components` fixed and sweep `cells` first.
- Run all mappings in the same executable invocation so the environment is
  identical.
- Use at least 5 warmup launches and 30-100 timed launches.
- Save both the Caliper profile and the derived throughput CSV for each build.
- Compare `flat`, `global`, `block`, and `thread_local` at each size, then
  examine how the gap changes with total problem size.
