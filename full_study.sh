#!/usr/bin/env bash

set -euo pipefail

export REPO_DIR=$(pwd)/RAJA
export CALIPER_INSTALL_PREFIX=$(pwd)/install-caliper
export CALIPER_DIR="${CALIPER_INSTALL_PREFIX}/share/cmake/caliper"

COMPILER_VERSION="${COMPILER_VERSION:-7.2.1}"
GPU_ARCH="${GPU_ARCH:-gfx942}"
BUILD_DIR="build_lc_toss4-amdclang-${COMPILER_VERSION}-${GPU_ARCH}"
SIZES="${SIZES:-65536x8,262144x8,1048576x8}"
WARMUP="${WARMUP:-5}"
REPETITIONS="${REPETITIONS:-50}"
TERMINAL_PLOT="${TERMINAL_PLOT:-0}"
TERMINAL_PLOT_SIZE="${TERMINAL_PLOT_SIZE:-120,35}"
PLOT_FORMAT="${PLOT_FORMAT:-svg}"
LAUNCH_ND_EXE="${REPO_DIR}/${BUILD_DIR}/bin/launch_nd"
LAUNCH_ND_SOURCE="${REPO_DIR}/examples/launch_nd.cpp"

echo "Running full study with ROCm ${COMPILER_VERSION} on ${GPU_ARCH}"

cd "${REPO_DIR}"
if [ ! -x "${LAUNCH_ND_EXE}" ]; then
  CALIPER_DIR="${CALIPER_DIR}" ./scripts/lc-builds/toss4_amdclang.sh "${COMPILER_VERSION}" "${GPU_ARCH}"

  cmake --build "${BUILD_DIR}" --target launch_nd -j
elif [ "${LAUNCH_ND_SOURCE}" -nt "${LAUNCH_ND_EXE}" ]; then
  echo "Rebuilding launch_nd because the source is newer than the executable"
  cmake --build "${BUILD_DIR}" --target launch_nd -j
else
  echo "Reusing existing build at ${BUILD_DIR}"
fi

cd "${BUILD_DIR}"
RAJA_CALIPER=1 \
CALI_CONFIG='runtime-profile(output=launch_nd.cali,output.format=cali)' \
./bin/launch_nd \
  --mapping all \
  --sizes "${SIZES}" \
  --warmup "${WARMUP}" \
  --repetitions "${REPETITIONS}"

"${CALIPER_INSTALL_PREFIX}/bin/cali-query" \
  -e \
  launch_nd.cali \
| awk -F, -v repetitions="${REPETITIONS}" '
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

    if (key == "loop" || key == "region" || key == "path") {
      if (value ~ /launch_nd_/) {
        kernel_name = value
      }
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
    avg_seconds = total_sec / repetitions
    throughput = total_iterations / avg_seconds
    print mapping "," cells "," comp "," total_iterations "," avg_seconds "," throughput
  }
}' > launch_nd_throughput.csv

awk -F, 'NR == 1 || $1 == "flat"' launch_nd_throughput.csv > launch_nd_throughput_flat.csv
awk -F, 'NR == 1 || $1 == "global"' launch_nd_throughput.csv > launch_nd_throughput_global.csv
awk -F, 'NR == 1 || $1 == "block"' launch_nd_throughput.csv > launch_nd_throughput_block.csv
awk -F, 'NR == 1 || $1 == "thread_local"' launch_nd_throughput.csv > launch_nd_throughput_thread_local.csv

if [ "${PLOT_FORMAT}" = "png" ]; then
gnuplot -e "
set datafile separator ',';
set terminal pngcairo size 1400,900 enhanced font ',14';
set output 'launch_nd_throughput.png';
set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb '#ffffff' behind;
set border lw 2 lc rgb '#000000';
set key outside right top;
set logscale x 2;
set logscale y 10;
set grid xtics ytics lc rgb '#bfbfbf' lw 1.5;
set tics nomirror;
set tics textcolor rgb '#000000';
set xlabel 'Total logical iterations';
set ylabel 'Throughput (iterations/s)';
set format x '2^{%L}';
set format y '10^{%L}';
set style line 1 lt 1 lw 5 pt 7 ps 1.8 lc rgb '#0072B2';
set style line 2 lt 1 lw 5 pt 5 ps 1.8 lc rgb '#009E73';
set style line 3 lt 1 lw 5 pt 9 ps 1.8 lc rgb '#D55E00';
set style line 4 lt 1 lw 5 pt 13 ps 1.8 lc rgb '#CC79A7';
plot \
  'launch_nd_throughput_flat.csv' using 4:6 with linespoints ls 1 title 'flat', \
  'launch_nd_throughput_global.csv' using 4:6 with linespoints ls 2 title 'global', \
  'launch_nd_throughput_block.csv' using 4:6 with linespoints ls 3 title 'block', \
  'launch_nd_throughput_thread_local.csv' using 4:6 with linespoints ls 4 title 'thread_local'
"
elif [ "${PLOT_FORMAT}" = "svg" ]; then
gnuplot -e "
set datafile separator ',';
set terminal svg size 1400,900 dynamic;
set output 'launch_nd_throughput.svg';
set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb '#ffffff' behind;
set border lw 2 lc rgb '#000000';
set key outside right top;
set logscale x 2;
set logscale y 10;
set grid xtics ytics lc rgb '#bfbfbf' lw 1.5;
set tics nomirror;
set tics textcolor rgb '#000000';
set xlabel 'Total logical iterations';
set ylabel 'Throughput (iterations/s)';
set format x '2^{%L}';
set format y '10^{%L}';
set style line 1 lt 1 lw 5 pt 7 ps 1.8 lc rgb '#0072B2';
set style line 2 lt 1 lw 5 pt 5 ps 1.8 lc rgb '#009E73';
set style line 3 lt 1 lw 5 pt 9 ps 1.8 lc rgb '#D55E00';
set style line 4 lt 1 lw 5 pt 13 ps 1.8 lc rgb '#CC79A7';
plot \
  'launch_nd_throughput_flat.csv' using 4:6 with linespoints ls 1 title 'flat', \
  'launch_nd_throughput_global.csv' using 4:6 with linespoints ls 2 title 'global', \
  'launch_nd_throughput_block.csv' using 4:6 with linespoints ls 3 title 'block', \
  'launch_nd_throughput_thread_local.csv' using 4:6 with linespoints ls 4 title 'thread_local'
"
elif [ "${PLOT_FORMAT}" != "none" ]; then
  echo "Unsupported PLOT_FORMAT=${PLOT_FORMAT} (expected png, svg, or none)" >&2
  exit 1
fi

if [ "${TERMINAL_PLOT}" = "1" ]; then
  gnuplot -e "
set datafile separator ',';
set terminal dumb size ${TERMINAL_PLOT_SIZE};
set key outside;
set logscale x 2;
set logscale y 10;
set xlabel 'Total logical iterations';
set ylabel 'Throughput (iterations/s)';
plot \
  'launch_nd_throughput_flat.csv' using 4:6 with linespoints title 'flat', \
  'launch_nd_throughput_global.csv' using 4:6 with linespoints title 'global', \
  'launch_nd_throughput_block.csv' using 4:6 with linespoints title 'block', \
  'launch_nd_throughput_thread_local.csv' using 4:6 with linespoints title 'thread_local'
"
fi

echo "Wrote $(pwd)/launch_nd.cali"
echo "Wrote $(pwd)/launch_nd_throughput.csv"
if [ "${PLOT_FORMAT}" = "png" ]; then
  echo "Wrote $(pwd)/launch_nd_throughput.png"
elif [ "${PLOT_FORMAT}" = "svg" ]; then
  echo "Wrote $(pwd)/launch_nd_throughput.svg"
fi
