//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"

/*
 *  RAJA::launch_nd benchmark driver
 *
 *  This example extends the launch_nd mapping demo into a benchmark driver for
 *  flattened and device launch_nd policies. It supports single-size runs and
 *  multi-size studies, and uses RAJA::Name so Caliper can aggregate results by
 *  policy/size combination.
 *
 *  Typical runs:
 *
 *    ./launch_nd --mapping all --repetitions 50 --warmup 5
 *    RAJA_CALIPER=1 CALI_CONFIG=runtime-report \
 *      ./launch_nd --mapping all --sizes 65536x8,262144x8
 *    RAJA_CALIPER=1 CALI_CONFIG=runtime-profile(output=launch_nd.cali,output.format=cali) \
 *      ./launch_nd --mapping all --sizes 65536x8,262144x8,1048576x8
 */

namespace
{

constexpr int block_size_1d = 256;
constexpr int block_x       = 16;
constexpr int block_y       = 16;

#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
int ceil_div(int value, int divisor) { return (value + divisor - 1) / divisor; }
#endif

enum class Mapping
{
  Flat,
  Global,
  Block,
  ThreadLocal,
  All
};

struct ProblemSize
{
  int cells      = 0;
  int components = 0;
};

struct Options
{
  Mapping mapping = Mapping::All;
  int warmup      = 5;
  int repetitions = 50;
  std::vector<ProblemSize> sizes = {};
};

struct RunResult
{
  int errors = 0;
};

#if defined(RAJA_ENABLE_CUDA)
using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<false>>;
using flat_exec     = RAJA::cuda_exec<block_size_1d>;
using global_cell_loop = RAJA::LoopPolicy<RAJA::cuda_global_x_direct>;
using global_comp_loop = RAJA::LoopPolicy<RAJA::cuda_global_y_direct>;
using block_cell_loop  = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;
using block_comp_loop  = RAJA::LoopPolicy<RAJA::cuda_block_y_direct>;
using thread_cell_loop = RAJA::LoopPolicy<RAJA::cuda_thread_x_loop>;
using thread_comp_loop = RAJA::LoopPolicy<RAJA::cuda_thread_y_loop>;
using resource_type = RAJA::resources::Cuda;

#elif defined(RAJA_ENABLE_HIP)
using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<false>>;
using flat_exec     = RAJA::hip_exec<block_size_1d>;
using global_cell_loop = RAJA::LoopPolicy<RAJA::hip_global_x_direct>;
using global_comp_loop = RAJA::LoopPolicy<RAJA::hip_global_y_direct>;
using block_cell_loop  = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;
using block_comp_loop  = RAJA::LoopPolicy<RAJA::hip_block_y_direct>;
using thread_cell_loop = RAJA::LoopPolicy<RAJA::hip_thread_x_loop>;
using thread_comp_loop = RAJA::LoopPolicy<RAJA::hip_thread_y_loop>;
using resource_type = RAJA::resources::Hip;

#else
using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;
using flat_exec     = RAJA::seq_exec;
using global_cell_loop = RAJA::LoopPolicy<RAJA::seq_exec>;
using global_comp_loop = RAJA::LoopPolicy<RAJA::seq_exec>;
using block_cell_loop  = RAJA::LoopPolicy<RAJA::seq_exec>;
using block_comp_loop  = RAJA::LoopPolicy<RAJA::seq_exec>;
using thread_cell_loop = RAJA::LoopPolicy<RAJA::seq_exec>;
using thread_comp_loop = RAJA::LoopPolicy<RAJA::seq_exec>;
using resource_type = RAJA::resources::Host;
#endif

const char* mapping_name(Mapping mapping)
{
  switch (mapping)
  {
    case Mapping::Flat:
      return "flat";
    case Mapping::Global:
      return "global";
    case Mapping::Block:
      return "block";
    case Mapping::ThreadLocal:
      return "thread_local";
    case Mapping::All:
      return "all";
  }

  return "unknown";
}

ProblemSize default_problem_size()
{
  return ProblemSize {262144, 8};
}

void print_usage(const char* executable)
{
  std::cout
      << "Usage: " << executable
      << " [flat|global|block|thread_local|all] [options]\n"
      << "Options:\n"
      << "  --mapping <flat|global|block|thread_local|all>\n"
      << "                            Select policy set to benchmark.\n"
      << "  --sizes <cellsxcomp,...>    Problem sizes, e.g. 262144x8 or"
      << " 65536x8,262144x8.\n"
      << "  --warmup <int>              Warmup launches per mapping/size.\n"
      << "  --repetitions <int>         Timed launches per mapping/size.\n"
      << "  --help                      Show this message.\n";
}

Mapping parse_mapping(const std::string& value)
{
  if (value == "flat" || value == "flattened")
  {
    return Mapping::Flat;
  }
  if (value == "global" || value == "grid")
  {
    return Mapping::Global;
  }
  if (value == "block")
  {
    return Mapping::Block;
  }
  if (value == "thread" || value == "thread_local" ||
      value == "thread-local")
  {
    return Mapping::ThreadLocal;
  }
  if (value == "all")
  {
    return Mapping::All;
  }

  throw std::runtime_error("unknown mapping '" + value +
                           "' (expected flat, global, block, thread_local,"
                           " or all)");
}

int parse_positive_int(const std::string& name, const std::string& value)
{
  try
  {
    const int parsed = std::stoi(value);
    if (parsed <= 0)
    {
      throw std::runtime_error(name + " must be greater than zero");
    }
    return parsed;
  }
  catch (const std::invalid_argument&)
  {
    throw std::runtime_error("invalid integer for " + name + ": '" + value + "'");
  }
  catch (const std::out_of_range&)
  {
    throw std::runtime_error("integer out of range for " + name + ": '" + value +
                             "'");
  }
}

ProblemSize parse_problem_size(const std::string& text)
{
  const std::size_t sep = text.find_first_of("xX");
  if (sep == std::string::npos)
  {
    throw std::runtime_error("invalid size '" + text +
                             "' (expected cellsxcomponents)");
  }

  return {parse_positive_int("cells", text.substr(0, sep)),
          parse_positive_int("components", text.substr(sep + 1))};
}

std::vector<ProblemSize> parse_problem_sizes(const std::string& text)
{
  std::vector<ProblemSize> result;
  std::size_t start = 0;

  while (start < text.size())
  {
    const std::size_t end = text.find(',', start);
    const std::string item =
        text.substr(start, end == std::string::npos ? std::string::npos
                                                    : end - start);
    if (item.empty())
    {
      throw std::runtime_error("empty entry in --sizes");
    }

    result.push_back(parse_problem_size(item));

    if (end == std::string::npos)
    {
      break;
    }
    start = end + 1;
  }

  if (result.empty())
  {
    throw std::runtime_error("--sizes requires at least one entry");
  }

  return result;
}

const char* require_value(int& index, int argc, char** argv, const char* option)
{
  if (index + 1 >= argc)
  {
    throw std::runtime_error(std::string(option) + " requires a value");
  }

  return argv[++index];
}

Options parse_options(int argc, char** argv)
{
  Options options;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h")
    {
      print_usage(argv[0]);
      std::exit(0);
    }
    else if (arg == "--mapping")
    {
      options.mapping = parse_mapping(require_value(i, argc, argv, "--mapping"));
    }
    else if (arg == "--sizes")
    {
      options.sizes = parse_problem_sizes(require_value(i, argc, argv, "--sizes"));
    }
    else if (arg == "--warmup")
    {
      options.warmup =
          parse_positive_int("--warmup", require_value(i, argc, argv, "--warmup"));
    }
    else if (arg == "--repetitions")
    {
      options.repetitions = parse_positive_int("--repetitions",
                                               require_value(i, argc, argv,
                                                             "--repetitions"));
    }
    else if (arg.rfind("--", 0) == 0)
    {
      throw std::runtime_error("unknown option '" + arg + "'");
    }
    else
    {
      options.mapping = parse_mapping(arg);
    }
  }

  return options;
}

std::vector<ProblemSize> selected_sizes(const Options& options)
{
  if (!options.sizes.empty())
  {
    return options.sizes;
  }

  return {default_problem_size()};
}

std::vector<Mapping> selected_mappings(Mapping mapping)
{
  if (mapping == Mapping::All)
  {
    return {Mapping::Flat, Mapping::Global, Mapping::Block,
            Mapping::ThreadLocal};
  }

  return {mapping};
}

std::string size_label(const ProblemSize& size)
{
  return std::to_string(size.cells) + "x" + std::to_string(size.components);
}

std::string kernel_name(Mapping mapping, const ProblemSize& size)
{
  return "launch_nd_" + std::string(mapping_name(mapping)) + "_cells" +
         std::to_string(size.cells) + "_comp" + std::to_string(size.components);
}

template<typename LaunchNdPolicy>
void launch_kernel(RAJA::resources::Resource res,
                   LaunchNdPolicy policy,
                   int* values_ptr,
                   const ProblemSize& size,
                   const std::string& name)
{
  auto cell_segment = RAJA::TypedRangeSegment<int>(0, size.cells);
  auto comp_segment = RAJA::TypedRangeSegment<int>(0, size.components);

  RAJA::launch_nd(res, policy, RAJA::segments(cell_segment, comp_segment),
                  RAJA::Name(name.c_str()),
                  [=] RAJA_HOST_DEVICE(int cell, int comp) {
                    values_ptr[comp + size.components * cell] =
                        1000 * cell + comp;
                  });

  // Synchronize so Caliper observes completed work for each named launch.
  res.wait();
}

int verify_result(RAJA::resources::Resource res,
                  int* values_ptr,
                  const ProblemSize& size)
{
  const std::size_t total =
      static_cast<std::size_t>(size.cells) * size.components;
  std::vector<int> values(total);

  res.memcpy(values.data(), values_ptr, sizeof(int) * total);
  res.wait();

  int errors = 0;
  for (int cell = 0; cell < size.cells; ++cell)
  {
    for (int comp = 0; comp < size.components; ++comp)
    {
      const int idx      = comp + size.components * cell;
      const int expected = 1000 * cell + comp;
      if (values[idx] != expected)
      {
        ++errors;
      }
    }
  }

  return errors;
}

template<typename LaunchNdPolicy>
RunResult benchmark_mapping(RAJA::resources::Resource res,
                            LaunchNdPolicy policy,
                            const Options& options,
                            Mapping mapping,
                            const ProblemSize& size)
{
  const std::size_t total =
      static_cast<std::size_t>(size.cells) * size.components;
  const std::string name = kernel_name(mapping, size);
  int* values_ptr = res.allocate<int>(total);

  for (int step = 0; step < options.warmup; ++step)
  {
    launch_kernel(res, policy, values_ptr, size, name);
  }

  for (int rep = 0; rep < options.repetitions; ++rep)
  {
    launch_kernel(res, policy, values_ptr, size, name);
  }

  RunResult result;
  result.errors = verify_result(res, values_ptr, size);

  std::cout << "  kernel: " << name << '\n'
            << "  logical size: " << size.cells << " x " << size.components
            << '\n'
            << "  warmup launches: " << options.warmup << '\n'
            << "  timed launches: " << options.repetitions << '\n'
            << "  result -- " << (result.errors == 0 ? "PASS" : "FAIL")
            << '\n';

  res.deallocate(values_ptr);
  return result;
}

RAJA::LaunchParams make_global_launch_params(const ProblemSize& size)
{
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  return RAJA::LaunchParams(
      RAJA::Teams(ceil_div(size.cells, block_x),
                  ceil_div(size.components, block_y)),
      RAJA::Threads(block_x, block_y));
#else
  RAJA_UNUSED_VAR(size);
  return RAJA::LaunchParams {};
#endif
}

RAJA::LaunchParams make_block_launch_params(const ProblemSize& size)
{
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  return RAJA::LaunchParams(RAJA::Teams(size.cells, size.components),
                            RAJA::Threads(1, 1));
#else
  RAJA_UNUSED_VAR(size);
  return RAJA::LaunchParams {};
#endif
}

RAJA::LaunchParams make_thread_launch_params(const ProblemSize& size)
{
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
  RAJA_UNUSED_VAR(size);
  return RAJA::LaunchParams(RAJA::Teams(1, 1), RAJA::Threads(block_x, block_y));
#else
  RAJA_UNUSED_VAR(size);
  return RAJA::LaunchParams {};
#endif
}

}  // namespace

int main(int argc, char** argv)
{
  try
  {
    const Options options = parse_options(argc, argv);
    const std::vector<ProblemSize> sizes = selected_sizes(options);

    std::cout << "\nRAJA launch_nd benchmark driver...\n";
#if defined(RAJA_ENABLE_CALIPER)
    std::cout << "  Caliper support: enabled\n";
#else
    std::cout << "  Caliper support: disabled\n";
#endif
    std::cout << "  study sizes:";
    for (const ProblemSize& size : sizes)
    {
      std::cout << ' ' << size_label(size);
    }
    std::cout << '\n';

    RAJA::resources::Resource res(resource_type {});

    int total_errors = 0;

    for (const ProblemSize& size : sizes)
    {
      std::cout << "\nStudy size " << size_label(size) << '\n';

      for (Mapping mapping : selected_mappings(options.mapping))
      {
        if (mapping == Mapping::Flat)
        {
          auto policy = RAJA::launch_nd_flattened_policy<flat_exec> {};
          const RunResult result =
              benchmark_mapping(res, policy, options, mapping, size);
          total_errors += result.errors;
          std::cout << "  flattened launch threads per block: " << block_size_1d
                    << "\n\n";
        }
        else if (mapping == Mapping::Global)
        {
          auto policy =
              RAJA::launch_nd_grid_policy<launch_policy,
                                          global_cell_loop,
                                          global_comp_loop>(
                  make_global_launch_params(size));
          const RunResult result =
              benchmark_mapping(res, policy, options, mapping, size);
          total_errors += result.errors;
          std::cout << "  global launch block shape: " << block_x << " x "
                    << block_y << "\n\n";
        }
        else if (mapping == Mapping::Block)
        {
          auto policy =
              RAJA::launch_nd_grid_policy<launch_policy,
                                          block_cell_loop,
                                          block_comp_loop>(
                  make_block_launch_params(size));
          const RunResult result =
              benchmark_mapping(res, policy, options, mapping, size);
          total_errors += result.errors;
          std::cout << "  block launch uses one logical iteration per team"
                    << "\n\n";
        }
        else
        {
          auto policy =
              RAJA::launch_nd_grid_policy<launch_policy,
                                          thread_cell_loop,
                                          thread_comp_loop>(
                  make_thread_launch_params(size));
          const RunResult result =
              benchmark_mapping(res, policy, options, mapping, size);
          total_errors += result.errors;
          std::cout << "  thread-local launch uses a single team with thread"
                    << " loops of shape " << block_x << " x " << block_y
                    << "\n\n";
        }
      }
    }

    std::cout << "DONE!...\n";
    return total_errors == 0 ? 0 : 1;
  }
  catch (const std::exception& ex)
  {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
}
