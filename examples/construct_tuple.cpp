#include <camp/tuple.hpp>
#include <iostream>

template<typename ... Args>
auto construct_tuple(Args... args) {
  return camp::tuple <Args...>(args...);
}

auto add1scalar(int a) { return a + 1; }

template<camp::idx_t ... indices, typename Tup>
auto add1(const camp::idx_seq<indices...>&, Tup tup) {
  return Tup(add1scalar(camp::get<indices>(tup))...);
}

int main() {
  auto tup = construct_tuple(1,2,3,4);
  std::cout << camp::get<0>( tup  ) << std::endl;
  auto tup_plus_one = add1(camp::make_idx_seq_t<4>{}, tup);
  std::cout << camp::get<0>( tup_plus_one  ) << std::endl;
  return 0;
}
