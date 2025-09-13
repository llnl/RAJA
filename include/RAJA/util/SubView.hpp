#include "camp/number.hpp"
#include "camp/tuple.hpp"
#include "camp/array.hpp"
#include <vector>
#include <cassert>
#include <variant>
#include <tuple>
#include <type_traits>
#include <iostream>

// Slice descriptors
struct Range { long start, end; };
struct Fixed { long idx; };

// Helper to count non-fixed dimensions at compile time
template <typename... Slices>
constexpr size_t count_ranges() {
    return (0 + ... + (std::is_same_v<Slices, Range> ? 1 : 0));
}

// Helper to get the i-th element from a tuple
template <size_t I, typename Tuple>
decltype(auto) get_elem(Tuple&& tup) {
    return camp::get<I>(std::forward<Tuple>(tup));
}

template <typename T, size_t N, size_t... Is>
auto array_to_tuple_impl(const camp::array<T, N>& arr, std::index_sequence<Is...>) {
    return camp::make_tuple(arr[Is]...);
}

template <typename T, size_t N>
auto array_to_tuple(const camp::array<T, N>& arr) {
    return array_to_tuple_impl(arr, std::make_index_sequence<N>{});
}

// The generic subview class
template <typename ViewType, typename... Slices>
class SubView {
    ViewType view_;
    camp::tuple<Slices...> slices_;

    // Helper to map subview indices to parent indices
    template <typename IndexType, IndexType... Is>
    auto map_indices(IndexType* idxs, camp::idx_seq<Is...>) const {
        camp::array<int, sizeof...(Slices)> parent_indices{};
        size_t range_idx = 0;
        (
            (
                parent_indices[Is] = [&] {
                    const auto& s = get_elem<Is>(slices_);
                    if constexpr (std::is_same_v<std::decay_t<decltype(s)>, Range>) {
                        return s.start + idxs[range_idx++];
                    } else {
                        return s.idx;
                    }
                }()
            ), ...
        );
        return parent_indices;
    }

public:
    SubView(ViewType view, Slices... slices)
        : view_(view), slices_(slices...) {}

    template <typename... Idxs>
    auto operator()(Idxs... idxs) const {
        constexpr size_t nidx = count_ranges<Slices...>();
        static_assert(sizeof...(idxs) == nidx, "Wrong number of indices for subview");
        camp::array<long, nidx> arr{static_cast<long>(idxs)...};
        auto parent_indices = map_indices(arr.data(), camp::make_idx_seq_t<sizeof...(Slices)>());
        return camp::apply(view_, array_to_tuple(parent_indices));
    }
};
