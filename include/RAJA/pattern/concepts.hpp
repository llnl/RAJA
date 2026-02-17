#ifndef RAJA_type_concepts_HPP
#define RAJA_type_concepts_HPP

#include "RAJA/config.hpp"
#include "RAJA/index/IndexSet.hpp"
namespace RAJA {
#define RAJA_DEFINE_TRAIT_FROM_TYPEDEF(TYPEDEF)         \
  template <typename T>                                        \
  struct is_##TYPEDEF {                                        \
   private:                                                    \
    template <typename U>                                      \
    using have_t = typename U::TYPEDEF;                        \
    template <typename U>                                      \
    using have_type_t = typename U::TYPEDEF##_type;            \
                                                               \
   public:                                                     \
    static constexpr bool value =                              \
        std::is_base_of_v<detected_t<have_t, T>, T> ||         \
        std::is_base_of_v<detected_t<have_type_t, T>, T>;      \
    constexpr operator bool() const noexcept { return value; } \
  };                                                           \
  template <typename T>                                        \
  inline constexpr bool is_##TYPEDEF##_v = is_##TYPEDEF<T>::value;

#define RAJA_DEFINE_CONCEPT_AND_TRAIT_FROM_TYPEDEF(TYPEDEF,       \
                                                          CXX20_CONCEPT) \
  RAJA_DEFINE_TRAIT_FROM_TYPEDEF(TYPEDEF)                         \
  template <typename T>                                                  \
  concept CXX20_CONCEPT = is_##TYPEDEF##_v<T>;


#endif 
}