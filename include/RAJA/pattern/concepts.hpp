#ifndef RAJA_type_concepts_HPP
#define RAJA_type_concepts_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/detail/TypeTraits.hpp"
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/MultiPolicy.hpp"
#include "RAJA/util/resource.hpp"
#include "RAJA/index/IndexSet.hpp"

namespace RAJA
{
#define RAJA_DEFINE_TRAIT_FROM_TYPEDEF(TYPEDEF)                                \
  template<typename T>                                                         \
  struct is_##TYPEDEF                                                          \
  {                                                                            \
  private:                                                                     \
    template<typename U>                                                       \
    using have_t = typename U::TYPEDEF;                                        \
    template<typename U>                                                       \
    using have_type_t = typename U::TYPEDEF##_type;                            \
                                                                               \
  public:                                                                      \
    static constexpr bool value =                                              \
        std::is_base_of_v<detected_t<have_t, T>, T> ||                         \
        std::is_base_of_v<detected_t<have_type_t, T>, T>;                      \
    constexpr operator bool() const noexcept { return value; }                 \
  };                                                                           \
  template<typename T>                                                         \
  inline constexpr bool is_##TYPEDEF##_v = is_##TYPEDEF<T>::value;

#define RAJA_DEFINE_CONCEPT_AND_TRAIT_FROM_TYPEDEF(TYPEDEF, CXX20_CONCEPT)     \
  RAJA_DEFINE_TRAIT_FROM_TYPEDEF(TYPEDEF)                                      \
  template<typename T>                                                         \
  concept CXX20_CONCEPT = is_##TYPEDEF##_v<T>;

namespace concepts
{

/// A RAJA ExecutionPolicy is a backend-specific directive supplied by the user
/// like hip_exec or seq_exec that instructs RAJA how to configure a parallel
/// kernel
template<typename Pol>
concept ExecutionPolicy =
    RAJA::type_traits::is_same_decay_v<decltype(Pol::policy), ::RAJA::Policy> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::pattern),
                                       ::RAJA::Pattern> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::launch), ::RAJA::Launch> &&
    RAJA::type_traits::is_same_decay_v<decltype(Pol::platform),
                                       ::RAJA::Platform>;

template<typename T>
concept IndexSetType =
    static_cast<bool>(RAJA::type_traits::is_index_set<std::decay_t<T>>::value);

template<typename T>
concept IndexSetPolicy =
    static_cast<bool>(type_traits::is_indexset_policy<std::decay_t<T>>::value);

template<typename T>
concept Resource = RAJA::type_traits::is_resource<std::decay_t<T>>::value;

template<typename T>
concept ForallParams =
    expt::type_traits::is_ForallParamPack<std::decay_t<T>>::value;

template<typename T>
concept MultiPolicyConcept =
    static_cast<bool>(RAJA::type_traits::is_multi_policy<T>::value);

}  // namespace concepts

namespace type_traits
{
DefineTypeTraitFromConcept(is_execution_policy,
                           RAJA::concepts::ExecutionPolicy);
}


}  // namespace RAJA

#endif