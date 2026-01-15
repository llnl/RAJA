###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# CAMP

## Type classes

### Expressions

An expression is a template of the form:

```c++
template <typename...Ts>
struct expr_s {
};
// OR
template <typename...Ts>
using expr = typename expr_s::type;
```

Generically it is an un-expanded template type that accepts one or more template
typename parameters.

### Values

Any complete type is a value

# Concepts

### namespaces:
* `RAJA::concepts` -- describes "Concepts" that RAJA provides
* `RAJA::type_traits` -- describes type traits of Concepts that can be used in SFINAE/static_assert contexts

## Concepts:
* Iterators: `Iterator`, `ForwardIterator`, `BidirectionalIterator`, `RandomAccessIterator`
* Ranges: `Range`, `ForwardRange`, `BidirectionalRange`, `RandomAccessRange`
* Types: `Arithmetic`, `Integral`, `Signed`, `Unsigned`, `FloatingPoint`
* Comparable: `LessThanComparable`, `GreaterThanComparable`, `LessEqualComparable`, `GreaterEqualComparable`, `EqualityComparable`, `ComparableTo<T,U>`, `Comparable`

## RAJA-Specific Concepts:
* `ExecutionPolicy` - checks to make sure a type models an ExecutionPolicy
* `BinaryFunction<Function, Return, Arg1, Arg2>` - checks to see that invoking `Function` with two arguments is valid and has a return type convertible to `Return`
* `UnaryFunction<Function, Return, Arg>` - checks to see that invoking `Function` with one argument is valid and has a return type convertible to `Return`

## Type Traits:
* Iterators: `is_iterator`, `is_forward_iterator`, `is_bidirectional_iterator`, `is_random_access_iterator`
* Ranges: `is_range`, `is_forward_range`, `is_bidirectional_range`, `is_random_access_range`
* Types: `is_arithmetic`, `is_integral`, `is_signed`, `is_unsigned`, `is_floating_point`
* Comparison: `is_comparable`, `is_comparable_to<T,U>`

## RAJA-Specific Type Traits:
* Execution Policies: `is_exection_policy`, `is_sequential_policy`, `is_simd_policy`, `is_openmp_policy`, `is_target_openmp_policy`, `is_cuda_policy`, `is_indexset_policy`
* Functions: `is_binary_function`, `is_unary_function`
* IndexSet: `is_index_set`

## Concept Components:

All building blocks live within the RAJA::concepts namespace. Defined concepts will be in UpperCamelCase while all building blocks adhere to snake_case.

### Naming convention:
* T, U, etc -- types
* B -- a boolean-like type (has `constexpr` value member of type `bool`)
* Ts..., Bs... -- a type or boolean-type variadic list
* bool-expr --expression that evaluates to true or false
* expr -- an expression

### Building Blocks for use within Concepts
* `val<T>()` -- an alias  #to `std::declval<T>()` -- types in a concept shall never be instantiated with a constructor (e.g. `T()`). `val<T>()` **must** be used
* `cval<T>()` -- alias for `val<T const>()`
* `has_type<T>(expr)` -- expression evaluates to return type of T
* `convertible_to<T>(expr)` -- expression evaluation return type is convertible to T
* `is(B)` -- results in a valid expression IFF `B::value` evaluates to true
* `is_not(B)` -- results in a valid expression IFF `B::value` evaluates to false

### Logical Reductions for use within type traits / SFINAE constructs
* `all_of<Bs...>` -- returns B where value is true IFF all `Bs::value ...` are true
* `none_of<Bs...>` -- returns B where value is true IFF all `Bs::value ...` are false
* `any_of<Bs...>` -- returns B where value is true IFF any `Bs::value ...` are true
* `negate<B>` -- returns B where value is `!B::value`
* `bool_<bool>` -- creates a BoolLike type

### Using Concepts at API/User-Level Constructs
* `enable_if<Bs...>` -- type to use to conditional enable/disable visibility. All `Bs::value ...` must evaluate to true
* `requires_<Concept, Ts...>` -- SFINAE wrapper around a concept check; usually to define Type Traits from Concepts

### Macros:

#### `DefineConcept(...)`

creates a new type expression (decltype) that can contain arbitrary code that must be valid in order for a type to conform to some defined Concept

#### `DefineTypeTraitFromConcept(type_trait_name, ConceptName)`

creates a new type called `type_trait_name` which derives from `requires_<ConceptName, ... >`. Useful for not rewriting boiler-plate code for type trait definitions

## Defining Concepts:

## Concept Definition Requirements:
* shall be defined as a new type (**not** an alias)
* shall adhere to CamelCase naming conventions
* shall exist under the `RAJA::concepts` namespace for API-provided concepts, otherwise shall exist under some `concepts` namespace which may be nested
* shall only accept `typename`s as template arguments (`template <class...> class` is not permitted)
* shall not specify any default template arguments -- prefer creating a new concept modeling the specialization of an existing Concept.
* shall not contain any type aliases, functions, or [static] [constexpr] members (`sizeof(Concept)` must always evaluate to 0)


## Examples

### Basic Example

```cpp
template <typename T>
struct Iterator : DefineConcept(*(val<T>()), has_type<T &>(++val<T &>()) {}
```

### Requiring Concepts modeling another Concept

```cpp
template <typename T>
struct ForwardIterator : DefineConcept(Iterator<T>(), val<T &>()++, *val<T &>()++) {}
```

Other usages can be found under:
* `include/RAJA/util/concepts.hpp`
* `include/RAJA/policy/PolicyBase.hpp`
* `include/RAJA/util/Operators.hpp`

# Deprecating older features in RAJA

As the RAJA codebase evolves, there may come a point where features once used have been replaced with more viable options. To aid users in transitioning from an older API call to a preferred API, we introduce deprecation macros which should *cautiously* and *effectively* be used by RAJA developers.

The following macros are defined by RAJA that assist with defining deprecation attributes for Functions, Types (structs/classes), and type aliases:

* `RAJA_DEPRECATE("Message")`
* `RAJA_DEPRECATE_ALIAS("Message")` -- this will **only** work with a C++14 - enabled compiler

The following preprocessor tokens are defined by RAJA with the following constraints:

* `RAJA_HAS_CXX14` -- defined IFF RAJA detects a C++14 compiler
* `RAJA_HAS_ATTRIBUTE_DEPRECATED` -- defined IFF RAJA detects compiler support of `[[deprecated("Message")]]`

RAJA will automatically expand `RAJA_DEPRECATE()` and `RAJA_DEPRECATE_ALIAS()` if they are supported with the toolchain used for compilation.

Below is a source code example where deprecation annotations are added to several constructs:

```cpp

// Deprecating a class member variable
////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct Pair {
  T first;
  RAJA_DEPRECATE("Second has been removed") T second;
};

// Deprecating a type
////////////////////////////////////////////////////////////////////////////////

struct RAJA_DEPRECATE("The Cilk execution policy has been removed from RAJA")
cilk_exec {
};


// Deprecating a function
////////////////////////////////////////////////////////////////////////////////

template <typename Exec, typename Index1, typename Index2, typename Body>
RAJA_DEPRECATE("This version of forall is deprecated. switch to using RAJA::make_range(begin, end) instead of listing the index parameters)")
int forall(Exec && p, Index1, Index2, Body &&) {
  return 0;
}


// Deprecating a type alias (requires C++14)
////////////////////////////////////////////////////////////////////////////////

using Real_ptr RAJA_DEPRECATE_ALIAS("Real_ptr will be removed in 2018 (JK)") = double*;

int bar() {
  return 0;
}

// Example code with information showing where warnings will be emitted by
// a compiler given the above deprecation attributes constructed above
////////////////////////////////////////////////////////////////////////////////

using exec_pol = cilk_exec; // warning is emitted here

int main() {
  Index_type ind = 4; // warning emitted here (Index_type)
  Real_ptr ptr = nullptr; // warning emitted here (Real_ptr) (C++14 only)
  forall(exec_pol{}, 0, 100, [] { }); // warnings emitted here (exec_pol, forall specialization)
  Pair<int> pair;
  pair.first = 4;
  pair.second = 4; // warning here (second member)
  return pair.first + ind + (ptr - ptr);
}

```

It is important to note that deprecation macros should only be used for user-facing API calls. An end user should never receive a compiler warning stemming from an internal API call. All requests for feature deprecation must be approved in a pull request.
