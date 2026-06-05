//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Registry_HPP
#define RAJA_Registry_HPP

#include <memory>

namespace RAJA
{
namespace util
{

template<typename T>
class RegistryEntry
{
  std::string Name, Desc;
  std::shared_ptr<T> object;

public:
  RegistryEntry(const std::string& N,
                const std::string& D,
                std::shared_ptr<T> (*C)())
      : Name(N),
        Desc(D),
        object(C())
  {}

  const std::string& getName() const { return Name; }

  const std::string& getDesc() const { return Desc; }

  T* get() const { return object.get(); }
};

/// A global registry used in conjunction with static constructors to make
/// pluggable components (like targets or garbage collectors) "just work" when
/// linked with an executable.
template<typename T>
class Registry
{
public:
  using type  = T;
  using entry = RegistryEntry<T>;

  class node;
  class iterator;

private:
  Registry() = delete;

  friend class node;
  static node *Head, *Tail;

public:
  /// Node in linked list of entries.
  ///
  class node
  {
    friend class iterator;
    friend Registry<T>;

    node* Next;
    const entry& Val;

  public:
    node(const entry& V) : Next(nullptr), Val(V) {}
  };

  /// Add a node to the Registry: this is the interface between the plugin and
  /// the executable.
  ///
  /// This function is exported by the executable and called by the plugin to
  /// add a node to the executable's registry. Therefore it's not defined here
  /// to avoid it being instantiated in the plugin and is instead defined in
  /// the executable (see RAJA_INSTANTIATE_REGISTRY below).
  static RAJASHAREDDLL_API void add_node(node* N);

  /// Iterators for registry entries.
  ///
  class iterator
  {
    const node* Cur;

  public:
    explicit iterator(const node* N) : Cur(N) {}

    bool operator==(const iterator& That) const { return Cur == That.Cur; }

    bool operator!=(const iterator& That) const { return Cur != That.Cur; }

    iterator& operator++()
    {
      Cur = Cur->Next;
      return *this;
    }

    const entry& operator*() const { return Cur->Val; }

    const entry* operator->() const { return &Cur->Val; }
  };

  // begin is not defined here in order to avoid usage of an undefined static
  // data member, instead it's instantiated by RAJA_INSTANTIATE_REGISTRY.
  static RAJASHAREDDLL_API iterator begin();

  static iterator end() { return iterator(nullptr); }

  /// A static registration template.
  template<typename V>
  class add
  {
    entry Entry;
    node Node;

    static std::shared_ptr<T> CtorFn() { return std::make_shared<V>(); }

  public:
    add(const std::string& Name, const std::string& Desc)
        : Entry(Name, Desc, CtorFn),
          Node(Entry)
    {
      add_node(&Node);
    }
  };
};

}  // namespace util
}  // namespace RAJA

#define RAJA_INSTANTIATE_REGISTRY(REGISTRY_CLASS)                              \
  namespace RAJA                                                               \
  {                                                                            \
  namespace util                                                               \
  {                                                                            \
  template<typename T>                                                         \
  typename Registry<T>::node* Registry<T>::Head = nullptr;                     \
  template<typename T>                                                         \
  typename Registry<T>::node* Registry<T>::Tail = nullptr;                     \
  template<typename T>                                                         \
  void Registry<T>::add_node(typename Registry<T>::node* N)                    \
  {                                                                            \
    if (Tail)                                                                  \
      Tail->Next = N;                                                          \
    else                                                                       \
      Head = N;                                                                \
    Tail = N;                                                                  \
  }                                                                            \
  template<typename T>                                                         \
  typename Registry<T>::iterator Registry<T>::begin()                          \
  {                                                                            \
    return iterator(Head);                                                     \
  }                                                                            \
  template REGISTRY_CLASS::node* Registry<REGISTRY_CLASS::type>::Head;         \
  template REGISTRY_CLASS::node* Registry<REGISTRY_CLASS::type>::Tail;         \
  template void Registry<REGISTRY_CLASS::type>::add_node(                      \
      REGISTRY_CLASS::node*);                                                  \
  template REGISTRY_CLASS::iterator Registry<REGISTRY_CLASS::type>::begin();   \
  }                                                                            \
  }

#endif
