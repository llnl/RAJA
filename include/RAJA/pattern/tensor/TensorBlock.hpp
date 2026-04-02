/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#if 0
#ifndef RAJA_pattern_tensor_TensorBlock_HPP
#define RAJA_pattern_tensor_TensorBlock_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/TensorRegister.hpp"
#include "RAJA/util/StaticLayout.hpp"

namespace RAJA
{
namespace expt
{
  template<camp::idx_t ... IDX>
  struct GetSeqValue;

  template<camp::idx_t IDX0, camp::idx_t ... IDX_REST>
  struct GetSeqValue<IDX0, IDX_REST...>
  {
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      camp::idx_t get(camp::idx_t idx){
        return idx == 0 ? IDX0 : GetSeqValue<IDX_REST...>::get(idx-1);
      }
  };
  template<>
    struct GetSeqValue<>
    {
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr
        camp::idx_t get(camp::idx_t ){
          return 0;
        }
    };

  template <typename IdxLin,
            IdxLin... RangeInts,
            IdxLin... Sizes,
            IdxLin... Strides>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  constexpr
  RAJA::internal::TensorTile<IdxLin, RAJA::internal::TENSOR_FULL, sizeof...(RangeInts)>
  get_layout_default_tile(
      RAJA::detail::StaticLayoutBase_impl<IdxLin,
      camp::int_seq<IdxLin, RangeInts...>,
      camp::int_seq<IdxLin, Sizes...>,
      camp::int_seq<IdxLin, Strides...>> const &)
  {
    using tile_type = RAJA::internal::TensorTile<IdxLin, RAJA::internal::TENSOR_FULL, sizeof...(RangeInts)>;
    return tile_type{ {IdxLin(0*Sizes)...}, {IdxLin(Sizes)...}};
  }


  template<typename STORAGE_POLICY, typename T,  typename LAYOUT>
  class TensorBlockStorage : public RAJA::internal::ViewBase<T, T*, LAYOUT>
  {
    public:
      using self_type = TensorBlockStorage<STORAGE_POLICY, T,  LAYOUT>;
      using base_type = RAJA::internal::ViewBase<T, T*, LAYOUT>;
      using element_type = T;
      using layout_type = LAYOUT;

      static constexpr camp::idx_t s_num_elem = LAYOUT::s_size;
      using tile_type = decltype(get_layout_default_tile(layout_type{}));
      using index_type = typename tile_type::index_type;
      using ref_type = internal::TensorRef<T*, typename tile_type::index_type, tile_type::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>;
      using const_ref_type = internal::TensorRef<T const *, typename tile_type::index_type, tile_type::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>;

      /*!
       * Gets the default tile of this storage block
       * That tile always start at 0, and extends to the full tile sizes
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      decltype(get_layout_default_tile(layout_type{}))
      s_get_default_tile()
      {
        return get_layout_default_tile(layout_type{});
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorBlockStorage() noexcept :
        base_type(&m_data[0], layout_type{})
      {
      }


      template<typename TILE_TYPE, camp::idx_t ... DIM_SEQ>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref_expanded(TILE_TYPE const &tile, camp::idx_seq<DIM_SEQ...> const &)  noexcept ->
        internal::TensorRef<T*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>
      {
        using ref_t =
            internal::TensorRef<T*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>;

        return ref_t{&m_data[0], {(typename TILE_TYPE::index_type)base_type::get_layout().template get_dim_stride<DIM_SEQ>()...}, tile};
      }

      template<typename TILE_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref(TILE_TYPE const &tile) noexcept ->
        decltype(create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{}))
      {
        return create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{});
      }

      template<typename TILE_TYPE, camp::idx_t ... DIM_SEQ>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref_expanded(TILE_TYPE const &tile, camp::idx_seq<DIM_SEQ...> const &) const noexcept ->
        internal::TensorRef<T const*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>
      {
        using ref_t =
            internal::TensorRef<T const*, typename TILE_TYPE::index_type, TILE_TYPE::s_tensor_size, layout_type::n_dims, layout_type::stride_one_dim>;

        return ref_t{&m_data[0], {(typename TILE_TYPE::index_type)base_type::get_layout().template get_dim_stride<DIM_SEQ>()...}, tile};
      }

      template<typename TILE_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      auto create_ref(TILE_TYPE const &tile) const noexcept ->
        decltype(create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{}))
      {
        return create_ref_expanded(tile, camp::make_idx_seq_t<layout_type::n_dims>{});
      }


      template<camp::idx_t ... DIM_SEQ>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      ref_type get_ref_expanded(camp::idx_seq<DIM_SEQ...> const &) noexcept
      {
        return ref_type{&m_data[0], {index_type(base_type::get_layout().template get_dim_stride<DIM_SEQ>())...},
          {
              {(0*DIM_SEQ)...},
              {index_type(base_type::get_layout().template get_dim_size<DIM_SEQ>())...}
          }};
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      ref_type get_ref() noexcept
      {
        return get_ref_expanded(camp::make_idx_seq_t<layout_type::n_dims>{});
      }

      template<camp::idx_t ... DIM_SEQ>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      const_ref_type get_ref_expanded(camp::idx_seq<DIM_SEQ...> const &) const noexcept
      {
        return const_ref_type{&m_data[0], {index_type(base_type::get_layout().template get_dim_stride<DIM_SEQ>())...},
          {
              {(0*DIM_SEQ)...},
              {index_type(base_type::get_layout().template get_dim_size<DIM_SEQ>())...}
          }};
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      const_ref_type get_ref() const noexcept
      {
        return get_ref_expanded(camp::make_idx_seq_t<layout_type::n_dims>{});
      }


    private:
      T m_data[s_num_elem];
  };


  class TensorBlockConcreteBase{};


  template<typename REGISTER_POLICY,
           typename ELEMENT_TYPE,
           typename LAYOUT,
           typename SIZES,
           typename STORAGE_POLICY>
  class TensorBlock;





  template<typename REGISTER_POLICY,
           typename ELEMENT_TYPE,
           camp::idx_t ... LAYOUT,
           camp::idx_t ... DIM_SIZES,
           typename STORAGE_POLICY>

  class TensorBlock<REGISTER_POLICY,
                    ELEMENT_TYPE,
                    TensorLayout<LAYOUT...>,
                    camp::idx_seq<DIM_SIZES...>,
                    STORAGE_POLICY> : public TensorBlockConcreteBase
  {


    public:
      using self_type = TensorBlock<REGISTER_POLICY, ELEMENT_TYPE,
          TensorLayout<LAYOUT...>,
          camp::idx_seq<DIM_SIZES...>,
          STORAGE_POLICY>;
      using element_type = camp::decay<ELEMENT_TYPE>;

      using layout_type = camp::idx_seq<LAYOUT...>;

      using register_type =
            TensorRegister<REGISTER_POLICY,
                           ELEMENT_TYPE,
                           TensorLayout<LAYOUT...>,
                           camp::idx_seq<
                              (0*LAYOUT+RegisterTraits<REGISTER_POLICY,ELEMENT_TYPE>::s_num_elem)...>>;

      static constexpr camp::idx_t s_num_dims = sizeof...(DIM_SIZES);

      using storage_layout = RAJA::StaticLayout<camp::idx_seq<LAYOUT...>, DIM_SIZES...>;
      //using storage_layout = RAJA::StaticLayout<camp::idx_seq<1,0>, DIM_SIZES...>;
      using storage_type = TensorBlockStorage<STORAGE_POLICY, ELEMENT_TYPE, storage_layout>;

      using index_type = camp::idx_t;

      using tile_type = typename storage_type::tile_type;
      using ref_type = internal::TensorRef<element_type *, typename tile_type::index_type, tile_type::s_tensor_size, storage_layout::n_dims, storage_layout::stride_one_dim>;

      // to masquerade as a ET node
      using result_type = register_type;



    public:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return true;
      }


      /*!
       * Gets the size of the tensor
       * Since this is a vector, just the length of the vector in dim 0
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr int s_dim_elem(int dim){
        return GetSeqValue<DIM_SIZES...>::get(dim);
      }

      /*!
       * Gets the size of the tensor
       * Since this is a vector, just the length of the vector in dim 0
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr int getDimSize(int dim) const{
        return GetSeqValue<DIM_SIZES...>::get(dim);
      }

      /*!
       * Gets the default tile of this tensor
       * That tile always start at 0, and extends to the full tile sizes
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      decltype(get_layout_default_tile(storage_layout{}))
      get_default_tile() const
      {
        return get_layout_default_tile(storage_layout{});
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      storage_type s_create_temporary() {
        return storage_type();
      }

      template<typename REF_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      internal::ET::TensorLoadStore<register_type, REF_TYPE>
      create_et_store_ref(REF_TYPE const &ref) {
        return internal::ET::TensorLoadStore<register_type, REF_TYPE>(ref);
      }

      template<typename REF_TYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      internal::ET::TensorLoadStore<register_type, REF_TYPE>
      s_load_ref(REF_TYPE const &ref) {
        return internal::ET::TensorLoadStore<register_type, REF_TYPE>(ref);
      }


      TensorBlock() = default;
      ~TensorBlock() = default;

      TensorBlock(TensorBlock const &) = default;

      self_type &operator=(self_type const &x) = default;




  };



namespace internal {

namespace ET{


} // namespace ET
} // namespace internal
} // namespace expt
}  // namespace RAJA


#endif

#endif
