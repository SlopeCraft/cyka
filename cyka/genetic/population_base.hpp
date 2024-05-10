//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GABASE_H
#define HEURISTICFLOWR_GABASE_H

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>

#include <Eigen/Dense>

namespace cyka::genetic {



template <class const_gene_view, class mut_gene_view>
using crossover_function = void(const const_gene_view &parent1,
                                const const_gene_view &parent2,
                                mut_gene_view &child1, mut_gene_view &child2);

template <class const_gene_view, class mut_gene_view>
using mutate_function = void(const const_gene_view &parent,
                             mut_gene_view &child);

namespace detail {
class population_common_base {
public:
  virtual ~population_common_base() = default;
};

} // namespace detail

template <class gene, class mut_gene_view, class const_gene_view>
  requires std::move_constructible<gene> and std::is_move_assignable_v<gene> and
           std::is_default_constructible_v<gene>
           //           std::is_constructible_v<mut_gene_view, gene> and
           //           std::is_constructible_v<const_gene_view, gene> and
           //           std::is_constructible_v<const_gene_view,
           //           mut_gene_view > and
           and std::is_assignable_v<gene, mut_gene_view> and
           std::is_assignable_v<gene, const_gene_view> and
           std::is_assignable_v<mut_gene_view, const_gene_view>
class population_base : public detail::population_common_base {
public:
  using gene_type = gene;
  using mut_gene_view_type = mut_gene_view;
  using const_gene_view_type = const_gene_view;

  virtual ~population_base() = default;

  [[nodiscard]] virtual size_t population_size() const noexcept = 0;

  [[nodiscard]] virtual mut_gene_view gene_at(size_t index) noexcept = 0;
  [[nodiscard]] virtual const_gene_view
  gene_at(size_t index) const noexcept = 0;

  virtual void set_gene_at(size_t index, const const_gene_view &g) noexcept = 0;

  // reset
  virtual void
  reset(size_t num_population,
        const std::function<void(mut_gene_view &)> &init_function) noexcept = 0;

  // crossover
  virtual void crossover(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<crossover_function<const_gene_view, mut_gene_view>>
          &crossover_function) noexcept = 0;

  // mutation
  virtual void
  mutate(std::span<const size_t> mutate_list,
         const std::function<mutate_function<const_gene_view, mut_gene_view>>
             &mutate_function) noexcept = 0;

  virtual void crossover_and_mutate(
      std::span<const std::pair<size_t, size_t>> crossover_list,
      const std::function<crossover_function<const_gene_view, mut_gene_view>>
          &crossover_function,
      std::span<const size_t> mutate_list,
      const std::function<mutate_function<const_gene_view, mut_gene_view>>
          &mutate_function) noexcept {
    this->crossover(crossover_list, crossover_function);
    this->mutate(mutate_list, mutate_function);
  }

  /// selection, returns index LUT from new population to old population
  virtual std::vector<size_t>
  select(std::span<const uint16_t> LUT_selected_count) noexcept = 0;
};

template <class pop_t>
concept is_population =
    requires(pop_t *p, const pop_t *p_const) {
      typename pop_t::gene_type;
      typename pop_t::mut_gene_view_type;
      typename pop_t::const_gene_view_type;
      p_const->population_size();
      p->gene_at(0);
      p_const->gene_at(0);
    } and
    std::is_base_of_v<population_base<typename pop_t::gene_type,
                                      typename pop_t::mut_gene_view_type,
                                      typename pop_t::const_gene_view_type>,
                      pop_t>;

} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GABASE_H
