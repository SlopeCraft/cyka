//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GABASE_H
#define HEURISTICFLOWR_GABASE_H

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

struct option {

  /**
   * \brief Size of the population. Default value is 100
   *
   */
  size_t population_size{100};

  /**
   * \brief GA will stop once best solution hasn't been improved for continuous
   * maxFailTimes generations. Default value is 100.
   *
   * \note This member doesn't works for MOGA solvers like NSGA2 and NSGA3 since
   * there's no proper way to estimate if the PF hasn't been changing for
   * generations.
   *
   */
  uint64_t early_stop_times{50};

  /**
   * \brief Maximum generation. GA will stop once reached this limitation.
   * Default value is 300
   *
   */
  uint64_t max_generations{300};

  /**
   * \brief Probability of a non-elite individual to join crossover. Default
   * value is 80%
   *
   */
  double crossover_probability{0.8};

  /**
   * \brief Probability of a non-elite individual to get mutated. Default value
   * is 5%
   */
  double mutate_probability{0.05};
};

// fixed multiple objectives
template <size_t n_objectives> class population_base {
public:
  using fitness_type = Eigen::Array<double, n_objectives, 1>;

  virtual ~population_base() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept {
    return n_objectives;
  }
};

template <> class population_base<1> {
public:
  using fitness_type = double;

  virtual ~population_base() = default;

  [[nodiscard]] constexpr size_t num_objectives() const noexcept { return 1; }
};

template <> class population_base<0> {
public:
  using fitness_type = Eigen::ArrayXd;

  virtual ~population_base() = default;

  [[nodiscard]] virtual size_t num_objectives() const noexcept = 0;
};

template <class const_gene_view, class fitness>
using fitness_function = fitness(const_gene_view);

template <class const_gene_view, class mut_gene_view>
using crossover_function = void(const_gene_view a, const_gene_view b,
                                mut_gene_view c, mut_gene_view d);

template <class const_gene_view, class mut_gene_view>
using mutate_function = void(const_gene_view src, mut_gene_view dest);

template <class gene, class mut_gene_view, class const_gene_view,
          size_t n_objectives>
  requires std::move_constructible<gene> && std::is_move_assignable_v<gene> &&
           std::is_default_constructible_v<gene> &&
           //           std::is_constructible_v<mut_gene_view, gene> &&
           //           std::is_constructible_v<const_gene_view, gene> &&
           //           std::is_constructible_v<const_gene_view, mut_gene_view>
           //           &&
           std::is_assignable_v<gene, mut_gene_view> &&
           std::is_assignable_v<gene, const_gene_view> &&
           std::is_assignable_v<mut_gene_view, const_gene_view>
class population : public population_base<n_objectives> {
public:
  using gene_type = gene;
  using mut_gene_view_type = mut_gene_view;
  using const_gene_view_type = const_gene_view;

  using base_t = population_base<n_objectives>;
  using fitness_type = base_t::fitness_type;
  using fitness_matrix = Eigen::Array<double, n_objectives, Eigen::Dynamic>;

  virtual ~population() = default;

  [[nodiscard]] virtual size_t population_size() const noexcept = 0;

  [[nodiscard]] virtual mut_gene_view gene_at(size_t index) noexcept = 0;
  [[nodiscard]] virtual const_gene_view
  gene_at(size_t index) const noexcept = 0;

  virtual void set_gene_at(size_t index, const_gene_view g) noexcept = 0;

  // reset
  virtual void
  reset(size_t num_population,
        const std::function<void(mut_gene_view)> &init_function) noexcept = 0;

  // compute fitness
  [[nodiscard]] virtual fitness_type
  fitness_of(size_t index) const noexcept = 0;

  virtual void fitness_of_all(fitness_matrix &result) const noexcept {
    result.setZero(this->num_objectives(), this->population_size());
    for (size_t idx = 0; idx < this->population_size(); idx++) {
      if constexpr (n_objectives == 1) {
        result(idx) = this->fitness_of(idx);
      } else {
        result.col(idx) = this->fitness_of(idx);
      }
    }
  }

  [[nodiscard]] virtual fitness_matrix fitness_of_all() const noexcept {
    fitness_matrix result;
    this->fitness_of_all(result);
    return result;
  }

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

  // selection
  virtual void select(std::span<const bool> LUT_is_selected) noexcept = 0;
};



} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GABASE_H
