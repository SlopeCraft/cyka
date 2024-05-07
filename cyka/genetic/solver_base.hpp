//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GA_H
#define HEURISTICFLOWR_GA_H

#include <cfloat>
#include <random>
#include <type_traits>

#include "GA_result.hpp"
#include "GA_system.hpp"
#include "crossover.hpp"
#include "fitness_computer.hpp"
#include "mutator.hpp"
#include "population_base.hpp"
#include "selector.hpp"
#include "single_object_selector.hpp"

namespace cyka::genetic {

struct GA_option {

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

namespace detail {

template <class GA_sys>
  requires is_GA_system<GA_sys>
class solver_base_impl {
public:
  using typename GA_sys::gene_type;
  using typename GA_sys::population_type;

  //  static_assert(
  //      std::is_base_of_v<selector_base<GA_sys::objective_num,
  //                                      typename
  //                                      selector::select_option_type>,
  //                        selector>);
  //  using gene_type = gene;
  //  using population_type = population_base<gene, mut_gene_view,
  //  const_gene_view>;

  explicit solver_base_impl(uint32_t rand_seed) : rand_engine{rand_seed} {}
  explicit solver_base_impl(std::random_device &rd) : rand_engine{rd()} {}
  solver_base_impl() : solver_base_impl(std::random_device{}){};

  [[nodiscard]] struct GA_option &GA_option() noexcept {
    return this->GA_option_;
  }
  void set_GA_option(struct GA_option &&opt) { this->GA_option_ = opt; }

  //  [[nodiscard]] const struct GA_option &GA_option() const noexcept {
  //    return this->option_;
  //  }

  auto &random_engine() noexcept { return this->rand_engine; }
  auto &random_engine() const noexcept { return this->rand_engine; }

protected:
  struct GA_option GA_option_;
  std::mt19937 rand_engine;

public:
};
} // namespace detail

template <class GA_sys, class selector, class crossover, class mutator>
  requires is_GA_system<GA_sys>
class solver_base : public detail::solver_base_impl<GA_sys>,
                    public selector,
                    public crossover,
                    public mutator {
public:
  using typename GA_sys::gene_type;
  using typename GA_sys::population_type;
  using result_type =
      GA_result<typename GA_sys::fitness_type, typename GA_sys::fitness_matrix>;

  static_assert(
      std::is_base_of_v<selector_base<GA_sys::objective_num,
                                      typename selector::select_option_type>,
                        selector>);

  explicit solver_base(selector &&s, crossover &&c, mutator &&m)
      : solver_base{detail::solver_base_impl<GA_sys>{}, s, c, m} {}

  solver_base() = delete;

  [[nodiscard]] virtual result_type optimize(population_type &pop) = 0;

protected:
  virtual void make_crossover_list(
      const GA_sys::fitness_matrix &,
      std::vector<std::pair<size_t, size_t>> &crossover_list) = 0;

  virtual void make_mutate_list(const GA_sys::fitness_matrix &,
                                std::vector<size_t> &mutate_list) = 0;
};

template <class GA_sys, class selector, class crossover, class mutator>
  requires is_GA_system<GA_sys>
class single_object_GA
    : public solver_base<GA_sys, selector, crossover, mutator> {
public:
  explicit single_object_GA(selector &&s, crossover &&c, mutator &&m)
      : single_object_GA{detail::solver_base_impl<GA_sys>{}, s, c, m} {}
  single_object_GA() : single_object_GA{selector{}, crossover{}, mutator{}} {};

  [[nodiscard]] single_object_GA::result_type
  optimize(single_object_GA::population_type &pop) {
    typename single_object_GA::result_type res;
    res.fitness_histroy.clear();
    res.fitness_histroy.reserve(this->GA_option().max_generations);

    double prev_best_fitness = DBL_MAX;
    size_t early_stop_counter = 0;
    std::vector<std::pair<size_t, size_t>> crossover_list;
    crossover_list.reserve(pop.pupulation_size() * 3);
    std::vector<size_t> mutate_list;
    mutate_list.reserve(pop.pupulation_size() * 3);

    size_t generations = 0;
    //    Eigen::ArrayX<uint16_t> selected_count;
    while (true) {
      // compute fitness
      const Eigen::Array<double, 1, Eigen::Dynamic> fitness_before_selection =
          pop.fitness_of_all();
      //      Eigen::Array<double, 1, Eigen::Dynamic> fitness_after_selection;
      // selection
      {
        const Eigen::ArrayX<uint16_t> selected_count =
            this->select(fitness_before_selection,
                         this->GA_option().population_size, this->rand_engine);

        const std::vector<size_t> index_LUT_new_2_old =
            pop.select(std::span<const uint16_t>{
                selected_count.data(), size_t(selected_count.size())});
        assert(index_LUT_new_2_old.size() == this->GA_option().population_size);

        Eigen::Array<double, 1, Eigen::Dynamic> fitness_after_selection =
            fitness_before_selection(index_LUT_new_2_old);
        res.fitness_histroy.emplace_back(std::move(fitness_after_selection));
      }

      const auto &fitness_after_selection = res.fitness_histroy.back();

      // find the best gene, decide whether to stop
      {
        ptrdiff_t best_gene_index = -1;
        const double cur_best_fitness =
            fitness_after_selection.maxCoeff(&best_gene_index);
        if (cur_best_fitness < prev_best_fitness) {
          early_stop_counter++;
        } else {
          early_stop_counter = 0;
        }
        prev_best_fitness = cur_best_fitness;

        if (generations >= this->GA_option().max_generations) {
          break;
        }
        generations++;
        if (early_stop_counter > this->GA_option().early_stop_times) {
          break;
        }
      }

      this->make_crossover_list(fitness_after_selection, crossover_list);
      this->make_mutate_list(fitness_after_selection, mutate_list);

      auto crossover_fun =
          [this](GA_sys::const_gene_view_type p1,
                 GA_sys::const_gene_view_type p2, GA_sys::mut_gene_view_type c1,
                 GA_sys::mut_gene_view_type c2, std::mt19937 &rand_engine) {
            this->crossover(p1, p2, c1, c2, rand_engine);
          };

      auto mutate_fun = [this](GA_sys::const_gene_view_type p,
                               GA_sys::mut_gene_view_type c,
                               std::mt19937 &rand_engine) {
        this->mutate(p, c, rand_engine);
      };

      pop.crossover_and_mutate(crossover_list, crossover_fun, mutate_list,
                               mutate_fun);
    }

    return res;
  }
};

} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GA_H
