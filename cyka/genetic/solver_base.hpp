//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GA_H
#define HEURISTICFLOWR_GA_H

#include <cfloat>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include "GA_result.hpp"
#include "GA_system.hpp"
#include "crossover.hpp"
#include "cyka/utils/size_t_literal.hpp"
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
  using gene_type = GA_sys::gene_type;
  using population_type = GA_sys::population_type;
  using GA_system_type = GA_sys;

  //  static_assert(
  //      std::is_base_of_v<selector_base<GA_sys::objective_num,
  //                                      typename
  //                                      selector::select_option_type>,
  //                        selector>);
  //  using gene_type = gene;
  //  using population_type = population_base<gene, mut_gene_view,
  //  const_gene_view>;

  explicit solver_base_impl(uint32_t rand_seed) : rand_engine{rand_seed} {}
  solver_base_impl() = delete;
  //  explicit solver_base_impl(std::random_device &rd) : rand_engine{rd()} {}
  //  solver_base_impl() : solver_base_impl(std::random_device{}()){};
  virtual ~solver_base_impl() = default;

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
  requires is_GA_system<GA_sys> and is_selector<selector> and
               is_crossover<crossover> and is_mutator<mutator>
class solver_base : public detail::solver_base_impl<GA_sys>,
                    public selector,
                    public crossover,
                    public mutator {
public:
  using result_type = GA_result<typename GA_sys::fitness_type,
                                typename GA_sys::fitness_matrix_type>;

  static_assert(
      std::is_base_of_v<selector_base<GA_sys::objective_num,
                                      typename selector::select_option_type>,
                        selector>);

  explicit solver_base(uint32_t rand_seed, selector &&s, crossover &&c,
                       mutator &&m)
      : detail::solver_base_impl<GA_sys>{rand_seed}, selector{s}, crossover{c},
        mutator{m} {}

  solver_base() = delete;

  [[nodiscard]] virtual result_type
  optimize(typename solver_base::GA_system_type &pop) = 0;

protected:
  virtual void
  make_crossover_list(typename solver_base::population_type &pop,
                      const typename GA_sys::fitness_matrix_type &,
                      std::vector<std::pair<size_t, size_t>> &crossover_list) {
    std::uniform_real_distribution<double> rand{0, 1};
    std::vector<size_t> queue;
    queue.reserve(pop.population_size());
    for (auto i = 0zu; i < pop.population_size(); i++) {
      if (rand(this->random_engine()) >=
          this->GA_option().crossover_probability) {
        queue.emplace_back(i);
      }
    }

    std::shuffle(queue.begin(), queue.end(), this->random_engine());
    if (queue.size() % 2 == 1) {
      queue.pop_back();
    }
    assert(queue.size() % 2 == 0);

    crossover_list.clear();
    crossover_list.reserve(queue.size() / 2);

    for (auto idx = 0zu; idx < queue.size(); idx += 2) {
      const auto j = idx + 1;
      assert(j < queue.size());
      crossover_list.emplace_back(idx, j);
    }
  }

  virtual void make_mutate_list(typename solver_base::population_type &pop,
                                const typename GA_sys::fitness_matrix_type &,
                                std::vector<size_t> &mutate_list) {
    mutate_list.clear();
    mutate_list.reserve(pop.population_size());
    std::uniform_real_distribution<double> rand{0, 1};

    for (auto idx = 0zu; idx < pop.population_size(); idx++) {
      if (rand(this->random_engine()) >= this->GA_option().mutate_probability) {
        mutate_list.emplace_back(idx);
      }
    }
  }
};

template <class GA_sys, class selector, class crossover, class mutator>
  requires is_GA_system<GA_sys>
class single_object_GA
    : public solver_base<GA_sys, selector, crossover, mutator> {
public:
  explicit single_object_GA(uint32_t rand_seed, selector &&s, crossover &&c,
                            mutator &&m)
      : solver_base<GA_sys, selector, crossover, mutator>{
            rand_seed, std::move(s), std::move(c), std::move(m)} {}
  explicit single_object_GA(uint32_t rand_seed)
      : single_object_GA{rand_seed, selector{}, crossover{}, mutator{}} {};

  single_object_GA() : single_object_GA{10} {}

  using typename solver_base<GA_sys, selector, crossover, mutator>::result_type;

  [[nodiscard]] result_type
  optimize(typename single_object_GA::GA_system_type &pop) override {
    typename single_object_GA::result_type res;
    res.fitness_history.clear();
    res.fitness_history.reserve(this->GA_option().max_generations);

    double prev_best_fitness = std::numeric_limits<double>::max();
    size_t early_stop_counter = 0;
    std::vector<std::pair<size_t, size_t>> crossover_list;
    crossover_list.reserve(pop.population_size() * 3);
    std::vector<size_t> mutate_list;
    mutate_list.reserve(pop.population_size() * 3);

    size_t generations = 0;
    //    Eigen::ArrayX<uint16_t> selected_count;
    while (true) {
      // compute fitness
      const Eigen::Array<double, 1, Eigen::Dynamic> fitness_before_selection =
          pop.fitness_of_all();
      {
        Eigen::ArrayX<uint16_t> selected_count;
        this->select(fitness_before_selection,
                     this->GA_option().population_size, selected_count,
                     this->rand_engine);

        const std::vector<size_t> index_LUT_new_2_old =
            pop.select(std::span<const uint16_t>{
                selected_count.data(), size_t(selected_count.size())});
        assert(index_LUT_new_2_old.size() == this->GA_option().population_size);

        Eigen::Array<double, 1, Eigen::Dynamic> fitness_after_selection =
            fitness_before_selection(index_LUT_new_2_old);
        res.fitness_history.emplace_back(std::move(fitness_after_selection));
      }

      const auto &fitness_after_selection =
          res.fitness_history.back().population_fitness;

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

      this->make_crossover_list(pop, fitness_after_selection, crossover_list);
      this->make_mutate_list(pop, fitness_after_selection, mutate_list);

      auto crossover_fun =
          [this](GA_sys::const_gene_view_type p1,
                 GA_sys::const_gene_view_type p2, GA_sys::mut_gene_view_type c1,
                                  GA_sys::mut_gene_view_type c2) {
        this->crossover(p1, p2, c1, c2, this->random_engine());
      };

      auto mutate_fun = [this](GA_sys::const_gene_view_type p,
                               GA_sys::mut_gene_view_type c) {
        this->mutate(p, c, this->random_engine());
      };

      pop.crossover_and_mutate(crossover_list, crossover_fun, mutate_list,
                               mutate_fun);
    }

    return res;
  }

protected:

};

} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GA_H
