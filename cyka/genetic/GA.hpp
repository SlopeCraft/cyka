//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GA_H
#define HEURISTICFLOWR_GA_H

#include <random>

#include "GA_system.hpp"
#include "population_base.hpp"

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

template <class GA_sys>
  requires is_GA_system<GA_sys>
class GA_base {
public:
  using typename GA_sys::gene_type;
  using typename GA_sys::population_type;
  //  using gene_type = gene;
  //  using population_type = population_base<gene, mut_gene_view,
  //  const_gene_view>;

  explicit GA_base(uint32_t rand_seed) : rand_engine{rand_seed} {}
  explicit GA_base(std::random_device &rd) : rand_engine{rd()} {}
  GA_base() : GA_base(std::random_device{}){};

  [[nodiscard]] struct option &option() noexcept { return this->option_; }

  [[nodiscard]] const struct option &option() const noexcept {
    return this->option_;
  }

  auto &random_engine() noexcept { return this->rand_engine; }
  auto &random_engine() const noexcept { return this->rand_engine; }

protected:
  struct option option_;
  std::mt19937 rand_engine;

  virtual void crossover(population_type &pop) noexcept = 0;

  virtual void mutate(population_type &pop) noexcept = 0;
};
} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GA_H
