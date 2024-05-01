//
// Created by Joseph on 2024/4/29.
//

#ifndef HEURISTICFLOWR_GA_H
#define HEURISTICFLOWR_GA_H

#include "population.hpp"

namespace cyka::genetic {
template <class gene, class mut_gene_view, class const_gene_view, size_t n_obj>
class GA_base {
public:
  using gene_type = gene;
  using population_type = population<gene, mut_gene_view, const_gene_view>;

  [[nodiscard]] struct option &option() noexcept { return this->option_; }

  [[nodiscard]] const struct option &option() const noexcept {
    return this->option_;
  }

protected:
  struct option option_;

  virtual void crossover(population_type &pop) noexcept = 0;

  virtual void mutate(population_type &pop) noexcept = 0;
};
} // namespace cyka::genetic

#endif // HEURISTICFLOWR_GA_H
