//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_GA_SYSTEM_HPP
#define CYKA_GA_SYSTEM_HPP
#include "fitness_computer.hpp"
#include "population_base.hpp"

namespace cyka::genetic {

/// A combination of population and fitness computer, it's not a complete type,
/// developer should inherit and implement fitness function
template <class population_t, size_t n_obj>
  requires is_population<population_t>
class GA_system_base
    : public population_t,
      public fitness_computer<n_obj,
                              typename population_t::const_gene_view_type> {
public:
  using population_type = population_t;
  using fitness_computer_type =
      fitness_computer<n_obj, typename population_t::const_gene_view_type>;

  static constexpr size_t objective_num = n_obj;

  [[nodiscard]] size_t population_size() const noexcept final {
    return population_t::population_size();
  }

  [[nodiscard]] population_t::const_gene_view_type
  gene_at(size_t idx) const noexcept override {
    return population_t::gene_at(idx);
  }

  //  [[nodiscard]] GA_system_base::fitness_matrix
  //  fitness_of_all() const noexcept override {
  //    return fitness_computer<
  //        n_obj, typename
  //        population_t::const_gene_view_type>::fitness_of_all();
  //  }
};

template <class GA_sys>
concept is_GA_system =
    requires() {
      typename GA_sys::fitness_type;
      typename GA_sys::fitness_matrix_type;
    } and std::is_base_of_v<typename GA_sys::population_type, GA_sys> and
    std::is_base_of_v<typename GA_sys::fitness_computer_type, GA_sys>;

/// GA system with changeable fitness function
template <class population_t, size_t n_obj>
class GA_system : public GA_system_base<population_t, n_obj> {
public:
  using base_t = GA_system_base<population_t, n_obj>;
  using typename base_t::const_gene_view_type;
  using typename base_t::fitness_matrix_type;
  using typename base_t::fitness_type;

  using fitness_fun_t =
      std::function<fitness_function<const_gene_view_type, fitness_type>>;
  GA_system() = delete;
  explicit GA_system(fitness_fun_t &&f) : fitness_fun{f} {}

protected:
  fitness_fun_t fitness_fun;

public:
  [[nodiscard]] const fitness_fun_t &fitness_function() const noexcept {
    return this->fitness_fun;
  }
  void set_fitness_function(fitness_fun_t &&f) noexcept {
    this->fitness_fun = f;
  }

  [[nodiscard]] fitness_type fitness_of(const_gene_view_type g) const noexcept {
    return this->fitness_fun(g);
  }

  //  [[nodiscard]] fitness_type fitness_of(size_t index) const noexcept
  //  override {
  //    return this->fitness_fun(this->gene_at(index));
  //  }
};

// template <class population_t>
// using GA_system_single_object = GA_system_base<population_t, 1>;

} // namespace cyka::genetic

#endif // CYKA_GA_SYSTEM_HPP
