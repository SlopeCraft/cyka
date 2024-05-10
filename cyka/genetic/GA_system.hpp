//
// Created by Joseph on 2024/5/2.
//

#ifndef CYKA_GA_SYSTEM_HPP
#define CYKA_GA_SYSTEM_HPP
#include "loss_computer.hpp"
#include "population_base.hpp"

namespace cyka::genetic {

/// A combination of population and loss computer, it's not a complete type,
/// developer should inherit and implement loss function
template <class population_t, size_t n_obj>
  requires is_population<population_t>
class GA_system_base
    : public population_t,
      public loss_computer<n_obj,
                              typename population_t::const_gene_view_type> {
public:
  using population_type = population_t;
  using loss_computer_type =
      loss_computer<n_obj, typename population_t::const_gene_view_type>;

  static constexpr size_t objective_num = n_obj;

  [[nodiscard]] size_t population_size() const noexcept final {
    return population_t::population_size();
  }

  [[nodiscard]] population_t::const_gene_view_type
  gene_at(size_t idx) const noexcept override {
    return population_t::gene_at(idx);
  }

  //  [[nodiscard]] GA_system_base::loss_matrix
  //  loss_of_all() const noexcept override {
  //    return loss_computer<
  //        n_obj, typename
  //        population_t::const_gene_view_type>::loss_of_all();
  //  }
};

template <class GA_sys>
concept is_GA_system =
    requires() {
      typename GA_sys::loss_type;
      typename GA_sys::loss_matrix_type;
    } and std::is_base_of_v<typename GA_sys::population_type, GA_sys> and
    std::is_base_of_v<typename GA_sys::loss_computer_type, GA_sys>;

/// GA system with changeable loss function
template <class population_t, size_t n_obj>
class GA_system : public GA_system_base<population_t, n_obj> {
public:
  using base_t = GA_system_base<population_t, n_obj>;
  using typename base_t::const_gene_view_type;
  using typename base_t::loss_matrix_type;
  using typename base_t::loss_type;

  using loss_fun_t =
      std::function<loss_function<const_gene_view_type, loss_type>>;
  GA_system() = delete;
  explicit GA_system(loss_fun_t &&f) : loss_fun{f} {}

protected:
  loss_fun_t loss_fun;

public:
  [[nodiscard]] const loss_fun_t &loss_function() const noexcept {
    return this->loss_fun;
  }
  void set_loss_function(loss_fun_t &&f) noexcept {
    this->loss_fun = f;
  }

  [[nodiscard]] loss_type loss_of(const_gene_view_type g) const noexcept {
    return this->loss_fun(g);
  }

  //  [[nodiscard]] loss_type loss_of(size_t index) const noexcept
  //  override {
  //    return this->loss_fun(this->gene_at(index));
  //  }
};

// template <class population_t>
// using GA_system_single_object = GA_system_base<population_t, 1>;

} // namespace cyka::genetic

#endif // CYKA_GA_SYSTEM_HPP
