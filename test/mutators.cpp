#include <Eigen/Dense>
#include <cyka/genetic/mutator.hpp>

#include <iomanip>
#include <iostream>

namespace cg = cyka::genetic;

constexpr size_t dim = 10;

void test_float_mutators(std::mt19937 &, bool output) noexcept;

void test_u8_mutators(std::mt19937 &, bool output) noexcept;

void test_bool_mutators(std::mt19937 &, bool output) noexcept;

int main(int, char **) {
  std::mt19937 rand{std::random_device{}()};
  for (int i = 0; i < 1000; i++) {
    test_float_mutators(rand, i == 0);
    test_u8_mutators(rand, i == 0);
    test_bool_mutators(rand, i == 0);
  }
}

void test_float_mutators(std::mt19937 &rand, bool output) noexcept {
  using vxf = Eigen::ArrayXf;
  using mxf_mut = Eigen::Map<Eigen::ArrayXf>;
  using mxf_const = Eigen::Map<const Eigen::ArrayXf>;

  const Eigen::ArrayXf p = Eigen::ArrayXf::LinSpaced(dim, 0.0, float(dim - 1));

  Eigen::ArrayXf c;

  auto print = [&p, output](const char *name, auto c) {
    if (not output) {
      return;
    }
    std::cout << "\n\n" << name << " mutate: ";
    std::cout << "\np1 = ";
    std::cout << std::fixed << std::setprecision(2);
    for (auto f : p) {
      if (f >= 0) {
        std::cout << ' ';
      }
      std::cout << f << ' ';
    }
    std::cout << "\nc1 = ";
    for (auto f : c) {
      if (f >= 0) {
        std::cout << ' ';
      }
      std::cout << f << ' ';
    }
    std::cout.clear();
  };

  {
    c.resize(0);
    cg::arithmetic_mutator<vxf &, const vxf &> m;
    {
      decltype(m)::mutate_option_type opt;
      opt.lower_bound = Eigen::ArrayXf::Constant(dim, -10);
      opt.upper_bound = Eigen::ArrayXf::Constant(dim, 10);
      opt.step_max = Eigen::ArrayXf::Constant(dim, 0.5);
      m.set_mutate_option(std::move(opt));
    }
    m.mutate(p, c, rand);
    print("[ref] arithmetic", c);
  }
  {
    c.resize(0);
    cg::single_point_arithmetic_mutator<vxf &, const vxf &> m;
    {
      decltype(m)::mutate_option_type opt;
      opt.lower_bound = Eigen::ArrayXf::Constant(dim, -10);
      opt.upper_bound = Eigen::ArrayXf::Constant(dim, 10);
      opt.step_max = Eigen::ArrayXf::Constant(dim, 0.5);
      m.set_mutate_option(std::move(opt));
    }
    m.mutate(p, c, rand);
    print("[ref] single point arithmetic", c);
  }

  mxf_const pm{p.data(), p.size()};
  c.setZero(p.size());
  mxf_mut cm{c.data(), c.size()};
  {
    cg::arithmetic_mutator<mxf_mut, mxf_const> m;
    {
      decltype(m)::mutate_option_type opt;
      opt.lower_bound = Eigen::ArrayXf::Constant(dim, -10);
      opt.upper_bound = Eigen::ArrayXf::Constant(dim, 10);
      opt.step_max = Eigen::ArrayXf::Constant(dim, 0.5);
      m.set_mutate_option(std::move(opt));
    }
    m.mutate(pm, cm, rand);
    print("[map] arithmetic", cm);
  }
  {
    cg::single_point_arithmetic_mutator<mxf_mut, mxf_const> m;
    {
      decltype(m)::mutate_option_type opt;
      opt.lower_bound = Eigen::ArrayXf::Constant(dim, -10);
      opt.upper_bound = Eigen::ArrayXf::Constant(dim, 10);
      opt.step_max = Eigen::ArrayXf::Constant(dim, 0.5);
      m.set_mutate_option(std::move(opt));
    }
    m.mutate(pm, cm, rand);
    print("[map] single point arithmetic", cm);
  }
}

void test_u8_mutators(std::mt19937 &rand, bool output) noexcept {
  using vxu = Eigen::ArrayX<uint8_t>;
  using mxu_mut = Eigen::Map<vxu>;
  using mxu_const = Eigen::Map<const vxu>;

  const vxu p = vxu::LinSpaced(dim, 0, uint8_t(dim - 1));
  vxu c;

  auto print = [&p, output](const char *name, auto c) {
    if (not output) {
      return;
    }
    std::cout << "\n\n" << name << " mutate: ";
    std::cout << "\np1 = ";
    std::cout << std::fixed << std::setprecision(2);
    for (auto f : p) {
      if (f >= 0) {
        std::cout << ' ';
      }
      std::cout << int(f) << ' ';
    }
    std::cout << "\nc1 = ";
    for (auto f : c) {
      if (f >= 0) {
        std::cout << ' ';
      }
      std::cout << int(f) << ' ';
    }
    std::cout.clear();
  };

  {
    c.resize(0);
    cg::single_point_discrete_mutator<vxu &, const vxu &> m;
    m.set_mutate_option(decltype(m)::mutate_option_type{
        .lower_bound = Eigen::ArrayX<uint8_t>::Constant(dim, 0),
        .upper_bound = Eigen::ArrayX<uint8_t>::Constant(dim, dim - 1),
    });

    m.mutate(p, c, rand);
    print("[ref] single point discrete", c);
    assert((p not_eq c).any());
  }

  c.resize(p.size());
  const mxu_const pm{p.data(), p.size()};
  mxu_mut cm{c.data(), c.size()};

  {
    cg::single_point_discrete_mutator<mxu_mut, mxu_const> m;
    m.set_mutate_option(decltype(m)::mutate_option_type{
        .lower_bound = Eigen::ArrayX<uint8_t>::Constant(dim, 0),
        .upper_bound = Eigen::ArrayX<uint8_t>::Constant(dim, dim - 1),
    });

    m.mutate(pm, cm, rand);
    print("[map] single point discrete", cm);
    assert((pm not_eq cm).any());
  }
}

void test_bool_mutators(std::mt19937 &rand, bool output) noexcept {
  using vxb = Eigen::ArrayX<bool>;
  using mxb_mut = Eigen::Map<vxb>;
  using mxb_const = Eigen::Map<const vxb>;

  const vxb p = vxb::Random(dim);
  vxb c;

  auto print = [&p, output](const char *name, auto c) {
    if (not output) {
      return;
    }
    std::cout << "\n\n" << name << " mutate: ";
    std::cout << "\np1 = ";
    std::cout << std::fixed << std::setprecision(2);
    for (auto f : p) {
      if (f >= 0) {
        std::cout << ' ';
      }
      std::cout << int(f) << ' ';
    }
    std::cout << "\nc1 = ";
    for (auto f : c) {
      if (f >= 0) {
        std::cout << ' ';
      }
      std::cout << int(f) << ' ';
    }
    std::cout.clear();
  };

  {
    c.resize(0);
    cg::single_point_boolean_mutator<vxb &, const vxb &> m;
    m.mutate(p, c, rand);
    print("[ref] single point boolean", c);
    assert((p not_eq c).any());
  }

  c.resize(p.size());
  const mxb_const pm{p.data(), p.size()};
  mxb_mut cm{c.data(), c.size()};

  {
    cg::single_point_boolean_mutator<mxb_mut, mxb_const> m;
    m.mutate(pm, cm, rand);
    print("[map] single point boolean", cm);
    assert((pm not_eq cm).any());
  }
}