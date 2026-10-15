// pybind11 bindings for the ported GJK core. Same return values as the Python
// functions in convex_distance.py, so the two can be compared pair for pair.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "gjk.hpp"

namespace py = pybind11;

namespace {

// Accept any (n,3) float64 array; copy to contiguous if it is not already.
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

const double* vdata(const Arr& a, int& n, const char* what) {
  const auto buf = a.request();
  if (buf.ndim != 2 || buf.shape[1] != 3) {
    throw std::invalid_argument(std::string(what) + " must have shape (n, 3)");
  }
  if (buf.shape[0] == 0) {
    throw std::invalid_argument(std::string(what) + " must have at least one vertex");
  }
  n = static_cast<int>(buf.shape[0]);
  return static_cast<const double*>(buf.ptr);
}

double gjk_distance_py(const Arr& A, const Arr& B, int max_iter, double tol) {
  int na = 0, nb = 0;
  const double* a = vdata(A, na, "A");
  const double* b = vdata(B, nb, "B");
  py::gil_scoped_release release;
  return fastgjk::gjk_distance(a, na, b, nb, max_iter, tol);
}

double distance_padded_py(const Arr& A, double padA, const Arr& B, double padB) {
  int na = 0, nb = 0;
  const double* a = vdata(A, na, "A");
  const double* b = vdata(B, nb, "B");
  py::gil_scoped_release release;
  return fastgjk::distance_padded(a, na, padA, b, nb, padB);
}

// Closest point on conv(W) to the origin, exposed so the subset search can be
// compared against _closest_on_simplex directly rather than only through gjk_distance.
py::tuple closest_on_simplex_py(const Arr& W) {
  int n = 0;
  const double* w = vdata(W, n, "W");
  if (n > 4) throw std::invalid_argument("W must have at most 4 rows");
  double Wc[4][3];
  for (int i = 0; i < n; ++i) {
    Wc[i][0] = w[i * 3]; Wc[i][1] = w[i * 3 + 1]; Wc[i][2] = w[i * 3 + 2];
  }
  double v[3];
  int idx[4];
  const int k = fastgjk::closest_on_simplex(Wc, n, v, idx);
  auto out = py::array_t<double>(3);
  auto* o = static_cast<double*>(out.request().ptr);
  o[0] = v[0]; o[1] = v[1]; o[2] = v[2];
  std::vector<int> keep(idx, idx + k);
  return py::make_tuple(out, keep);
}

}  // namespace

PYBIND11_MODULE(fastgjk, m) {
  m.doc() = "C++17 port of the LIBERO-Safety GJK clearance core (fastvla M2)";
  m.def("gjk_distance", &gjk_distance_py,
        py::arg("A"), py::arg("B"), py::arg("max_iter") = 64, py::arg("tol") = 1e-12,
        "Distance between conv(A) and conv(B); 0.0 if they intersect.");
  m.def("distance_padded", &distance_padded_py,
        py::arg("A"), py::arg("pad_a"), py::arg("B"), py::arg("pad_b"),
        "gjk_distance(A, B) - pad_a - pad_b, clamped at 0.");
  m.def("closest_on_simplex", &closest_on_simplex_py, py::arg("W"),
        "Closest point to the origin on conv(W) and the surviving subset indices.");
}
