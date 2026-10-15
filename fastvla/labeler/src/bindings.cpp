// pybind11 bindings for the ported GJK core. Same return values as the Python
// functions in convex_distance.py, so the two can be compared pair for pair.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "gjk.hpp"
#include "labeler.hpp"

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

// ---------------------------------------------------------------- Labeler ----------
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

void add_geom_py(fastgjk::Labeler& L, int gid, const Arr& verts, double pad) {
  int n = 0;
  const double* v = vdata(verts, n, "verts");
  L.add_geom(gid, v, n, pad);
}

void set_union_hull_py(fastgjk::Labeler& L, int key, const Arr& verts) {
  int n = 0;
  const double* v = vdata(verts, n, "verts");
  L.set_union_hull(key, v, n);
}

py::dict min_clearance_py(fastgjk::Labeler& L,
                          const IArr& actor_ids, const Arr& actor_xmat, const Arr& actor_xpos,
                          const IArr& piece_ids, const Arr& piece_xmat, const Arr& piece_xpos,
                          int union_key, const Arr& union_xmat, const Arr& union_xpos,
                          double best_init, int n_threads) {
  const auto ab = actor_ids.request();
  const auto pb = piece_ids.request();
  const int na = static_cast<int>(ab.size);
  const int np_ = static_cast<int>(pb.size);
  const auto axm = actor_xmat.request(), axp = actor_xpos.request();
  const auto pxm = piece_xmat.request(), pxp = piece_xpos.request();
  if (axm.size != static_cast<ssize_t>(na) * 9 || axp.size != static_cast<ssize_t>(na) * 3) {
    throw std::invalid_argument("actor_xmat must be (n,9) and actor_xpos (n,3)");
  }
  if (pxm.size != static_cast<ssize_t>(np_) * 9 || pxp.size != static_cast<ssize_t>(np_) * 3) {
    throw std::invalid_argument("piece_xmat must be (n,9) and piece_xpos (n,3)");
  }
  fastgjk::Result r;
  {
    py::gil_scoped_release release;
    r = L.min_clearance(static_cast<const int*>(ab.ptr), na,
                        static_cast<const double*>(axm.ptr),
                        static_cast<const double*>(axp.ptr),
                        static_cast<const int*>(pb.ptr), np_,
                        static_cast<const double*>(pxm.ptr),
                        static_cast<const double*>(pxp.ptr),
                        union_key,
                        static_cast<const double*>(union_xmat.request().ptr),
                        static_cast<const double*>(union_xpos.request().ptr),
                        best_init, n_threads);
  }
  py::dict out;
  out["best"] = r.best;
  out["actor_idx"] = r.actor_idx;
  out["piece_idx"] = r.piece_idx;
  out["exact_pairs"] = r.exact_pairs;
  out["union_hull_calls"] = r.union_hull_calls;
  return out;
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

  py::class_<fastgjk::Labeler>(m, "Labeler",
      "Branch-and-bound clearance with the local hulls cached on the C++ side.")
      .def(py::init<>())
      .def("add_geom", &add_geom_py, py::arg("gid"), py::arg("verts"), py::arg("pad"),
           "Register a geom's local-frame hull and swept-radius pad.")
      .def("has_geom", &fastgjk::Labeler::has_geom, py::arg("gid"))
      .def("n_geoms", &fastgjk::Labeler::n_geoms)
      .def("set_union_hull", &set_union_hull_py, py::arg("key"), py::arg("verts"),
           "Register an obstacle's union convex hull, in its body frame.")
      .def("has_union", &fastgjk::Labeler::has_union, py::arg("key"))
      .def("min_clearance", &min_clearance_py,
           py::arg("actor_ids"), py::arg("actor_xmat"), py::arg("actor_xpos"),
           py::arg("piece_ids"), py::arg("piece_xmat"), py::arg("piece_xpos"),
           py::arg("union_key"), py::arg("union_xmat"), py::arg("union_xpos"),
           py::arg("best_init"), py::arg("n_threads") = 1,
           "One clearance query for one obstacle; returns best/actor_idx/piece_idx.");
}
