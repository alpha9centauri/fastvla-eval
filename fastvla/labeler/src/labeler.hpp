// Branch-and-bound clearance over convex hulls -- the C++ side of
// SafetyMonitor._update_clearance (physics_safety.py md5 78a026bb7f8ce81d2e1e6e3589bb04e2).
//
// The Python version crosses into C++ once per hull pair, which at ~101 exact pairs and
// ~67 union-hull calls per clearance call is ~168 crossings plus the numpy bound
// arithmetic and two argsorts in between. This holds the local hull vertices on the C++
// side, keyed by geom id, so one call per obstacle does transform + prune + GJK.
//
// The algorithm is unchanged, including both exact pruning levels and the order they
// visit things in:
//   1. distance(actor, obstacle union hull) is an exact lower bound on the distance to
//      any piece; actors are visited nearest-bound first and the scan stops when an
//      actor's bound is no better than the best distance found so far.
//   2. within a surviving actor, bounding-sphere lower bounds order the pieces and cut
//      the scan the same way.
// Both can only skip pairs that provably cannot hold the minimum, so the result is the
// same regardless of visiting order. Ordering is made deterministic here with
// stable_sort on (bound, index); numpy's default argsort is introsort and is not stable,
// so on an exact tie the reference itself has no defined order.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "gjk.hpp"

namespace fastgjk {

// Below this many live pieces a parallel region costs more than it saves. Measured in
// step 4c; see DESIGN.md.
constexpr int kThreadCutoff = 32;

struct Hull {
  std::vector<double> v;   // local-frame vertices, n*3 row-major
  int n = 0;
  double pad = 0.0;
  double c[3] = {0, 0, 0}; // tight bounding-sphere centre, local
  double r = 0.0;          // tight bounding-sphere radius, including pad
};

struct Result {
  double best;
  int actor_idx;   // index into the actor list passed in, -1 if nothing beat best_init
  int piece_idx;   // index into the piece list passed in, -1 likewise
  int64_t exact_pairs;      // GJK calls in stage 5
  int64_t union_hull_calls; // GJK calls in stage 3
};

// Transform n local vertices by R (row-major 3x3) and p, into out (n*3).
inline void transform(const double* v, int n, const double* R, const double* p,
                      double* out) {
  for (int i = 0; i < n; ++i) {
    const double x = v[i * 3], y = v[i * 3 + 1], z = v[i * 3 + 2];
    out[i * 3]     = R[0] * x + R[1] * y + R[2] * z + p[0];
    out[i * 3 + 1] = R[3] * x + R[4] * y + R[5] * z + p[1];
    out[i * 3 + 2] = R[6] * x + R[7] * y + R[8] * z + p[2];
  }
}

class Labeler {
 public:
  // Register a geom's local hull once. Mirrors GeomHulls.local() + .bound(): the bound is
  // centred on the hull's own AABB centre, not the geom frame, and the pad is added to
  // the radius.
  void add_geom(int gid, const double* verts, int n, double pad) {
    Hull h;
    h.n = n;
    h.pad = pad;
    h.v.assign(verts, verts + static_cast<size_t>(n) * 3);
    double mn[3] = {verts[0], verts[1], verts[2]};
    double mx[3] = {verts[0], verts[1], verts[2]};
    for (int i = 1; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        mn[k] = std::min(mn[k], verts[i * 3 + k]);
        mx[k] = std::max(mx[k], verts[i * 3 + k]);
      }
    }
    for (int k = 0; k < 3; ++k) h.c[k] = (mx[k] + mn[k]) * 0.5;
    double r2max = 0.0;
    for (int i = 0; i < n; ++i) {
      double d2 = 0.0;
      for (int k = 0; k < 3; ++k) {
        const double d = verts[i * 3 + k] - h.c[k];
        d2 += d * d;
      }
      r2max = std::max(r2max, d2);
    }
    h.r = std::sqrt(r2max) + pad;
    hulls_[gid] = std::move(h);
  }

  bool has_geom(int gid) const { return hulls_.count(gid) != 0; }
  size_t n_geoms() const { return hulls_.size(); }

  // The obstacle's union convex hull, in its body frame.
  void set_union_hull(int key, const double* verts, int n) {
    std::vector<double> v(verts, verts + static_cast<size_t>(n) * 3);
    union_[key] = std::move(v);
    union_n_[key] = n;
  }
  bool has_union(int key) const { return union_.count(key) != 0; }

  // One clearance query for one obstacle.
  //   actor_ids / piece_ids   geom ids, previously registered with add_geom
  //   *_xmat                  row-major 3x3 per geom, world orientation (data.geom_xmat)
  //   *_xpos                  world position per geom (data.geom_xpos)
  //   union_key               which union hull, or -1 for none (multi-body obstacle)
  //   union_xmat / union_xpos the obstacle BODY frame (data.xmat / data.xpos)
  //   best_init               incoming best, i.e. distmax
  //   n_threads               1 = sequential; >1 threads the piece scan of one actor
  Result min_clearance(const int* actor_ids, int n_actors,
                       const double* actor_xmat, const double* actor_xpos,
                       const int* piece_ids, int n_pieces,
                       const double* piece_xmat, const double* piece_xpos,
                       int union_key, const double* union_xmat, const double* union_xpos,
                       double best_init, int n_threads);

 private:
  std::unordered_map<int, Hull> hulls_;
  std::unordered_map<int, std::vector<double>> union_;
  std::unordered_map<int, int> union_n_;
  // scratch, reused across calls so a query allocates nothing steady-state
  std::vector<double> aw_, pw_, uw_;
  std::vector<int> aoff_, poff_;
  std::vector<double> ac_, ar_, pc_, pr_;
};

inline Result Labeler::min_clearance(
    const int* actor_ids, int n_actors, const double* actor_xmat, const double* actor_xpos,
    const int* piece_ids, int n_pieces, const double* piece_xmat, const double* piece_xpos,
    int union_key, const double* union_xmat, const double* union_xpos,
    double best_init, int n_threads) {
  Result out{best_init, -1, -1, 0, 0};
  if (n_actors == 0 || n_pieces == 0) return out;

  // ---- stage 1 + 5b: transform actor hulls to world once, with their bounds ----------
  aoff_.assign(n_actors + 1, 0);
  for (int i = 0; i < n_actors; ++i) aoff_[i + 1] = aoff_[i] + hulls_.at(actor_ids[i]).n;
  aw_.resize(static_cast<size_t>(aoff_[n_actors]) * 3);
  ac_.resize(static_cast<size_t>(n_actors) * 3);
  ar_.resize(n_actors);
  for (int i = 0; i < n_actors; ++i) {
    const Hull& h = hulls_.at(actor_ids[i]);
    transform(h.v.data(), h.n, actor_xmat + i * 9, actor_xpos + i * 3,
              aw_.data() + static_cast<size_t>(aoff_[i]) * 3);
    const double* R = actor_xmat + i * 9;
    const double* p = actor_xpos + i * 3;
    ac_[i * 3]     = R[0] * h.c[0] + R[1] * h.c[1] + R[2] * h.c[2] + p[0];
    ac_[i * 3 + 1] = R[3] * h.c[0] + R[4] * h.c[1] + R[5] * h.c[2] + p[1];
    ac_[i * 3 + 2] = R[6] * h.c[0] + R[7] * h.c[1] + R[8] * h.c[2] + p[2];
    ar_[i] = h.r;
  }

  // ---- stage 2 + 3: union hull to world, then one GJK per actor as a lower bound -----
  std::vector<std::pair<double, int>> order(n_actors);
  const bool have_union = (union_key >= 0) && union_.count(union_key);
  if (have_union) {
    const std::vector<double>& uv = union_.at(union_key);
    const int un = union_n_.at(union_key);
    uw_.resize(static_cast<size_t>(un) * 3);
    transform(uv.data(), un, union_xmat, union_xpos, uw_.data());
    for (int i = 0; i < n_actors; ++i) {
      const Hull& h = hulls_.at(actor_ids[i]);
      const double d = gjk_distance(aw_.data() + static_cast<size_t>(aoff_[i]) * 3, h.n,
                                    uw_.data(), un);
      ++out.union_hull_calls;
      order[i] = {std::max(0.0, d - h.pad), i};
    }
    std::stable_sort(order.begin(), order.end());
  } else {
    for (int i = 0; i < n_actors; ++i) order[i] = {0.0, i};
  }

  // ---- stage 4: piece hulls to world, with bounds ------------------------------------
  poff_.assign(n_pieces + 1, 0);
  for (int j = 0; j < n_pieces; ++j) poff_[j + 1] = poff_[j] + hulls_.at(piece_ids[j]).n;
  pw_.resize(static_cast<size_t>(poff_[n_pieces]) * 3);
  pc_.resize(static_cast<size_t>(n_pieces) * 3);
  pr_.resize(n_pieces);
  for (int j = 0; j < n_pieces; ++j) {
    const Hull& h = hulls_.at(piece_ids[j]);
    transform(h.v.data(), h.n, piece_xmat + j * 9, piece_xpos + j * 3,
              pw_.data() + static_cast<size_t>(poff_[j]) * 3);
    const double* R = piece_xmat + j * 9;
    const double* p = piece_xpos + j * 3;
    pc_[j * 3]     = R[0] * h.c[0] + R[1] * h.c[1] + R[2] * h.c[2] + p[0];
    pc_[j * 3 + 1] = R[3] * h.c[0] + R[4] * h.c[1] + R[5] * h.c[2] + p[1];
    pc_[j * 3 + 2] = R[6] * h.c[0] + R[7] * h.c[1] + R[8] * h.c[2] + p[2];
    pr_[j] = h.r;
  }

  // ---- stage 5: branch and bound ------------------------------------------------------
  double best = best_init;
  int b_a = -1, b_p = -1;
  std::vector<std::pair<double, int>> cand(n_pieces);
  for (const auto& oa : order) {
    if (oa.first >= best) break;        // this actor and every farther one cannot win
    const int i = oa.second;
    const Hull& ha = hulls_.at(actor_ids[i]);
    for (int j = 0; j < n_pieces; ++j) {
      const double dx = ac_[i * 3] - pc_[j * 3];
      const double dy = ac_[i * 3 + 1] - pc_[j * 3 + 1];
      const double dz = ac_[i * 3 + 2] - pc_[j * 3 + 2];
      cand[j] = {std::sqrt(dx * dx + dy * dy + dz * dz) - ar_[i] - pr_[j], j};
    }
    std::stable_sort(cand.begin(), cand.end());

    // How many pieces could possibly beat `best` on today's bound. Sequential scanning
    // shrinks this as it goes; a threaded scan cannot, so this is also the threading
    // decision: below the cutoff the parallel region costs more than the pairs save.
    int n_live = 0;
    while (n_live < n_pieces && cand[n_live].first < best) ++n_live;

#if defined(_OPENMP)
    if (n_threads > 1 && n_live >= kThreadCutoff) {
      // Threaded: `best` can only fall, and both prunes are exact, so reading a slightly
      // stale best is safe -- it evaluates pairs a sequential scan would have skipped but
      // never skips one it would have taken. The minimum is identical; only the pair
      // count changes. Ties are resolved toward the lower (actor, piece) index so the
      // reported nearest pair stays deterministic.
      double local_best = best;
      int local_a = b_a, local_p = b_p;
      int64_t pairs = 0;
      #pragma omp parallel num_threads(n_threads) reduction(+ : pairs)
      {
        double tb = local_best;
        int ta = -1, tp = -1;
        #pragma omp for schedule(dynamic, 4) nowait
        for (int q = 0; q < n_live; ++q) {
          double snap;
          #pragma omp atomic read
          snap = local_best;
          if (cand[q].first >= snap) continue;
          const int j = cand[q].second;
          const Hull& hp = hulls_.at(piece_ids[j]);
          const double d = distance_padded(aw_.data() + static_cast<size_t>(aoff_[i]) * 3,
                                           ha.n, ha.pad,
                                           pw_.data() + static_cast<size_t>(poff_[j]) * 3,
                                           hp.n, hp.pad);
          ++pairs;
          if (d < tb || (d == tb && tp >= 0 && j < tp)) {
            tb = d;
            ta = i;
            tp = j;
          }
          if (tb < local_best) {
            #pragma omp critical(fastgjk_best)
            if (tb < local_best) local_best = tb;
          }
        }
        #pragma omp critical(fastgjk_merge)
        {
          if (ta >= 0 && (tb < best || (tb == best && (b_p < 0 || tp < b_p)))) {
            best = tb;
            b_a = ta;
            b_p = tp;
          }
        }
      }
      out.exact_pairs += pairs;
      continue;
    }
#endif
    for (const auto& cj : cand) {
      if (cj.first >= best) break;      // remaining pieces are provably worse
      const int j = cj.second;
      const Hull& hp = hulls_.at(piece_ids[j]);
      const double d = distance_padded(aw_.data() + static_cast<size_t>(aoff_[i]) * 3,
                                       ha.n, ha.pad,
                                       pw_.data() + static_cast<size_t>(poff_[j]) * 3,
                                       hp.n, hp.pad);
      ++out.exact_pairs;
      if (d < best) {
        best = d;
        b_a = i;
        b_p = j;
      }
    }
  }
#if !defined(_OPENMP)
  (void)n_threads;
#endif
  out.best = best;
  out.actor_idx = b_a;
  out.piece_idx = b_p;
  return out;
}

}  // namespace fastgjk
