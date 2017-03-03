#include "simplify.h"
#include <Eigen/StdVector>
#include <vector>
#include <map>
#include <set>
#include <deque>


using namespace Eigen;
using namespace std;

enum collapse_method {
  COLLAPSE_TO_V1,
  COLLAPSE_TO_V2,
  COLLAPSE_TO_MEAN
};

inline VectorXf homogenous(const VectorXf &vec) {

	VectorXf ret(vec.size() + 1);
	for (int i = 0; i < vec.size(); i++)
		ret(i) = vec(i);
	ret(vec.size()) = 1.0;
	return ret;
}

float error(const VectorXf &homo, const MatrixXf &Q) {
  return abs((float) (homo.transpose() * Q * homo));
}

class contraction {
  public:
    vertpair vp;
    MatrixXf Q;
    VectorXf loc;
    collapse_method method;
    float resultError;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    contraction(
        const vertpair &vertp, const vector<vertex*> &verteces,
        const vector<MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &initialQ) {
      vp = vertp;
      findMinError(verteces, initialQ);
    }

    void findMinError(const vector<vertex*> &verteces,
        const vector<MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &initialQ) {
      VectorXf loc_v1 = homogenous(verteces[vp.first]->m_vp);
      VectorXf loc_v2 = homogenous(verteces[vp.second]->m_vp);
      VectorXf loc_vm = (loc_v1 + loc_v2) / 2;

      Q = initialQ[vp.first] + initialQ[vp.second];

      float err_v1 = error(loc_v1, Q),
            err_v2 = error(loc_v2, Q),
            err_vm = error(loc_vm, Q);

      /* unrolled 3-min */
      if (err_v1 < err_v2) {
        if (err_vm < err_v1) method = COLLAPSE_TO_MEAN;
        else method = COLLAPSE_TO_V1;
      } else {
        if (err_vm < err_v2) method = COLLAPSE_TO_MEAN;
        else method = COLLAPSE_TO_V2;
      }

      if (method == COLLAPSE_TO_V1) {
        loc = loc_v1;
        resultError = err_v1;
      } else if (method == COLLAPSE_TO_V2) {
        loc = loc_v2;
        resultError = err_v2;
      } else {
        loc = loc_vm;
        resultError = err_vm;
      }
    }

    bool operator<(const contraction &other) const {
      return resultError < other.resultError;
    }

    bool contains(int vid) {
      return vid == vp.first || vid == vp.second;
    }

    void perform(vector<vertex*> &verteces, vector<MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> &initialQ,
        vector<vector<face*> > &vertexToFaces,
        set<int> &facesToRemove, deque<contraction, Eigen::aligned_allocator<contraction>> &edges,
        set<vertpair> &existingedges) {
      /* Update Q for v1 */
      initialQ[vp.first] = Q;

      /* For code simplicity, we handle all three cases the same way. If we're
       * merging to the midpoint, we move v1 to the midpoint and remove v2. If
       * we're merging to v2, we move v1 to v2 and remove v2. */
      int keep = vp.first, remove = vp.second;
      if (method == COLLAPSE_TO_MEAN) {
        verteces[keep]->m_vp += verteces[remove]->m_vp;
        verteces[keep]->m_vp /= 2;
      } else if (method == COLLAPSE_TO_V2) {
        verteces[keep]->m_vp = verteces[remove]->m_vp;
      }

      /* Find all old faces on v2 */
      vector<face*> faces = vertexToFaces[remove];
      set<vertpair> potentiallyRemovableEdges;
      for (int i = 0; i < faces.size(); i++) {
        face* f = faces[i];

        /* If v1 and v2 are on this face, we mark it for removal. If
         * only v2 is in this face, we replace v2 with v1. */
        int v1idx = -1, v2idx = -1;
        for (int j = 0; j < f->verts.size(); j++) {
          if (f->verts[j]->id == keep) v1idx = j;
          else if (f->verts[j]->id == remove) v2idx = j;
        }

        if (v1idx == -1) {
          f->verts[v2idx] = verteces[keep];
          for (int j = 0; j < f->connectivity.size(); j++) {
            if (f->connectivity[j].first == remove)
              f->connectivity[j] = makeVertpair(keep, f->connectivity[j].second);
            else if (f->connectivity[j].second == remove)
              f->connectivity[j] = makeVertpair(keep, f->connectivity[j].first);
          }
          vertexToFaces[keep].push_back(f);
        } else {
          for (int j = 0; j < f->connectivity.size(); j++) {
            vertpair p = f->connectivity[j];
            if (p.first == remove || p.second == remove)
              potentiallyRemovableEdges.insert(f->connectivity[j]);
          }
          facesToRemove.insert(f->id);
        }
      }

	  faces.clear();

      /* Update edge heap */
      deque<int> edgesToRemove;
      for (int i = 0; i < edges.size(); i++) {
        if (vp == edges[i].vp) continue;
        if (edges[i].contains(keep)) {
          /* This edge contains the vertex we kept. We just have to update its
           * error value. */
          edges[i].findMinError(verteces, initialQ);
        } else if (edges[i].contains(remove)) {
          /* This edge contains the vertex we removed. This gets a bit hairy.
           * If this edge is on a triangle marked for removal, we remove it
           * entirely. If it's not, then we just replace v2 with v1. */

          vertpair possible;
          if (edges[i].vp.first == remove) {
            possible = makeVertpair(edges[i].vp.second, keep);
          } else {
            possible = makeVertpair(edges[i].vp.first, keep);
          }

          if (existingedges.find(possible) != existingedges.end()) {
            edgesToRemove.push_back(i);
          } else {
            edges[i].vp = possible;
            edges[i].findMinError(verteces, initialQ);
          }
        }
      }

      int offset = 0;
      for (int i = 0; i < edgesToRemove.size(); i++) {
        existingedges.erase(edges[edgesToRemove[i]].vp);
        edges.erase(edges.begin() + edgesToRemove[i] + offset);
        offset--;
      }

      vertexToFaces[remove].clear();
    }
};

contraction popmin(deque<contraction, Eigen::aligned_allocator<contraction>> &edges) {
  contraction best = edges.front();
  int bestidx = 0;
  for (int i = 1; i < edges.size(); i++) {
    if (edges[i] < best) {
      bestidx = i;
      best = edges[i];
    }
  }
  edges[bestidx] = edges.back();
  edges.pop_back();
  return best;
}

void simplifyMesh(mesh &mesh, float factor) {
  if (!mesh.manifold) {
    cout << "Simplification will not work on non-manifold meshes." << endl;
    return;
  }

  mesh.perservefeature(Position);
  const int N_Vec = 3 * 1;
  /*
    1. Compute the Q matrices for all the initial vertices.
    2. Select all valid pairs
  */
  vector<vertex*> verteces(mesh.max_vertex_id + 1, NULL);
  vector<MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> initialQ(mesh.max_vertex_id + 1, MatrixXf::Zero(N_Vec + 1,N_Vec + 1));


  vector<vector<face*> > vertexToFaces(mesh.max_vertex_id + 1, vector<face*>());

  set<vertpair> edgeset;

  for (int i = 0; i < mesh.faces.size(); i++) {
    face *f = mesh.faces[i];

	VectorXf v1 = f->verts[0]->m_vp,
		v2 = f->verts[1]->m_vp,
		v3 = f->verts[2]->m_vp;

	VectorXf e1 = (v2 - v1);
	e1 = e1 / e1.norm();
	VectorXf e2 = (v3 - v1 - (e1.dot(v3- v1))*(e1));
	e2 = e2 / e2.norm();
	MatrixXf A = MatrixXf::Identity(N_Vec, N_Vec) - e1 * e1.transpose() - e2 * e2.transpose();
	VectorXf b = (v1.dot(e1))* e1 + (v1.dot(e2))*e2 - v1;
	float c = v1.dot(v1) - (v1.dot(e1)) * (v1.dot(e1)) - (v1.dot(e2)) * (v1.dot(e2));
	MatrixXf pp = MatrixXf::Zero(N_Vec + 1, N_Vec + 1);
	for (int j = 0; j < N_Vec; j++)
	{
		for (int k = 0; k < N_Vec; k++)
			pp(j, k) = A(j, k);

		pp(j, N_Vec) = b(j);
		pp(N_Vec, j) = b(j);
	}
	pp(N_Vec, N_Vec) = c;

    for (auto v_iter = f->verts.begin();
        v_iter != f->verts.end(); v_iter++) {
      vertex* vert = *v_iter;

      verteces[vert->id] = vert;
      vertexToFaces[vert->id].push_back(f);

      /* Add the plane represented by this face on this vertex. */
      initialQ[vert->id] += pp;
    }

    /* Add edges to set */
    for (int j = 0; j < f->connectivity.size(); j++) {
      vertpair edge = f->connectivity[j];
      edgeset.insert(edge);
    }
  }

  /*
    3. Compute the optimal contraction target v' for each valid pair (v1, v2).
       The error v'^T (Q1+Q2) v' of this target vertex becomes the cost of 
       contracting that pair.
    4. Place all the pairs in a heap keyed on cost with the minimum cost
       pair at the top.
  */
  deque<contraction,Eigen::aligned_allocator<contraction>> edges;
  for (auto edge = edgeset.begin(); edge != edgeset.end(); edge++) {
    edges.push_back(contraction(*edge, verteces, initialQ));
  }
  //make_heap(edges.begin(), edges.end());

  /*
  5. Iteratively remove the pair (v1,v2) of least cost from the heap, contract
     this pair, and update the costs of all valid pairs involving v1.
  */
  set<int> facesToRemove;

  int target_edges = (int) (factor * (float) edges.size());
  while (edges.size() > target_edges) {

    contraction best = popmin(edges);
    best.perform(verteces, initialQ, vertexToFaces, facesToRemove, edges, edgeset);

  }

  vector<face*> newfaces;
  for (int i = 0; i < mesh.faces.size(); i++) {
    if (facesToRemove.find(mesh.faces[i]->id) == facesToRemove.end()) {
      newfaces.push_back(mesh.faces[i]);
    }
  }
  mesh.faces.assign(newfaces.begin(), newfaces.end());
  newfaces.clear();
  cout << "Simplification End." << endl;
}
