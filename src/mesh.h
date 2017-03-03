#ifndef __MESH_H__
#define __MESH_H__

#include <Eigen/Dense>
#include <vector>
#include <iostream>

using namespace Eigen;
using namespace std;

enum normal_mode {
  NO_NORMALS,
  AVERAGE,
  AREA_WEIGHTS,
  ANGLE_WEIGHTS
};

enum Feature {
	Position,
	Position_Color,
	Position_Normal,
	Position_Normal_Color
};

typedef pair<int, int> vertpair;
vertpair makeVertpair(int v1, int v2);

class vertex {
  public:
    vertex();
    vertex(const vertex &other);
    bool operator==(const vertex &other);

    int id;
    Vector3f loc;
	Vector3f color;
    Vector3f normal;

	VectorXf m_vp; // e.g. p = [px, py, pz, pr, pg, pb]T;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class face {
  public:
    face();
    face(const face &other);
    float area() const;

    int id;
    Vector3f normal;
    vector<vertex*> verts;
    vector<face*> neighbors;
    vector<vertpair> connectivity;
};

class mesh {
  public:
    mesh();
    void calculateNormals(normal_mode mode);

	void perservefeature(Feature mode);

    vector<face*> faces;
    bool manifold;
    int max_vertex_id;
};

#endif
