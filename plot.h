#ifndef VIS_H
#define VIS_H

#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/PI.h>
#include <igl/barycenter.h>
#include <set>

void add_route_edges(
  const std::vector<int>& route,
  std::set<std::pair<int,int>>& edges
);

void reset_scene(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F
);

void plot_extra_layer_mesh(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& I,
  bool invert_normals,
  const Eigen::RowVector3d& color
);

template <typename DerivedL>
void plot_label_on_face(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixBase<DerivedL>& label,
  const Eigen::VectorXi& mask
);

template <typename DerivedL>
void plot_label_on_halfedges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixBase<DerivedL>& label,
  const Eigen::MatrixXi& mask
);

void plot_tree(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  Eigen::SparseMatrix<int>& mat
);

void plot_cross_field(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& R,
  int N = 4,
  double arrow_len = 1.0
);

void plot_dir_along_path(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<int>& route,
  const Eigen::RowVector3d& c,
  bool is_closed
);

void plot_dir_of_halfedges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXi& D
);

void plot_singularity(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXd& S,
  double point_scale = 1.0
);
void cylinder(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0,
  double radius_top,
  double radius_bot,
  int index
);

void cylinder_filled(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0,
  double radius_top,
  double radius_bot,
  int index_v,
  int index_f
);

void plot_edges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& color_in,
  std::vector<std::pair<int,int>> & E
);

void plot_edges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::RowVector3d& c,
  std::set<std::pair<int,int>> & E,
  bool is_dual = false
);

void plot_bars(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& P1,
  const Eigen::MatrixXd& P2
);

void build_cylinder(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::RowVector3d& n1,
  double avg_len,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0
);

void show_boundary(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F);

void show_paths(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<std::vector<int>>& paths
);

void highlight_edges(
    const Eigen::SparseMatrix<int>& G,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool is_face = false
);

void plot_cut_graph_and_loop(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& T,
  const Eigen::VectorXd& S,
  const std::vector<std::vector<int>>& loops,
  const std::set<std::pair<int,int>>& edges,
  bool is_dual
);

void plot_cut_graph(
  igl::opengl::glfw::Viewer& vr,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& T,
  bool is_dual
);

void plot_tree(
    const std::vector<Eigen::VectorXi>& L, 
    const std::vector<bool>& I,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::VectorXd& sing,
    const Eigen::VectorXi& turning_n,
    std::set<std::pair<int,int>> & show_edge
);

void plot_highlight_faces(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<int>& L
);

void plot_TT(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXi& TT,
  const Eigen::VectorXi& I_vertex,
  const std::vector<std::vector<int>>& basis,
  const std::vector<std::vector<int>>& connectors,
  const std::vector<int>& dropped,
  int genus
);

void plot_pj(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& PJ
);

void plot_adj(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<bool>& I,
  const std::vector<std::vector<std::vector<int>>>& Ax,
  const std::set<std::pair<int,int>> edges,
  const Eigen::VectorXd& sing,
  const std::vector<int>& L,
  const std::vector<std::vector<int>>& loops,
  const std::vector<std::vector<int>>& paths,  
  const Eigen::MatrixXi& turning_n,
  const Eigen::VectorXd& WT
);

void plot_loops(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<std::vector<int>>& loops,
  const Eigen::VectorXi& He,
  const Eigen::VectorXi& Hf
);

void plot_paths(
    std::map<std::pair<int,int>,std::pair<int,int>>& branch,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
);

void plot_mesh_nrosy(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  int N,
  const Eigen::MatrixXd& PD1,
  const Eigen::VectorXd& S,
  const Eigen::VectorXi& b
);

#endif
