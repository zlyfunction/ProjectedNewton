#include "plot.h"
#include <igl/avg_edge_length.h>
#include <igl/local_basis.h>
#include <unordered_set>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/boundary_loop.h>
#include <igl/rotate_vectors.h>
#include <igl/comb_cross_field.h>
#include <igl/avg_edge_length.h>

void add_route_edges(
  const std::vector<int>& route,
  std::set<std::pair<int,int>>& edges
){
  if(route.size() <= 1) return;
  for(int i=0;i<route.size()-1;i++){
    int v0 = route[i];
    int v1 = route[i+1];
    edges.insert(std::make_pair(v0,v1));
  }
}

void reset_scene(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F
){
  viewer.data_list.resize(1);
  viewer.selected_data_index = 0;
  viewer.data().clear();
  viewer.data().set_mesh(V,F);
  Eigen::MatrixXd color(F.rows(),3);
  color.setConstant(1);
  viewer.data().set_colors(color);
}

void plot_extra_layer_mesh(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& I,
  bool invert_normals,
  const Eigen::RowVector3d& color
){
  if(I.rows() != F.rows()) return;
  Eigen::MatrixXd V_part = V;
  if(V_part.cols() == 2){
    V_part.conservativeResize(V_part.rows(),3);
    V_part.col(2).setConstant(1e-3);
  }
  Eigen::MatrixXi F_part(I.sum(), 3);
  Eigen::MatrixXd colors(I.sum(), 3);
  int index = 0;
  for(int i=0;i<F.rows();i++){
    if(I(i)){
      F_part.row(index) << F.row(i);
      colors.row(index) << color;
      index++;
    }
  }
  
  viewer.append_mesh();
  viewer.data().set_mesh(V_part,F_part);
  viewer.data().invert_normals = invert_normals;
  viewer.data().set_colors(colors);
  
}

void plot_tree(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  Eigen::SparseMatrix<int>& mat
){
  std::set<std::pair<int,int>> left_tree;
  for (int k=0; k<mat.outerSize(); ++k){
    for (Eigen::SparseMatrix<int>::InnerIterator it(mat,k); it; ++it){
      left_tree.insert(std::make_pair(it.row(), it.col()));
    }
  }
  plot_edges(viewer, V, F, Eigen::RowVector3d(0,0,0), left_tree);
}

void plot_singularity(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXd& S,
  double point_scale
){
  
  auto V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  int n_pos = 0, n_neg = 0;
  for(int i=0;i<S.rows();i++){
    if(S(i) > 0)
      n_pos++;
    else if(S(i) < 0)
      n_neg++;
  }
  
  double avg = igl::avg_edge_length(V,F);
  
  Eigen::MatrixXd point_mesh_V_red(n_pos*8, 3);
  Eigen::MatrixXi point_mesh_F_red(n_pos*12, 3);
  Eigen::MatrixXd point_mesh_V_blue(n_neg*8, 3);
  Eigen::MatrixXi point_mesh_F_blue(n_neg*12, 3);
  
  int index_red_v = 0, index_red_f = 0;
  int index_blue_v = 0, index_blue_f = 0;
  for(int i=0;i<S.rows();i++){
    if(S(i) > 0){
      Eigen::RowVector3d pos = V.row(i);
      Eigen::RowVector3d v0 = pos.array() + avg*0.1*point_scale;
      Eigen::RowVector3d v1 = pos.array() - avg*0.1*point_scale;
      cylinder_filled(v0, v1, point_mesh_V_red, point_mesh_F_red, avg*0.2*point_scale, avg*0.2*point_scale, index_red_v, index_red_f);
      index_red_v += 8;
      index_red_f += 12;
    }else if(S(i) < 0){
      Eigen::RowVector3d pos = V.row(i);
      Eigen::RowVector3d v0 = pos.array() + avg*0.1*point_scale;
      Eigen::RowVector3d v1 = pos.array() - avg*0.1*point_scale;
      cylinder_filled(v0, v1, point_mesh_V_blue, point_mesh_F_blue, avg*0.2*point_scale, avg*0.2*point_scale, index_blue_v, index_blue_f);
      index_blue_v += 8;
      index_blue_f += 12;
    }
  }
  viewer.append_mesh();
  viewer.data().set_mesh(point_mesh_V_red,point_mesh_F_red);
  viewer.data().show_lines = false;
  Eigen::MatrixXd color_red(point_mesh_F_red.rows(),3);
  for(int i=0;i<color_red.rows();i++)
    color_red.row(i) << 1,0,0;
  viewer.data().set_colors(color_red);
  
  viewer.append_mesh();
  viewer.data().set_mesh(point_mesh_V_blue,point_mesh_F_blue);
  viewer.data().show_lines = false;
  Eigen::MatrixXd color_blue(point_mesh_F_blue.rows(),3);
  for(int i=0;i<color_blue.rows();i++)
    color_blue.row(i) << 0,0,1;
  viewer.data().set_colors(color_blue);
}

void cylinder_filled(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0,
  double radius_top,
  double radius_bot,
  int index_v,
  int index_f
){
  Eigen::RowVector3d n1;
  Eigen::RowVector3d v1v0 = v0-v1;
  if(v1v0(0) != 0)
    n1 << (-v1v0(1)-v1v0(2))/v1v0(0), 1, 1;
  else if(v1v0(1) != 0)
    n1 << 1, (-v1v0(0)-v1v0(2))/v1v0(1), 1;
  else
    n1 << 1, 1, (-v1v0(0)-v1v0(1))/v1v0(2);
  n1.normalize();
  
  Eigen::RowVector3d n2 = ((v0-v1).cross(n1)).normalized();
  Eigen::RowVector3d p1 = v0 + n1 * radius_top;
  Eigen::RowVector3d q1 = v1 + n1 * radius_bot;
  Eigen::RowVector3d p2 = v0 + n2 * radius_top;
  Eigen::RowVector3d q2 = v1 + n2 * radius_bot;
  Eigen::RowVector3d p3 = v0 - n1 * radius_top;
  Eigen::RowVector3d q3 = v1 - n1 * radius_bot;
  Eigen::RowVector3d p4 = v0 - n2 * radius_top;
  Eigen::RowVector3d q4 = v1 - n2 * radius_bot;
  V0.block(index_v,0,8,3)<<p1,p2,p3,p4,q1,q2,q3,q4;
  F0.block(index_f,0,12,3)<<1,0,5,0,4,5,0,3,7,0,7,4,3,2,7,7,2,6,2,1,5,2,5,6,1,2,0,0,2,3,4,6,5,7,6,4;
  F0.block(index_f,0,12,3)<<F0.block(index_f,0,12,3).array()+index_v;
  
}

void cylinder(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0,
  double radius_top,
  double radius_bot,
  int index
){
  Eigen::RowVector3d n1;
  Eigen::RowVector3d v1v0 = v0-v1;
  if(v1v0(0) != 0)
    n1 << (-v1v0(1)-v1v0(2))/v1v0(0), 1, 1;
  else if(v1v0(1) != 0)
    n1 << 1, (-v1v0(0)-v1v0(2))/v1v0(1), 1;
  else
    n1 << 1, 1, (-v1v0(0)-v1v0(1))/v1v0(2);
  n1.normalize();
  
  Eigen::RowVector3d n2 = ((v0-v1).cross(n1)).normalized();
  Eigen::RowVector3d p1 = v0 + n1 * radius_top;
  Eigen::RowVector3d q1 = v1 + n1 * radius_bot;
  Eigen::RowVector3d p2 = v0 + n2 * radius_top;
  Eigen::RowVector3d q2 = v1 + n2 * radius_bot;
  Eigen::RowVector3d p3 = v0 - n1 * radius_top;
  Eigen::RowVector3d q3 = v1 - n1 * radius_bot;
  Eigen::RowVector3d p4 = v0 - n2 * radius_top;
  Eigen::RowVector3d q4 = v1 - n2 * radius_bot;
  V0.block(index,0,8,3)<<p1,p2,p3,p4,q1,q2,q3,q4;
  F0.block(index,0,8,3)<<1,0,5,0,4,5,0,3,7,0,7,4,3,2,7,7,2,6,2,1,5,2,5,6;
  F0.block(index,0,8,3)<<F0.block(index,0,8,3).array()+index;
}

template <typename DerivedL>
void plot_label_on_face(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixBase<DerivedL>& label,
  const Eigen::VectorXi& mask
){
  Eigen::MatrixXd bc;
  igl::barycenter(V,F,bc);
  for(int i=0;i<F.rows();i++){
    if(mask(i)){
      auto pos = bc.row(i);
      viewer.data().add_label(pos, std::to_string(int(label(i))));
    }
  }
}

template <typename DerivedL>
void plot_label_on_halfedges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixBase<DerivedL>& label,
  const Eigen::MatrixXi& mask
){
  auto V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  Eigen::MatrixXd bc;
  igl::barycenter(V,F,bc);
  for(int i=0;i<F.rows();i++){
    for(int k=0;k<3;k++){
      if(mask(i,k)){
        int u = F(i,k);
        int v = F(i,(k+1)%3);
        Eigen::RowVector3d v0 = V.row(u)*0.8+bc.row(i)*0.2;
        Eigen::RowVector3d v1 = V.row(v)*0.8+bc.row(i)*0.2;
        Eigen::RowVector3d pos = 0.5*v0 + 0.5*v1;
        if(label(i,k) > 0)
          viewer.data().add_label(pos, std::to_string((label(i,k))));
      }
    }
  }
  
}

void plot_dir_of_halfedges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXi& D
){
  
  auto V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  
  // +U dark red
  // -U light red
  // +V dark blue
  // -V light blue
  int he_num = 0;
  for(int i=0;i<D.rows();i++){
    for(int k=0;k<3;k++){
      if(D(i,k) != -1)
        he_num++;
    }
  }
  
  Eigen::MatrixXd edge_mesh_V(he_num*16,3);
  Eigen::MatrixXi edge_mesh_F(he_num*16,3);
  
  double avg = igl::avg_edge_length(V,F);
  Eigen::MatrixXd color(edge_mesh_F.rows(),3);
  Eigen::MatrixXd bc;
  igl::barycenter(V,F,bc);
  int index = 0;
  for(int i=0;i<D.rows();i++){
    for(int k=0;k<3;k++){
      int u = F(i,k);
      int v = F(i,(k+1)%3);
      int dir = D(i,k);
      if(dir != -1){
        Eigen::RowVector3d v0 = (V.row(u)+bc.row(i))/2;
        Eigen::RowVector3d v1 = (V.row(v)+bc.row(i))/2;
        Eigen::RowVector3d v2 = 0.2*v0 + 0.8*v1;
        cylinder(v0, v2, edge_mesh_V, edge_mesh_F, avg*0.03, avg*0.03, index);
        cylinder(v2, v1, edge_mesh_V, edge_mesh_F, avg*0.08, 1e-10, index+8);
        for(int f=index;f<index+16;f++){
          if(dir == 0)
            color.row(f) << 1,0,0;
          else if(dir == 1)
            color.row(f) << 0.25,0.25,1;
          else if(dir == 2)
            color.row(f) <<1, 0.5, 0;
          else if(dir == 3)
            color.row(f) << 0.25,1,0.5;
          else if(dir == 4)
            color.row(f) << 0.05,1,0.75;
          else if(dir == 5)
            color.row(f) << 0.1,0.1,0.1;
          else if(dir == 6)
            color.row(f) << 0.8,0.3,0.7;
          else if(dir == 7)
            color.row(f) << 1,1,1;
          else if(dir % 2 == 0)
            color.row(f) << 0.9,0.9,0.2;
          else
            color.row(f) << 0.9,0.2,0.9;
            
        }
        index+=16;
      }
    }
  }
  viewer.append_mesh();
  viewer.data().set_mesh(edge_mesh_V,edge_mesh_F);
  viewer.data().show_lines = false;
  viewer.data().set_colors(color);
}

void plot_dir_along_path(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<int>& route,
  const Eigen::RowVector3d& c,
  bool is_closed
){
  if(route.size() <= 1) return;
  int size = is_closed ? route.size() : route.size()-1;
  Eigen::MatrixXd edge_mesh_V(size*16,3);
  Eigen::MatrixXi edge_mesh_F(size*16,3);
  
  double avg = igl::avg_edge_length(V,F);
  
  int index = 0; // ptr to bottom of matrix
  int end = is_closed ? route.size() : route.size()-1;
  for(int i=0;i<end;i++){
    int u = route[i];
    int v = route[(i+1)%route.size()];
    Eigen::RowVector3d v0 = V.row(u);
    Eigen::RowVector3d v1 = V.row(v);
    Eigen::RowVector3d v2 = 0.1*V.row(u) + 0.9*V.row(v);
    cylinder(v0, v2, edge_mesh_V, edge_mesh_F, avg*0.005, avg*0.005, index);
    cylinder(v2, v1, edge_mesh_V, edge_mesh_F, avg*0.008, 1e-10, index+8);
    index += 16;
  }
  viewer.append_mesh();
  Eigen::MatrixXd color(edge_mesh_F.rows(),3);
  for(int i=0;i<color.rows();i++)
    color.row(i) << c;
  viewer.data().set_mesh(edge_mesh_V,edge_mesh_F);
  viewer.data().show_lines = false;
  viewer.data().set_colors(color);
}

void plot_edges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& color_in,
  std::vector<std::pair<int,int>> & E
){
 
  Eigen::MatrixXd V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  Eigen::MatrixXd edge_mesh_V(E.size()*8,3);
  Eigen::MatrixXi edge_mesh_F(E.size()*8,3);
  
  double avg = igl::avg_edge_length(V,F);
  int index = 0; // ptr to bottom of matrix
  for(auto e: E){
    int u = e.first;
    int v = e.second;
    Eigen::RowVector3d v0 = V.row(u);
    Eigen::RowVector3d v1 = V.row(v);
    cylinder(v0, v1, edge_mesh_V, edge_mesh_F, avg*0.02, avg*0.02, index);
    index += 8;
  }
 
  viewer.append_mesh();
  Eigen::MatrixXd color(edge_mesh_F.rows(),3);
  
  for(int i=0;i<edge_mesh_F.rows();i+=8){
    color.block(i,0,8,3).col(0).setConstant(color_in(i/8,0));
    color.block(i,0,8,3).col(1).setConstant(color_in(i/8,1));
    color.block(i,0,8,3).col(2).setConstant(color_in(i/8,2));
  }

  viewer.data().set_mesh(edge_mesh_V,edge_mesh_F);
  viewer.data().show_lines = false;
  viewer.data().set_colors(color);
}

void plot_edges(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V_s,
  const Eigen::MatrixXi& F,
  const Eigen::RowVector3d& c,
  std::set<std::pair<int,int>> & E,
  bool is_dual
){
  Eigen::MatrixXd V = V_s;
  if(V.cols() == 2){
    V.conservativeResize(V.rows(),3);
    V.col(2).setZero();
  }
  Eigen::MatrixXd edge_mesh_V(E.size()*8,3);
  Eigen::MatrixXi edge_mesh_F(E.size()*8,3);
  
  double avg = igl::avg_edge_length(V,F);
  
  int index = 0; // ptr to bottom of matrix
  if(!is_dual){
    for(auto e: E){
      int u = e.first;
      int v = e.second;
      Eigen::RowVector3d v0 = V.row(u);
      Eigen::RowVector3d v1 = V.row(v);
      cylinder(v0, v1, edge_mesh_V, edge_mesh_F, avg*0.01, avg*0.01, index);
      index += 8;
    }
  }else{
    for(auto e: E){
      int f0 = e.first;
      int f1 = e.second;
      Eigen::RowVector3d v0 = (V.row(F(f0,0)) + V.row(F(f0,1)) + V.row(F(f0,2)))/3;
      Eigen::RowVector3d v1 = (V.row(F(f1,0)) + V.row(F(f1,1)) + V.row(F(f1,2)))/3;
      cylinder(v0, v1, edge_mesh_V, edge_mesh_F, avg*0.01, avg*0.01,index);
      index += 8;
    }
  }
  viewer.append_mesh();
  Eigen::MatrixXd color(edge_mesh_F.rows(),3);
  for(int i=0;i<color.rows();i++)
    color.row(i) << c;
  viewer.data().set_mesh(edge_mesh_V,edge_mesh_F);
  viewer.data().show_lines = false;
  viewer.data().set_colors(color);
}

// for every face plot a cross
void plot_cross_field(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& R,
  int N,
  double arrow_len
){
  std::vector<Eigen::RowVector3d> C;
  C.push_back(Eigen::RowVector3d(1,0,0));
  C.push_back(Eigen::RowVector3d(0.25,0.25,1));
  C.push_back(Eigen::RowVector3d(1,0.5,0));
  C.push_back(Eigen::RowVector3d(0.25,1,0.5));
  
  Eigen::MatrixXd B1,B2,B3;
  igl::local_basis(V, F, B1, B2, B3);
  std::vector<Eigen::MatrixXd> X;
  X.push_back(R);
  X.push_back(igl::rotate_vectors(R, Eigen::VectorXd::Constant(1, igl::PI/2), B1, B2));
  X.push_back(igl::rotate_vectors(R, Eigen::VectorXd::Constant(1, igl::PI), B1, B2));
  X.push_back(igl::rotate_vectors(R, Eigen::VectorXd::Constant(1, 3*igl::PI/2), B1, B2));
  
  Eigen::MatrixXd center;
  igl::barycenter(V,F,center);
  
  double avg = igl::avg_edge_length(V,F);
  
  for(int k=0;k<N;k++){
    Eigen::MatrixXd edge_mesh_V(F.rows()*16,3);
    Eigen::MatrixXi edge_mesh_F(F.rows()*16,3);
    int index = 0; // ptr to bottom of matrix
    for(int i=0;i<F.rows();i++){
      Eigen::RowVector3d v0 = center.row(i);
      Eigen::RowVector3d v1 = v0+X[k].row(i)*arrow_len*avg/2;
      Eigen::RowVector3d v2 = 0.2*v0 + 0.8*v1;
      cylinder(v0, v2, edge_mesh_V, edge_mesh_F, avg*0.01, avg*0.01, index);
      cylinder(v2, v1, edge_mesh_V, edge_mesh_F, avg*0.02, 1e-10, index+8);
      index+=16;
    }
    viewer.append_mesh();
    Eigen::MatrixXd color(edge_mesh_F.rows(),3);
    for(int i=0;i<color.rows();i++)
      color.row(i) << C[k];
    viewer.data().set_mesh(edge_mesh_V,edge_mesh_F);
    viewer.data().show_lines = false;
    viewer.data().set_colors(color);
  }
  
}
void build_cylinder(
  Eigen::RowVector3d& v0,
  Eigen::RowVector3d& v1,
  Eigen::RowVector3d& n1,
  double avg_len,
  Eigen::MatrixXd& V0,
  Eigen::MatrixXi& F0
){
  double radius = avg_len * 0.1;
  Eigen::RowVector3d n2 = ((v0-v1).cross(n1)).normalized();
  Eigen::RowVector3d p1 = v0 + n1 * radius;
  Eigen::RowVector3d q1 = v1 + n1 * radius;
  Eigen::RowVector3d p2 = v0 + n2 * radius;
  Eigen::RowVector3d q2 = v1 + n2 * radius;
  Eigen::RowVector3d p3 = v0 - n1 * radius;
  Eigen::RowVector3d q3 = v1 - n1 * radius;
  Eigen::RowVector3d p4 = v0 - n2 * radius;
  Eigen::RowVector3d q4 = v1 - n2 * radius;
  int n = V0.rows();
  V0.conservativeResize(V0.rows()+8,3);
  F0.conservativeResize(F0.rows()+8,3);
  V0.bottomRows(8)<<p1,p2,p3,p4,q1,q2,q3,q4;
  F0.bottomRows(8)<<1,0,5,0,4,5,0,3,7,0,7,4,3,2,7,7,2,6,2,1,5,2,5,6;
  F0.bottomRows(8)<<F0.bottomRows(8).array()+n;
}

void show_boundary(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F
){
  
  std::vector<std::vector<int>> bds;
  igl::boundary_loop(F, bds);
  
  for(auto bd : bds){
    if(bd.size() <= 1){ continue; }
    std::set<std::pair<int,int>> bd_e;
    for(int i=0;i<bd.size();i++){
      int v0 = bd[i];
      int v1 = bd[(i+1)%bd.size()];
      bd_e.insert(std::make_pair(v0, v1));
    }
    plot_edges(viewer, V, F, Eigen::RowVector3d(0,0,0), bd_e);
    viewer.data().show_lines = true;
  }
}


// Converts a representative vector per face in the full set of vectors that describe
// an N-RoSy field
void representative_to_nrosy(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& R,
  const int N,
  Eigen::MatrixXd& Y)
{
  using namespace Eigen;
  using namespace std;
  MatrixXd B1, B2, B3;

  igl::local_basis(V,F,B1,B2,B3);

  Y.resize(F.rows()*N,3);
  for (unsigned i=0;i<F.rows();++i)
  {
    double x = R.row(i) * B1.row(i).transpose();
    double y = R.row(i) * B2.row(i).transpose();
    double angle = atan2(y,x);

    for (unsigned j=0; j<N;++j)
    {
      double anglej = angle + 2*igl::PI*double(j)/double(N);
      double xj = cos(anglej);
      double yj = sin(anglej);
      Y.row(i*N+j) = xj * B1.row(i) + yj * B2.row(i);
    }
  }
}

// void plot_vectors(
//   igl::opengl::glfw::Viewer& viewer,
//   Eigen::MatrixXd& vec,
//   int color
// ){
  
//   // build an overlay mesh composed of 
//   // cylinders for every unit length vector in vec
  
  
  
// }

// Plots the mesh with an N-RoSy field and its singularities on top
// The constrained faces (b) are colored in red.
void plot_mesh_nrosy(
  igl::opengl::glfw::Viewer& viewer,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  int N,
  const Eigen::MatrixXd& PD1,
  const Eigen::VectorXd& S,
  const Eigen::VectorXi& b)
{
  using namespace Eigen;
  using namespace std;
  // Clear the mesh

  // Expand the representative vectors in the full vector set and plot them as lines
  double avg = igl::avg_edge_length(V, F);
  // MatrixXd Y;
  // representative_to_nrosy(V, F, PD1, N, Y);

  MatrixXd B;
  igl::barycenter(V,F,B);
  std::vector<Eigen::RowVector3d> cc;
  cc = {Eigen::RowVector3d(1,0,0),
        Eigen::RowVector3d(0,1,0),
        Eigen::RowVector3d(0,0,1),
        Eigen::RowVector3d(0,0,0)};
  viewer.data().add_edges(B,B+PD1*(avg/5),cc[N]);
  // } 

  // Plot the singularities as colored dots (red for negative, blue for positive)
  for (unsigned i=0; i<S.size();++i)
  {
    if (S(i) < -0.001)
      viewer.data().add_points(V.row(i),RowVector3d(0,0,1));
    else if (S(i) > 0.001)
      viewer.data().add_points(V.row(i),RowVector3d(1,0,0));
  }

  // Highlight in red the constrained faces
  MatrixXd C = MatrixXd::Constant(F.rows(),3,1);
  for (unsigned i=0; i<b.size();++i)
    C.row(b(i)) << 1, 1, 0;
  viewer.data().set_colors(C);
}

void highlight_edges(
    const Eigen::SparseMatrix<int>& G,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool is_face
){
    igl::opengl::glfw::Viewer v;
    v.data().set_mesh(V,F);
    for (int k=0; k<G.outerSize(); ++k)
        for (Eigen::SparseMatrix<int>::InnerIterator it(G,k); it; ++it){
            it.value();
            it.row();   // row index
            it.col();   // col index (here it is equal to k)
            it.index(); // inner index, here it is equal to it.row()
            if(!is_face)
                v.data().add_edges(V.row(it.row()),V.row(it.col()),Eigen::RowVector3d(1,0,0));
            else{
                Eigen::RowVector3d a,b;
                a = (V.row(F(it.row(),0))+V.row(F(it.row(),1))+V.row(F(it.row(),2)))/3;
                b = (V.row(F(it.col(),0))+V.row(F(it.col(),1))+V.row(F(it.col(),2)))/3;
                v.data().add_edges(a,b,Eigen::RowVector3d(1,0,0));
            }
        }
    v.launch();
}
void collect_edges(
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& T,
  std::set<std::pair<int,int>>& edges,
  bool is_face
){
  if(!is_face){
    for(int i=0;i<F.rows();i++){
      for(int k=0;k<3;k++){
        int fk = F(i,k);
        int fk_1 = F(i,(k+1)%3);
        if(T(fk_1) == fk || T(fk) == fk_1){
          if(fk > fk_1) std::swap(fk,fk_1);
          edges.insert(std::make_pair(fk,fk_1));
        }
      }
    }
  }else{
    for(int i=0;i<T.rows();i++)
      if(T(i)!=-1)
        edges.insert(std::make_pair(T(i),i));
  }
}

void plot_cut_graph_and_loop(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& T,
  const Eigen::VectorXd& sing,
  const std::vector<std::vector<int>>& loops,
  const std::set<std::pair<int,int>>& edges,
  bool is_dual
){
  igl::opengl::glfw::Viewer viewer;
  plot_cut_graph(viewer,V,F,T,is_dual);
  for(auto path: loops){
    for(int i=0;i<path.size();i++){
      int i_1 = (i+1) % path.size();
      viewer.data().add_edges(V.row(path[i]),V.row(path[i_1]),Eigen::RowVector3d(1,0,0));
    }
  }
  for(int i=0;i<sing.rows();i++){
    if(sing(i)!=0)
      viewer.data().add_points(V.row(i),Eigen::RowVector3d(1,0,0));
  }
  for(auto p: edges){
    viewer.data().add_points(V.row(p.first),Eigen::RowVector3d(0,0,0));
    viewer.data().add_points(V.row(p.second),Eigen::RowVector3d(0,0,0));
    viewer.data().add_edges(V.row(p.first),V.row(p.second),Eigen::RowVector3d(0,0,0));
  }
  viewer.launch();
}
void plot_tree(
   const std::vector<Eigen::VectorXi>& L, 
   const std::vector<bool>& I,
   const Eigen::MatrixXd& V,
   const Eigen::MatrixXi& F,
   const Eigen::VectorXd& sing,
   const Eigen::VectorXi& turning_n,
   std::set<std::pair<int,int>> & show_edge
){
    std::cout<<"plot spanning tree\n";
    std::vector<std::set<std::pair<int,int>>> edges;
    int id = 0;
    for(bool is_face: I){
        std::set<std::pair<int,int>> E;
        int num_ele = is_face ? F.rows() : V.rows();
        collect_edges(F,L[id++],E,is_face);
        edges.push_back(E);
    }
    igl::opengl::glfw::Viewer v;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    v.plugins.push_back(&menu);
    v.data().set_mesh(V,F);
    
    // std::cout<<"#T:  "<<edges[0].size()<<std::endl;
    // std::cout<<"#T0: "<<edges[1].size()<<std::endl;
    id = 0;
    Eigen::MatrixXd C(3,3);
    C<<1,0,0,0,0,0,0,0,1;
    Eigen::MatrixXd C0 = Eigen::MatrixXd::Constant(F.rows(),3,1);
    v.data().set_colors(C0);
    
    Eigen::MatrixXd N;
    igl::per_face_normals(V,F,N);
    auto key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
      int id = key-'0';
      if(id >= edges.size()) return false;
      auto E = edges[id];
      v.data().clear();
      v.data().set_mesh(V,F);
      v.data().set_colors(C0);
      for(auto e: E){
          if(!I[id])
              v.data().add_edges(V.row(e.first),V.row(e.second),C.row(id));
          else{
              Eigen::RowVector3d a,b;
              a = (V.row(F(e.first,0))+V.row(F(e.first,1))+V.row(F(e.first,2)))/3+N.row(e.first)*0.02;
              b = (V.row(F(e.second,0))+V.row(F(e.second,1))+V.row(F(e.second,2)))/3+N.row(e.second)*0.02;
              v.data().add_edges(a,b,C.row(id));
          }
      }
      for(auto p: show_edge){
        v.data().add_points(V.row(p.first),Eigen::RowVector3d(0,0,0));
        v.data().add_points(V.row(p.second),Eigen::RowVector3d(0,0,0));
        v.data().add_edges(V.row(p.first),V.row(p.second),Eigen::RowVector3d(0,0,0));
      }
      return false;
    };
    v.callback_key_down=key_down;
    // for(int i=0;i<F.rows();i++){
    //     for(int j=0;j<3;j++){
    //         if(F(i,j)>F(i,(j+1)%3)) continue;
    //         int e = i*3+j;
    //         Eigen::RowVector3d p = (V.row(F(i,j))+V.row(F(i,(j+1)%3)))/2;
    //         v.data().add_label(p,std::to_string(turning_n(e)));
    //     }
    // }
    for(int i=0;i<sing.rows();i++){
        if(sing(i)>0)
            v.data().add_points(V.row(i),Eigen::RowVector3d(1,0,0));
    }
    v.launch();
}

void plot_cut_graph(
  igl::opengl::glfw::Viewer& vr,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& T,
  bool is_dual
){
  
  Eigen::MatrixXi TT0,TT0i;
  vr.data().set_mesh(V,F);
  igl::triangle_triangle_adjacency(F,TT0,TT0i);
  Eigen::MatrixXd N;
  igl::per_face_normals(V,F,N);
  if(is_dual){
    for(int i=0;i<F.rows();i++){
      for(int k=0;k<3;k++){
        int a = F(i,k);
        int b = F(i,(k+1)%3);
        int f0 = i;
        int f1 = TT0(i,k);
        if(T(f0)==f1 || T(f1)==f0){
          auto pt0 = (V.row(F(f0,0))+V.row(F(f0,1))+V.row(F(f0,2)))/3+N.row(f0)*0.001;
          auto pt1 = (V.row(F(f1,0))+V.row(F(f1,1))+V.row(F(f1,2)))/3+N.row(f1)*0.001;
          vr.data().add_edges(pt0,pt1,Eigen::RowVector3d(0,1,0));
        }
        //if(T(f0)==f1 || T(f1)==f0)
        //  vr.data().add_edges(V.row(a),V.row(b),Eigen::RowVector3d(0,1,0));
      }
    }
  }
}

void plot_highlight_faces(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<int>& L
){
  igl::opengl::glfw::Viewer vr;
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  vr.data().set_mesh(V,F);
  vr.plugins.push_back(&menu);
  Eigen::MatrixXd C(F.rows(),3);
  C.setConstant(1);
  for(int i=0;i<L.size();i++){
    C.row(L[i]) << 1,0,0;
  }
  vr.data().set_colors(C);
  vr.launch();
}

// void plot_adj(
//   const Eigen::MatrixXd& V,
//   const Eigen::MatrixXi& F,
//   const std::vector<bool>& I,
//   const std::vector<std::vector<std::vector<int>>>& Ax,
//   const std::set<std::pair<int,int>> edges,
//   const Eigen::VectorXd& sing,
//   const std::vector<int>& L,
//   const std::vector<std::vector<int>>& loops,
//   const std::vector<std::vector<int>>& paths,
//   const Eigen::MatrixXi& turning_n,
//   const Eigen::VectorXd& WT
// ){
  
//   igl::opengl::glfw::Viewer v;
//   igl::opengl::glfw::imgui::ImGuiMenu menu;
//   v.plugins.push_back(&menu);
//   EMap em;
//   init_EMap(F,em);
//   Eigen::MatrixXd N;
//   igl::per_face_normals(V,F,N);
//   v.data().set_mesh(V,F);
//   double avg_len = 0.0;
//   int num_edge = 0;
//   for(int i=0;i<F.rows();i++){
//     for(int k=0;k<3;k++){
//       int u = F(i,k);
//       int v = F(i,(k+1)%3);
//       if(u > v) continue;
//       avg_len += (V.row(u) - V.row(v)).norm();
//       num_edge++;
//     }
//   }
//   avg_len /= num_edge;
//   double radius = avg_len;
//   bool show_text = false;
//   bool show_bar = false;
//   bool show_dot = false;
//   auto key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
//     int id = key-'0';
//     bool show_path = true;
//     if(key == ' ') show_text = !show_text;
//     if(key == ',') show_bar = !show_bar;
//     if(key == '.') show_dot = !show_dot;
//     if(id >= Ax.size())
//       show_path = false;
//     v.data().clear();
//     v.data().set_mesh(V,F);
//     Eigen::MatrixXd C0 = Eigen::MatrixXd::Constant(F.rows(),3,1);
//     Eigen::VectorXi on_path(V.rows());
//     on_path.setZero();
//     if(show_path){
//       auto A = Ax[id];
//       bool is_dual = I[id];
//       for(int i=0;i<A.size();i++){
//         for(int j=0;j<A[i].size();j++){
//           if(!is_dual)
//             v.data().add_edges(V.row(i),V.row(A[i][j]),Eigen::RowVector3d(1,0,0));
//           else{
//             Eigen::RowVector3d a,b;
//             a = (V.row(F(i,0))+V.row(F(i,1))+V.row(F(i,2)))/3+N.row(i)*0.002;
//             b = (V.row(F(A[i][j],0))+V.row(F(A[i][j],1))+V.row(F(A[i][j],2)))/3+N.row(A[i][j])*0.002;
//             v.data().add_edges(a,b,Eigen::RowVector3d(0,1,0));
//           }
//         }
//       }
//     }
//     Eigen::MatrixXd V3;
//     Eigen::MatrixXi F3;
//     for(auto p: edges){
//       // if(show_bar){
//       //   v.erase_mesh(0);
//       //   Eigen::RowVector3d v0;
//       //   int a = p.first;
//       //   int b = p.second;
//       //   v0 << V(a,0),V(a,1),V(a,2);
//       //   Eigen::RowVector3d v1;
//       //   v1 << V(b,0),V(b,1),V(b,2);
//       //   Eigen::RowVector3d n1;
//       //   auto edge = em[std::make_pair(a,b)];
//       //   int fid = edge.first;
//       //   n1 << N(fid,0),N(fid,1),N(fid,2);
//       //   build_cylinder(V3,F3,v0,v1,n1,radius);
//       // }else{
//       //   if(v.data_list.size() > 1)
//       //     for(int i=1;i<v.data_list.size();i++)
//       //       v.erase_mesh(i);
//       // }
//       v.data().add_points(V.row(p.first),Eigen::RowVector3d(0,0,0));
//       v.data().add_points(V.row(p.second),Eigen::RowVector3d(0,0,0));
//       v.data().add_edges(V.row(p.first),V.row(p.second),Eigen::RowVector3d(0,0,0));
//     }
//     if(show_bar){
//       if(F3.rows() != 0){
//         v.append_mesh();
//         v.data().show_lines = false;
//         Eigen::MatrixXd C(F3.rows(),3);
//         for(int i=0;i<C.rows();i++)
//           C.row(i) << 0,1,1;
//         v.data().set_mesh(V3,F3);
//         v.data().set_colors(C);
//       }
//     }
//     if(show_dot){
//       for(int i=0;i<sing.rows();i++){
//         if(sing(i) > 1e-10 || sing(i) < -1e-10){
//           v.data().add_points(V.row(i),Eigen::RowVector3d(1,0,0));
//           auto str = std::to_string(sing(i));
//           str = str.substr(0, str.find_first_of('5')+1);
//           v.data().add_label(V.row(i),str);
//         }
//       }
//     }
//     Eigen::MatrixXd C(F.rows(),3);
//     C.setConstant(1);
//     for(int i=0;i<L.size();i++){
//       C.row(L[i]) << 1,0,0;
//     }
//     v.data().set_colors(C);
    
//     Eigen::MatrixXd V0;
//     Eigen::MatrixXi F0;
//     for(auto l: loops){
//       for(int i=0;i<l.size();i++){
//         int i_1 = (i+1) % l.size();
//         // v.data().add_edges(V.row(l[i]),V.row(l[i_1]),Eigen::RowVector3d(0,0,1));
//         on_path(l[i]) = 1;
//         if(show_bar){
//           v.erase_mesh(0);
//           Eigen::RowVector3d v0;
//           v0 << V(l[i],0),V(l[i],1),V(l[i],2);
//           Eigen::RowVector3d v1;
//           v1 << V(l[i_1],0),V(l[i_1],1),V(l[i_1],2);
//           Eigen::RowVector3d n1;
//           auto edge = em[std::make_pair(l[i],l[i_1])];
//           int fid = edge.first;
//           n1 << N(fid,0),N(fid,1),N(fid,2);
//           build_cylinder(v0,v1,n1,radius,V0,F0);
//         }else{
//           if(v.data_list.size() > 1)
//             for(int i=1;i<v.data_list.size();i++)
//               v.erase_mesh(i);
//         }
//       }
//     }
//     if(show_bar){
//       if(F0.rows() != 0){
//         v.append_mesh();
//         v.data().show_lines = false;
//         Eigen::MatrixXd C(F0.rows(),3);
//         for(int i=0;i<C.rows();i++)
//           C.row(i) << 0,0,1;
//         v.data().set_mesh(V0,F0);
//         v.data().set_colors(C);
//       }
//     }
    
      
//     Eigen::MatrixXd V1;
//     Eigen::MatrixXi F1;
//     if(id < paths.size()){
//       auto l = paths[id];
//       std::cout<<"plot connector #"<<id<<std::endl;
//       if(!l.empty()){
//         for(int i=0;i<l.size()-1;i++){
//           int i_1 = (i+1) % l.size();
//           // v.data().add_edges(V.row(l[i]),V.row(l[i_1]),Eigen::RowVector3d(1,0,0));
//           if(show_bar){
//             Eigen::RowVector3d v0;
//             v0 << V(l[i],0),V(l[i],1),V(l[i],2);
//             Eigen::RowVector3d v1;
//             v1 << V(l[i_1],0),V(l[i_1],1),V(l[i_1],2);
//             Eigen::RowVector3d n1;
//             auto edge = em[std::make_pair(l[i],l[i_1])];
//             int fid = edge.first;
//             n1 << N(fid,0),N(fid,1),N(fid,2);
//             build_cylinder(v0,v1,n1,radius,V1,F1);
//           }
//         }
//         if(show_bar){
//           if(F1.rows()!=0){
//             v.append_mesh();
//             v.data().show_lines = false;
//             Eigen::MatrixXd C(F1.rows(),3);
//             for(int i=0;i<C.rows();i++)
//               C.row(i) << 1,0,0;
//             v.data().set_mesh(V1,F1);
//             v.data().set_colors(C);
//           }
//         }
//       }
//     }else{
//       for(auto l: paths){
//         if(l.empty()) continue;
//         for(int i=0;i<l.size()-1;i++){
//           int i_1 = (i+1) % l.size();
//           on_path(l[i]) = 1;
//           on_path(l[i_1]) = 1;
//           // v.data().add_edges(V.row(l[i]),V.row(l[i_1]),Eigen::RowVector3d(1,0,0));
//           if(show_bar){
//             Eigen::RowVector3d v0;
//             v0 << V(l[i],0),V(l[i],1),V(l[i],2);
//             Eigen::RowVector3d v1;
//             v1 << V(l[i_1],0),V(l[i_1],1),V(l[i_1],2);
//             Eigen::RowVector3d n1;
//             auto edge = em[std::make_pair(l[i],l[i_1])];
//             int fid = edge.first;
//             n1 << N(fid,0),N(fid,1),N(fid,2);
//             build_cylinder(v0,v1,n1,radius,V1,F1);
//           }else{
//             if(v.data_list.size() > 1)
//               for(int i=1;i<v.data_list.size();i++)
//                 v.erase_mesh(i);
//           }
//         }
//       }
//       if(show_bar){
//         if(F1.rows()!=0){
//           v.append_mesh();
//           v.data().show_lines = false;
//           Eigen::MatrixXd C(F1.rows(),3);
//           for(int i=0;i<C.rows();i++)
//             C.row(i) << 1,0,0;
//           v.data().set_mesh(V1,F1);
//           v.data().set_colors(C);
//         }
//       }
//     }
    
//     if(show_text){
//       for(int i=0;i<F.rows();i++){
//         for(int k=0;k<3;k++){
//           int e = F.rows()*k+i;
//           auto pt = (V.row(F(i,k)) + V.row(F(i,(k+1)%3))) / 2;
//           if(F(i,k) > F(i,(k+1) % 3) && turning_n(i,k) != 0){
//             v.data().add_label(pt,std::to_string(turning_n(i,k)));
//             v.data().add_edges(V.row(F(i,k)),V.row(F(i,(k+1)%3)),Eigen::RowVector3d(1,0,0));
//           }
//         }
//       }
//       // for(int i=0;i<V.rows();i++){
//       //   if(on_path(i) == 1)
//       //     //v.data().add_label(pt,std::to_string(turning_n(i,k)));
//       //     v.data().add_label(V.row(i),std::to_string( int(WT(i)) ));
//       // }
//     }

    
//     return false;
//   };
//   v.callback_key_down = key_down;
//   v.launch();
// }

void plot_loops(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const std::vector<std::vector<int>>& loops,
  const Eigen::VectorXi& Hv,
  const Eigen::VectorXi& Hf
){
  igl::opengl::glfw::Viewer vr;
  vr.data().set_mesh(V,F);
  for(auto loop: loops){
    for(int i=0;i<loop.size();i++){
      int i_1 = (i+1) % loop.size();
      vr.data().add_edges(V.row(loop[i]),V.row(loop[i_1]),Eigen::RowVector3d(1,0,0));
    }
  }
  for(int i=0;i<Hv.rows();i++){
    if(Hv(i) == 1)
      vr.data().add_points(V.row(i),Eigen::RowVector3d(0,0,0));
  }
  int face_selected = -1;
  
  Eigen::MatrixXd C(F.rows(),3);
  C.setConstant(1);
  for(int i=0;i<Hf.rows();i++){
    if(Hf(i) == 1)
      C.row(i) << 1,0,0;
  }
  vr.callback_mouse_down =
  [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
  {
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if(igl::unproject_onto_mesh(
      Eigen::Vector2f(x,y),
      viewer.core().view,
      viewer.core().proj,
      viewer.core().viewport,
      V,
      F,
      fid,
      bc))
    {
      if(face_selected!=-1){
        C.row(face_selected) << 1,1,1;
      }
      C.row(fid) << 0,1,0;
      vr.data().set_colors(C);
      face_selected = fid;
      std::cout<<"fid: "<<fid<<"("<<F.row(fid)<<")"<<std::endl;
    }
    return false;
  };
  vr.data().set_colors(C);
  vr.launch();
}

void plot_pj(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& PJ
){
    igl::opengl::glfw::Viewer vr;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    vr.plugins.push_back(&menu);
    vr.data().set_mesh(V,F);
    for(int i=0;i<PJ.rows();i++){
        Eigen::RowVector3d c = (V.row(F(i,0))+V.row(F(i,1))+V.row(F(i,2)))/3;
        for(int k=0;k<3;k++){
            Eigen::RowVector3d d = (V.row(F(i,k))+V.row(F(i,(k+1)%3)))/2;
            vr.data().add_label((c+d)/2,std::to_string(PJ(i,k)));
        }
    }
    vr.launch();
}

void plot_TT(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXi& TT,
  const Eigen::VectorXi& I_vertex,
  const std::vector<std::vector<int>>& basis,
  const std::vector<std::vector<int>>& connectors,
  const std::vector<int>& dropped,
  int genus
){
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V,F);
  
  bool show_dropped = true;
  bool show_TT = false;
  int edge_num = 0;
  for(int i=0;i<TT.rows();i++){
    for(int k=0;k<3;k++){
      if(TT(i,k) != -1){
        edge_num++;
      }
    }
  }
  auto key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier){
    if(key == ' '){
      viewer.data().clear();
      viewer.data().set_mesh(V,F);
      show_dropped = !show_dropped;
    }

    if(key == ','){
      viewer.data().clear();
      viewer.data().set_mesh(V,F);
      show_TT = !show_TT;
      Eigen::MatrixXd P1(edge_num,V.cols()), P2(edge_num,V.cols());
      int c = 0;
      if(show_TT){
        for(int i=0;i<TT.rows();i++){
          for(int k=0;k<3;k++){
            if(TT(i,k) != -1){
              int k_1 = (k+1) % 3;
              P1.row(c) << V.row(F(i,k));
              P2.row(c) << V.row(F(i,k_1));
              c++;
            }
          }
        }
        viewer.data().add_edges(P1,P2,Eigen::RowVector3d(0,0,0));
      }
    }

    
    for(int i=0;i<V.rows();i++){
      if(I_vertex.rows() != V.rows()) break;
      if(I_vertex(i) > 0)
        viewer.data().add_points(V.row(i),Eigen::RowVector3d(0,1,0));
    }
    
    for(int i=0;i<basis.size();i++){
      auto cut = basis[i];
      for(int j=0;j<cut.size();j++){
        int j_1 = (j+1) % cut.size();
        viewer.data().add_edges(V.row(cut[j]),V.row(cut[j_1]),Eigen::RowVector3d(0,1,0));
      }
    }
    for(int i=0;i<connectors.size();i++){
      auto cut = connectors[i];
      for(int j=0;j<cut.size()-1;j++){
        viewer.data().add_edges(V.row(cut[j]),V.row(cut[j+1]),Eigen::RowVector3d(0,0,1));
      }
    }
    if(show_dropped && dropped.size()!=0){
      for(int i=0;i<dropped.size()-1;i++){
        viewer.data().add_edges(V.row(dropped[i]),V.row(dropped[i+1]),Eigen::RowVector3d(1,0,0));
      }
    }
    Eigen::MatrixXd C(F.rows(),3);
    C.setConstant(1);
    viewer.data().set_colors(C);
   
    return false;
  };
  viewer.callback_key_down=key_down;

  
  viewer.launch();
}

void plot_paths(
    std::map<std::pair<int,int>,std::pair<int,int>>& branch,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F
){
    igl::opengl::glfw::Viewer vr;
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    vr.plugins.push_back(&menu);
    static int vid = 0;
    static int pj = 0;
    static int root = 0;
    menu.callback_draw_viewer_menu = [&](){
        // Add new group
        if (ImGui::CollapsingHeader("show paths", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputInt("root",&root);
            ImGui::InputInt("vid",&vid);
            ImGui::InputInt("pj",&pj);
            // We can also use a std::vector<std::string> defined dynamically
            static int num_choices = 3;
            static std::vector<std::string> choices;
            static int idx_choice = 0;
            std::vector<std::pair<int,int>> path;
            if (ImGui::Button("update path", ImVec2(-1,0))){
                // check whether the combination is valid
                if(branch.find(std::make_pair(vid,pj)) == branch.end()){
                    std::cout<<"path does not exist\n";
                }else{
                    vr.data().clear();
                    vr.data().set_mesh(V,F);
                    vr.data().add_points(V.row(root),Eigen::RowVector3d(0,0,0));
                    vr.data().add_points(V.row(vid),Eigen::RowVector3d(0,1,0));
                    // traverse to root
                    // auto parent = branch[std::make_pair(vid,pj)];
                    auto node = std::make_pair(vid,pj);
                    while(node.first!=root){
                        // std::cout<<parent.first<<","<<parent.second<<std::endl;
                        vr.data().add_points(V.row(node.first),Eigen::RowVector3d(1,0,0));
                        auto prev = node;
                        node = branch[node];
                        vr.data().add_edges(V.row(prev.first),V.row(node.first),Eigen::RowVector3d(0,0,0));
                    }
                }
            }
        }
    };
    vr.data().set_mesh(V,F);
    vr.launch();
}

template void plot_label_on_halfedges<Eigen::Matrix<double, -1, -1, 0, -1, -1> >(igl::opengl::glfw::Viewer&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, -1, 0, -1, -1> > const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&);
template void plot_label_on_halfedges<Eigen::Matrix<int, -1, -1, 0, -1, -1> >(igl::opengl::glfw::Viewer&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::MatrixBase<Eigen::Matrix<int, -1, -1, 0, -1, -1> > const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&);
template void plot_label_on_face<Eigen::Matrix<double, -1, 1, 0, -1, 1> >(igl::opengl::glfw::Viewer&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<int, -1, -1, 0, -1, -1> const&, Eigen::MatrixBase<Eigen::Matrix<double, -1, 1, 0, -1, 1> > const&, Eigen::Matrix<int, -1, 1, 0, -1, 1> const&);
