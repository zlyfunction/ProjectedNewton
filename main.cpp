#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/flip_avoiding_line_search.h>
#include <igl/grad.h>
#include <igl/harmonic.h>
#include <igl/local_basis.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/matrix_to_list.h>
#include <igl/read_triangle_mesh.h>
#include <igl/serialize.h>
#include <igl/writeDMAT.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/write_triangle_mesh.h>
// #include <igl/copyleft/cgal/orient2D.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "projected_newton.hpp"


void prepare(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, spXd &Dx,
             spXd &Dy)
{
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);
    auto face_proj = [](Eigen::MatrixXd &F) {
        std::vector<Eigen::Triplet<double>> IJV;
        int f_num = F.rows();
        for (int i = 0; i < F.rows(); i++)
        {
            IJV.push_back(Eigen::Triplet<double>(i, i, F(i, 0)));
            IJV.push_back(Eigen::Triplet<double>(i, i + f_num, F(i, 1)));
            IJV.push_back(Eigen::Triplet<double>(i, i + 2 * f_num, F(i, 2)));
        }
        Eigen::SparseMatrix<double> P(f_num, 3 * f_num);
        P.setFromTriplets(IJV.begin(), IJV.end());
        return P;
    };

    Dx = face_proj(F1) * G;
    Dy = face_proj(F2) * G;
}

spXd combine_Dx_Dy(const spXd &Dx, const spXd &Dy)
{
    // [Dx, 0; Dy, 0; 0, Dx; 0, Dy]
    spXd hstack = igl::cat(1, Dx, Dy);
    spXd empty(hstack.rows(), hstack.cols());
    // gruesom way for Kronecker product.
    return igl::cat(1, igl::cat(2, hstack, empty), igl::cat(2, empty, hstack));
}

#include <igl/slim.h>
Xd timing_slim(const Xd &V, const Xi &F, const Xd &uv)
{
    igl::SLIMData data;
    Eigen::VectorXi b;
    Xd bc;
    igl::Timer timer;
    timer.start();
    igl::slim_precompute(V, F, uv, data,
                         igl::MappingEnergyType::SYMMETRIC_DIRICHLET, b, bc, 0.);
    for (int i = 0; i < 100; i++)
    {
        igl::slim_solve(data, 1);
        std::cout << "SLIM e=" << data.energy
                  << "\tTimer:" << timer.getElapsedTime() << std::endl;
    }
    return data.V_o;
}

long global_autodiff_time = 0;
long global_project_time = 0;
int main(int argc, char *argv[])
{
    Xd V;
    Xi F;
    Xd uv_init;
    Eigen::VectorXi bnd;
    Xd bnd_uv;
    double mesh_area;
    Xd CN;
    Xi FN, FTC;

    std::string model = argv[1];
    igl::deserialize(F, "F", model);
    igl::deserialize(uv_init, "uv", model);
    igl::deserialize(V, "V", model);
    Vd dblarea_uv;
    igl::doublearea(uv_init, F, dblarea_uv);
    // for (int i = 0; i < F.rows(); i++)
    // {
    //     double max_e = (uv_init(F(i,0), 0) - uv_init(F(i, 1), 0)) * (uv_init(F(i,0), 0) - uv_init(F(i, 1), 0)) + (uv_init(F(i,0), 1) - uv_init(F(i, 1), 1)) * (uv_init(F(i,0), 1) - uv_init(F(i, 1), 1));
    //     for (int j = 1; j < 3; j++)
    //     {
    //         double tmp = (uv_init(F(i,j), 0) - uv_init(F(i, (j+1)%3), 0)) * (uv_init(F(i,j), 0) - uv_init(F(i, (j+1)%3), 0)) + (uv_init(F(i,j), 1) - uv_init(F(i, (j+1)%3), 1)) * (uv_init(F(i,j), 1) - uv_init(F(i, (j+1)%3), 1));
    //         if (tmp > max_e)
    //         {
    //             max_e = tmp;
    //         }
    //     }
    //     std::cout << std::setprecision(17) << max_e / dblarea_uv(i) << std::endl;
    // }
    // return 0;
    
    // igl::read_triangle_mesh(argv[1], V, F);
    // igl::readOBJ(model, V, uv_init, CN, F, FTC, FN);

    // igl::boundary_loop(F, bnd);
    // igl::map_vertices_to_circle(V, bnd, bnd_uv);
    // igl::harmonic(V, F, bnd, bnd_uv, 1, uv_init);
    // igl::writeOBJ("in.obj", V, F, CN, FN, uv_init, F);
    Vd dblarea;
    igl::doublearea(V, F, dblarea);
    dblarea *= 0.5;
    mesh_area = dblarea.sum();

    spXd Dx, Dy, G;
    prepare(V, F, Dx, Dy);
    G = combine_Dx_Dy(Dx, Dy);
    
    int start_iter = 0;
    Xd cur_uv;
    if (argc > 2)
    {
        std::string s = argv[2];
        start_iter = std::atoi(argv[2]);
        std::string infilename = "./serialized/cur_uv_step" + std::to_string(start_iter);
        igl::deserialize(cur_uv, "cur_uv", infilename);
    }
    else
    {
        cur_uv = uv_init;
    }

    auto compute_energy = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return compute_energy_from_jacobian(Ji, dblarea);
    };

    auto compute_energy_max = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        auto E = symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
        double max_e = E(0);
        for (int i = 1; i < E.size(); i++)
        {
            if (E(i) > max_e)
            {
                max_e = E(i);
            }
        }
        return max_e;
    };

    auto compute_energy_all = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    };

    double energy = compute_energy(cur_uv);
    std::cout << "Start Energy" << energy << std::endl;
    double old_energy = energy;

    bool use_gd = false;
    double lambda = 1;

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    for (int ii = start_iter + 1; ii < 100000; ii++)
    {
        spXd hessian;
        Vd grad;
        std::cout << "\nIt" << ii << std::endl;
        double e1 = get_grad_and_hessian(G, dblarea, cur_uv, grad, hessian);
        
        // spXd Id(hessian.rows(), hessian.cols());
        // Id.setIdentity();
        // hessian = hessian + lambda * Id;
        
        if (ii == start_iter + 1)
        {
            solver.analyzePattern(hessian);
            std::cout << std::setprecision(17);
            std::cout << "Hessian at beginning: " << std::endl;
            std::cout << "Row\tCol\tVal" << std::endl;
                for (int k = 0; k < hessian.outerSize(); ++k)
                {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it; ++it)
                    {
                        std::cout << 1 + it.row() << "\t"; // row index
                        std::cout << 1 + it.col() << "\t"; // col index (here it is equal to k)
                        std::cout << it.value() << std::endl;
                    }
                }
                std::cout << grad << std::endl;
        }
        // if (ii > 1000)
        // {
        //   use_gd = true;
        //   std::cout << "change to gd\n";
        // }

        Xd new_dir;
        if (use_gd)
        {
            new_dir = -Eigen::Map<Xd>(grad.data(), V.rows(), 2); // gradient descent
        }
        else
        {
            solver.factorize(hessian);
            Vd newton = solver.solve(grad);

            if (solver.info() != Eigen::Success)
            {
                std::cout << "solver.info() = " << solver.info() << std::endl;
                std::cout << "start using gd\n";
                // use_gd = true;
                // exit(1);
            }
            new_dir = -Eigen::Map<Xd>(newton.data(), V.rows(), 2); // newton
            std::cout << "<grad, newton> = "<<acos(newton.dot(grad)/newton.norm()/grad.norm()) << "\n";
        }

        
        // energy = wolfe_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy, use_gd);
        energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy, use_gd);

        std::cout << std::setprecision(20)
                  << "E=" << compute_energy(cur_uv) << "\t\tE_max=" << compute_energy_max(cur_uv)
                  << "\n |new_dir|=" << new_dir.norm() << "\t|grad|=" << grad.norm() << std::endl;
        std::cout << "#fl = " << check_flip(cur_uv, F) << std::endl;

        if (std::abs(energy - 4) < 1e-10)
            // norm of the grad
            // if (std::abs(energy - old_energy) < 1e-9)
            break;
        old_energy = energy;

        // save the cur_uv
        std::string outfilename = "./serialized/cur_uv_step" + std::to_string(ii);
        igl::serialize(cur_uv, "cur_uv", outfilename, true);
    }

    std::cout << "write mesh\n";
    igl::writeOBJ("out.obj", V, F, CN, FN, cur_uv, F);
    // std::cout << compute_energy_all(cur_uv) << std::endl;
}
