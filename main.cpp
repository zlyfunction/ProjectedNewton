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
#include <igl/facet_components.h>
#include <igl/remove_unreferenced.h>
// #include <igl/copyleft/cgal/orient2D.h>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "projected_newton.hpp"
void construct_Fn(Eigen::MatrixXi &Fn, const Eigen::MatrixXi &F_copy, const std::vector<Eigen::Vector3i> &new_face_list, const std::vector<int> &delete_faces)
{

    Fn.resize(F_copy.rows() - delete_faces.size() + new_face_list.size(), F_copy.cols());
    int tmp = -1;
    if (delete_faces.size() >= 1)
    {
        for (int j = 0; j < delete_faces[0]; j++)
        {
            tmp++;
            Fn.row(tmp) = F_copy.row(j);
        }
        for (int i = 0; i < delete_faces.size() - 1; i++)
        {
            for (int j = delete_faces[i] + 1; j < delete_faces[i + 1]; j++)
            {
                tmp++;
                Fn.row(tmp) = F_copy.row(j);
            }
        }
        for (int j = delete_faces.back() + 1; j < F_copy.rows(); j++)
        {
            tmp++;
            Fn.row(tmp) = F_copy.row(j);
        }
    }
    else
    {
        for (int j = 0; j < F_copy.rows(); j++)
        {
            tmp++;
            Fn.row(tmp) = F_copy.row(j);
        }
    }

    for (int i = 0; i < new_face_list.size(); i++)
    {
        tmp++;
        Fn.row(tmp) = new_face_list[i];
    }
}
void prepare(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, spXd &Dx,
             spXd &Dy)
{
    Eigen::MatrixXd F1, F2, F3;
    igl::local_basis(V, F, F1, F2, F3);
    Eigen::SparseMatrix<double> G;
    igl::grad(V,F,G);
    // igl::grad(V, F, G, true); // use uniform mesh instead of V
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

void buildAeq(
    const Eigen::MatrixXi &cut,
    const Eigen::MatrixXd &uv,
    Eigen::SparseMatrix<double> &Aeq)
{
    Eigen::VectorXd tail;
    int N = uv.rows();
    int c = 0;
    int m = cut.rows();
    Aeq.resize(2 * m, uv.rows() * 2);
    int A, B, C, D, A2, B2, C2, D2;
    for (int i = 0; i < cut.rows(); i++)
    {
        int A2 = cut(i, 0);
        int B2 = cut(i, 1);
        int C2 = cut(i, 2);
        int D2 = cut(i, 3);

        std::complex<double> l0, l1, r0, r1;
        l0 = std::complex<double>(uv(A2, 0), uv(A2, 1));
        l1 = std::complex<double>(uv(B2, 0), uv(B2, 1));
        r0 = std::complex<double>(uv(C2, 0), uv(C2, 1));
        r1 = std::complex<double>(uv(D2, 0), uv(D2, 1));

        int r = std::round(2.0 * std::log((l0 - l1) / (r0 - r1)).imag() / igl::PI);
        r = ((r % 4) + 4) % 4; // ensure that r is between 0 and 3
        switch (r)
        {
        case 0:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2) += -1;
            Aeq.coeffRef(c, D2) += 1;
            Aeq.coeffRef(c + 1, A2 + N) += 1;
            Aeq.coeffRef(c + 1, B2 + N) += -1;
            Aeq.coeffRef(c + 1, C2 + N) += -1;
            Aeq.coeffRef(c + 1, D2 + N) += 1;
            c = c + 2;
            break;
        case 1:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2 + N) += 1;
            Aeq.coeffRef(c, D2 + N) += -1;
            Aeq.coeffRef(c + 1, C2) += 1;
            Aeq.coeffRef(c + 1, D2) += -1;
            Aeq.coeffRef(c + 1, A2 + N) += -1;
            Aeq.coeffRef(c + 1, B2 + N) += 1;
            c = c + 2;
            break;
        case 2:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2) += 1;
            Aeq.coeffRef(c, D2) += -1;
            Aeq.coeffRef(c + 1, A2 + N) += 1;
            Aeq.coeffRef(c + 1, B2 + N) += -1;
            Aeq.coeffRef(c + 1, C2 + N) += 1;
            Aeq.coeffRef(c + 1, D2 + N) += -1;
            c = c + 2;
            break;
        case 3:
            Aeq.coeffRef(c, A2) += 1;
            Aeq.coeffRef(c, B2) += -1;
            Aeq.coeffRef(c, C2 + N) += -1;
            Aeq.coeffRef(c, D2 + N) += 1;
            Aeq.coeffRef(c + 1, C2) += 1;
            Aeq.coeffRef(c + 1, D2) += -1;
            Aeq.coeffRef(c + 1, A2 + N) += 1;
            Aeq.coeffRef(c + 1, B2 + N) += -1;
            c = c + 2;
            break;
        }
    }
    Aeq.conservativeResize(c, uv.rows() * 2);
    Aeq.makeCompressed();
    std::cout << "Aeq size " << Aeq.rows() << "," << Aeq.cols() << std::endl;
    // test initial violation
    Eigen::VectorXd UV(uv.rows() * 2);
    UV << uv.col(0), uv.col(1);
    Eigen::SparseMatrix<double> t = UV.sparseView();
    t.makeCompressed();
    Eigen::SparseMatrix<double> mm = Aeq * t;
    Eigen::VectorXd z = Eigen::VectorXd(mm);
    if (z.rows() > 0)
        std::cout << "max violation " << z.cwiseAbs().maxCoeff() << std::endl;
}

void buildkkt(spXd &hessian, spXd &Aeq, spXd &AeqT, spXd &kkt)
{
    std::cout << "build kkt\n";
    kkt.reserve(hessian.nonZeros() + Aeq.nonZeros() + AeqT.nonZeros());
    for (Eigen::Index c = 0; c < kkt.cols(); ++c)
    {
        kkt.startVec(c);
        if (c < hessian.cols())
        {
            for (Eigen::SparseMatrix<double>::InnerIterator ithessian(hessian, c); ithessian; ++ithessian)
            kkt.insertBack(ithessian.row(), c) = ithessian.value();
            for (Eigen::SparseMatrix<double>::InnerIterator itAeq(Aeq, c); itAeq; ++itAeq)
            kkt.insertBack(itAeq.row() + hessian.rows(), c) = itAeq.value();
        }
        else
        {
            for (Eigen::SparseMatrix<double>::InnerIterator itAeqT(AeqT, c - hessian.cols()); itAeqT; ++itAeqT)
            kkt.insertBack(itAeqT.row(), c) = itAeqT.value();
        }
        
        
    }
    kkt.finalize();
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
    Xi cut;

    std::string model = argv[1];
    igl::deserialize(F, "F", model);
    igl::deserialize(uv_init, "uv", model);
    igl::deserialize(V, "V", model);

    // deserialize cut
    igl::deserialize(cut, "cut", model);
    spXd Aeq;
    buildAeq(cut, uv_init, Aeq);
    spXd AeqT = Aeq.transpose();
    Vd dblarea_uv;
    igl::doublearea(uv_init, F, dblarea_uv);
    igl::writeOBJ("input_init.obj", V, F, CN, FN, uv_init, F);
    // std::cout << "#fl: " << check_flip(uv_init, F) << std::endl;
    // return 0;
    for (int i = 0; i < F.rows(); i++)
    {
        double max_e = (uv_init(F(i, 0), 0) - uv_init(F(i, 1), 0)) * (uv_init(F(i, 0), 0) - uv_init(F(i, 1), 0)) + (uv_init(F(i, 0), 1) - uv_init(F(i, 1), 1)) * (uv_init(F(i, 0), 1) - uv_init(F(i, 1), 1));
        for (int j = 1; j < 3; j++)
        {
            double tmp = (uv_init(F(i, j), 0) - uv_init(F(i, (j + 1) % 3), 0)) * (uv_init(F(i, j), 0) - uv_init(F(i, (j + 1) % 3), 0)) + (uv_init(F(i, j), 1) - uv_init(F(i, (j + 1) % 3), 1)) * (uv_init(F(i, j), 1) - uv_init(F(i, (j + 1) % 3), 1));
            if (tmp > max_e)
            {
                max_e = tmp;
            }
        }
        std::cout << std::setprecision(17) << max_e / dblarea_uv(i) << std::endl;
    }

    /*
    // use only one compo
    Eigen::VectorXi C;
    igl::facet_components(F, C);
    std::vector<int> delete_face_list;
    std::vector<Eigen::Vector3i> new_face_list;

    for (int i = 0; i < C.size(); i++)
    {
        if (C(i) == 1) delete_face_list.push_back(i);
    }
    auto F_copy = F;
    construct_Fn(F, F_copy, new_face_list, delete_face_list);
    Eigen::VectorXi II, JJ;
    Eigen::VectorXi tmp(3);
    tmp << 0, 1, 2;
    auto uv_copy = uv_init;
    auto V_copy = V;
    F_copy = F;
    igl::remove_unreferenced(uv_copy, F_copy, uv_init, F, II, JJ);
    igl::slice(V_copy, JJ, tmp, V);
    igl::writeOBJ("in_ap_oneop0.obj", V, F, CN, FN, uv_init, F);
    return 0;
    */

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
    double lambda = 0.999;

    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    for (int ii = start_iter + 1; ii < 100000; ii++)
    {
        spXd hessian;
        Vd grad;
        std::cout << "\nIt" << ii << std::endl;
        double e1 = get_grad_and_hessian(G, dblarea, cur_uv, grad, hessian);
        // std::cout << cur_uv.rows() << " " << hessian.rows() << std::endl;
        spXd Id(hessian.rows(), hessian.cols());
        Id.setIdentity();
        hessian = lambda * hessian + (1 - lambda) * Id;
        spXd kkt(hessian.rows() + Aeq.rows(), hessian.cols() + Aeq.rows());
        buildkkt(hessian, Aeq, AeqT, kkt);

        // spXd kkt = hessian;
        // kkt.conservativeResize(hessian.rows() + Aeq.cols(), hessian.rows() + Aeq.rows());
        // kkt.block(0, hessian.cols() + 1, AeqT.rows(), AeqT.cols()) = AeqT;
        // kkt.block(hessian.rows() + 1, 0, Aeq.rows(), Aeq.cols()) = Aeq;
        if (ii == start_iter + 1)
        {
            // solver.analyzePattern(hessian);
            solver.analyzePattern(kkt);
            // std::cout << std::setprecision(17);
            // std::cout << "Hessian at beginning: " << std::endl;
            // std::cout << "Row\tCol\tVal" << std::endl;
            //     for (int k = 0; k < hessian.outerSize(); ++k)
            //     {
            //         for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it; ++it)
            //         {
            //             std::cout << 1 + it.row() << "\t"; // row index
            //             std::cout << 1 + it.col() << "\t"; // col index (here it is equal to k)
            //             std::cout << it.value() << std::endl;
            //         }
            //     }
            //     std::cout << grad << std::endl;
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
            // solver.factorize(hessian);
            // Vd newton = solver.solve(grad);
            grad.conservativeResize(kkt.cols());
            for (int i = hessian.cols() + 1; i < kkt.cols(); i++)
            {
                grad(i) = 0;
            }
            solver.factorize(kkt);
            Vd newton = solver.solve(grad);
            newton.conservativeResize(hessian.cols());
            grad.conservativeResize(hessian.cols());
            
            if (solver.info() != Eigen::Success)
            {
                std::cout << "solver.info() = " << solver.info() << std::endl;
                std::cout << "start using gd\n";
                // use_gd = true;
                // exit(1);
            }
            new_dir = -Eigen::Map<Xd>(newton.data(), V.rows(), 2); // newton
            std::cout << "<grad, newton> = " << acos(newton.dot(grad) / newton.norm() / grad.norm()) << "\n";
        }

        // energy = wolfe_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy, use_gd);
        energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, grad, energy, use_gd);

        std::cout << std::setprecision(20)
                  << "E=" << compute_energy(cur_uv) << "\t\tE_max=" << compute_energy_max(cur_uv)
                  << "\n |new_dir|=" << new_dir.norm() << "\t|grad|=" << grad.norm() << std::endl;
        std::cout << "#fl = " << check_flip(cur_uv, F) << std::endl;
        std::cout << "lambda = " << lambda << std::endl;
        if (std::abs(energy - 4) < 1e-10)
            // norm of the grad
            // if (std::abs(energy - old_energy) < 1e-9)
            break;
        if (fabs(old_energy - energy) < 1e-16)
        {
            lambda = lambda * 0.8;
            std::cout << "lambda ->" << lambda << std::endl;
        }
        else
        {
            lambda = lambda / 0.8 > 0.99? 0.99:lambda/0.8;
            std::cout << "lambda ->" << lambda << std::endl;
        }
        
        old_energy = energy;

        // save the cur_uv
        std::string outfilename = "./serialized/cur_uv_step" + std::to_string(ii);
        igl::serialize(cur_uv, "cur_uv", outfilename, true);
    }

    std::cout << "write mesh\n";
    igl::writeOBJ("out.obj", V, F, CN, FN, cur_uv, F);
    // std::cout << compute_energy_all(cur_uv) << std::endl;
}
