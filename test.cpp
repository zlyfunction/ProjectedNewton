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

#include <fstream>
#include <igl/triangle_triangle_adjacency.h>
void prepare(const Eigen::MatrixXd &V, Eigen::MatrixXi &F, const std::vector<double> &trg, const Vd &area, spXd &Dx,
             spXd &Dy, bool uniform)
{
    const double eps = 1e-8;
    Eigen::MatrixXd F1(F.rows(), 3), F2(F.rows(), 3), F3(F.rows(), 3);
    igl::local_basis(V, F, F1, F2, F3);
    Eigen::SparseMatrix<double> G;

    if (uniform)
    {
        igl::grad(V, F, G, true); // use uniform mesh instead of V
    }
    else
    {
        igl::grad_plastic(V, F, trg, G);
    }

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
    if (uniform)
    {
        F1.col(0).setConstant(1);
        F1.col(1).setConstant(0);
        F1.col(2).setConstant(0);
        F2.col(0).setConstant(0);
        F2.col(1).setConstant(1);
        F2.col(2).setConstant(0);
    }
    else
    {
        for (int i = 0; i < F.rows(); i++)
        {
            if (trg[i] != 0 || area(i) < eps)
            {
                F1(i, 0) = 1;
                F1(i, 1) = 0;
                F1(i, 2) = 0;
                F2(i, 0) = 0;
                F2(i, 1) = 1;
                F2(i, 2) = 0;
            }
        }
    }

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

void buildAeq(
    const Eigen::MatrixXi &cut,
    const Eigen::MatrixXd &uv,
    const Eigen::MatrixXi &F,
    Eigen::SparseMatrix<double> &Aeq)
{
    std::cout << "build constraint matrix\n";
    Eigen::VectorXd tail;
    int N = uv.rows();
    int c = 0;
    int m = cut.rows();

    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);

    std::cout << "#components = " << bds.size() << std::endl;
    // Aeq.resize(2 * m, uv.rows() * 2);
    // try to fix 2 dof for each component
    Aeq.resize(2 * m + 2 * bds.size(), uv.rows() * 2);

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
    // add 2 constraints for each component
    for (auto l : bds)
    {
        std::cout << "fix " << l[0]  << std::endl;
        Aeq.coeffRef(c, l[0]) = 1;
        Aeq.coeffRef(c + 1, l[0] + N) = 1;
        c = c + 2;
    }

    Aeq.makeCompressed();
    std::cout << "Aeq size " << Aeq.rows() << "," << Aeq.cols() << std::endl;
    // test initial violation
    // Eigen::VectorXd UV(uv.rows() * 2);
    // UV << uv.col(0), uv.col(1);
    // Eigen::SparseMatrix<double> t = UV.sparseView();
    // t.makeCompressed();
    // Eigen::SparseMatrix<double> mm = Aeq * t;
    // Eigen::VectorXd z = Eigen::VectorXd(mm);
    // if (z.rows() > 0)
    //     std::cout << "max violation " << z.cwiseAbs().maxCoeff() << std::endl;
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
    std::cout << "finish build" << std::endl;
}

long global_autodiff_time = 0;
long global_project_time = 0;

void write_hessian_to_file(const spXd &hessian, const std::string filename)
{
    std::ofstream writeH;
    writeH.open(filename);
    writeH << std::setprecision(20);
    for (int k = 0; k < hessian.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it; ++it)
        {
            writeH << 1 + it.row() << "\t"; // row index
            writeH << 1 + it.col() << "\t"; // col index (here it is equal to k)
            writeH << it.value() << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    Xd V;
    Xi F;
    Xd uv_init;
    Eigen::VectorXi bnd;
    Xd bnd_uv;
    double mesh_area;
    Xd CN, scale;
    Xi FN, FTC;
    Xi cut;

    const double eps = 1e-8;

    std::string model = argv[1];
    igl::deserialize(F, "Fuv", model);
    igl::deserialize(uv_init, "uv", model);
    igl::deserialize(V, "V", model);
    V.conservativeResize(uv_init.rows(), 3);
    for (int i = 0; i < V.rows(); i++)
    {
        for (int j = 0; j < V.cols(); j++)
        {
            if (abs(V(i, j)) < 1e-15)
                V(i, j) = 0;
        }
    }

    // deserialize cut
    igl::deserialize(cut, "cut", model);
    // igl::deserialize(scale, "scale", model);
    std::cout << F.rows() << " " << uv_init.rows() << " " << V.rows() << " " << cut.rows() << std::endl;
    spXd Aeq;
    buildAeq(cut, uv_init, F, Aeq);

    spXd AeqT = Aeq.transpose();
    Vd dblarea_uv;
    igl::doublearea(uv_init, F, dblarea_uv);
    igl::writeOBJ("input_init.obj", V, F, CN, FN, uv_init, F);

    std::cout << "#fl = " << check_flip(uv_init, F);
    Vd dblarea;

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

    // prepare
    Xd V_uv = cur_uv;
    V_uv.conservativeResize(cur_uv.rows(), 3);
    V_uv.col(2).setConstant(0);
    // build target
    std::vector<double> trg(F.rows() * 3, 0);
    Xi TT;
    igl::triangle_triangle_adjacency(F, TT);

    // TODO: add support for 2-trg-edge triangles
    for (int i = 0; i < F.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (TT(i, j) == -1)
            {
                trg[i] = 1;
                // if (trg[i] == 0)
                //     trg[i] = scale(F(i, j), F(i, (j + 1) % 3));
                // else
                // {
                //     trg[i + F.rows()] = trg[i];
                //     trg[i + 2 * F.rows()] = scale(F(i, j), F(i, (j + 1) % 3));
                //     trg[i] = -1;
                //     std::cout << "triangle " << i << " : " << trg[i + F.rows()] << " " << trg[i + F.rows() * 2] << std::endl;
                // }
            }
        }
    }

    spXd Dx, Dy, G;
    igl::doublearea(cur_uv, F, dblarea);
    prepare(V_uv, F, trg, dblarea, Dx, Dy, true);
    G = combine_Dx_Dy(Dx, Dy);
    // update area for uniform
    dblarea.setConstant(sqrt(3) / 2);
    dblarea = dblarea / 2;
    mesh_area = dblarea.sum();

    auto compute_energy = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return compute_energy_from_jacobian(Ji, dblarea);
    };

    auto compute_grad = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        spXd hessian;
        Vd gradE;
        get_grad_and_hessian(G, dblarea, aaa, gradE, hessian);
        return gradE;
    };

    auto compute_energy_max_no_plastic = [&G, &dblarea, &mesh_area, &trg](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        auto E = symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
        double max_e = -1;
        for (int i = 0; i < E.size(); i++)
        {
            // std::cout << "E(triangle " << i << "): " << E(i) << std::endl;
            if (trg[i] != 0 && E(i) > max_e)
            {
                max_e = E(i);
            }
        }
        return max_e;
    };
    auto compute_energy_max = [&G, &dblarea, &mesh_area, &trg](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        auto E = symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
        double max_e = -1;
        for (int i = 0; i < E.size(); i++)
        {
            // std::cout << "E(triangle " << i << "): " << E(i) << std::endl;
            if (E(i) > max_e)
            {
                max_e = E(i);
            }
        }
        return max_e;
    };

    auto compute_energy_no_plastic = [&G, &dblarea, &mesh_area, &trg](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        auto E = symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
        double e_avg = 0.0;
        double area = 0;
        for (int i = 0; i < E.size(); i++)
        {
            if (trg[i] != 0)
            {
                e_avg += E(i) * dblarea(i);
                area += dblarea(i);
            }
        }
        return e_avg / area;
    };

    auto compute_energy_all = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    };

    double energy = compute_energy(cur_uv);
    std::cout << "Start Energy" << energy << std::endl;
    double old_energy = energy;

    double lambda = 1.0;

    std::ofstream writecsv;
    writecsv.open("log.csv");
    writecsv << "step,E_avg,E_max,step_size,|dir|,|gradL|,newton_dec^2,lambda,#flip,trg_diff_max,trg_diff_avg,ratio_min,ratio_max,ratio_avg" << std::endl;
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    
    int total_step = 20000;
    int uniform_step = 20000;
    int gd_step = 100;

    double step_size_last_it = 1;
    std::vector<bool> do_gd(total_step, false);
    // start!!
    for (int ii = start_iter + 1; ii < total_step; ii++)
    {
        spXd hessian;
        Vd gradE;
        std::cout << "\nIt" << ii << std::endl;
        if (ii == uniform_step + 1) // adjust the targets to LS
        {
            double sum1 = 0, sum2 = 0;
            for (int i = 0; i < F.rows(); i++)
            {
                if (trg[i] > 0)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        if (TT(i, j) == -1)
                        {
                            sum1 += trg[i] * trg[i];
                            sum2 += ((cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm()) * trg[i];
                        }
                    }
                }
                else
                {
                    int k = 0;
                    for (int j = 0; j < 3; j++)
                    {
                        if (TT(i, j) == -1)
                        {
                            k++;
                            sum1 += trg[k * F.rows() + i] * trg[k * F.rows() + i];
                            sum2 += ((cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm()) * trg[k * F.rows() + i];
                        }
                    }
                }
            }
            for (int i = 0; i < trg.size(); i++)
            {
                if (trg[i] > 0)
                {
                    trg[i] = trg[i] / sum1 * sum2;
                }
                // for (int j = 0; j < 3; j++)
                // {
                //     if (TT(i, j) == -1)
                //     {
                //         std::cout << "trg: " << trg[i] << "\treal: " << (cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm() << std::endl;
                //     }
                // }
            }
        }
        // prepare for each iteration
        // V_uv = cur_uv;
        // V_uv.conservativeResize(cur_uv.rows(), 3);
        // V_uv.col(2).setConstant(0);
        // Vd dblarea_tmp;
        // igl::doublearea(cur_uv, F, dblarea_tmp);
        // if (ii < uniform_step + 1)
        // {
        //     prepare(V_uv, F, trg, dblarea_tmp, Dx, Dy, true);
        // }
        // else
        // {
        //     prepare(V_uv, F, trg, dblarea_tmp, Dx, Dy, false);
        // }
        // G = combine_Dx_Dy(Dx, Dy);

        // for (int i = 0; i < F.rows(); i++)
        // {
        //     if (trg[i] > 0)
        //         dblarea(i) = trg[i] * trg[i] * sqrt(3) / 2;
        //     else if (trg[i] == -1)
        //     {
        //         dblarea(i) = trg[i + F.rows()] * trg[i + 2 * F.rows()] * sqrt(3) / 2;
        //     }
        //     else if (dblarea_tmp(i) > eps)
        //     {
        //         dblarea(i) = dblarea_tmp(i);
        //     }
        //     else
        //     {
        //         dblarea(i) = eps;
        //     }
        // }
        // if (ii < uniform_step + 1) // set the are to be uniform
        //     dblarea.setConstant(sqrt(3) / 2);
        // dblarea = dblarea / 2;
        // mesh_area = dblarea.sum();
        if (ii >= uniform_step + 1)
        {
            energy = compute_energy(cur_uv);
            old_energy = energy;
        }
        get_grad_and_hessian(G, dblarea, cur_uv, gradE, hessian);
        if (step_size_last_it == 0)
        {
            for (int kk = 0; kk < gd_step; kk++) do_gd[ii + kk] = true;
        }
        if (do_gd[ii])
        {
            std::cout << "do gradient descent\n";
            hessian.setIdentity();
        }
        spXd kkt(hessian.rows() + Aeq.rows(), hessian.cols() + Aeq.rows());
        buildkkt(hessian, Aeq, AeqT, kkt);

        solver.analyzePattern(kkt); // analyze pattern for each iteration

        // resize gradE
        gradE.conservativeResize(kkt.cols());
        for (int i = hessian.cols(); i < kkt.cols(); i++)
        {
            gradE(i) = 0;
        }

        // solve
        solver.factorize(kkt);
        std::cout << "solver.info() = " << solver.info() << std::endl;
        Vd newton = solver.solve(gradE);

        Vd w = -newton.tail(newton.rows() - hessian.cols());
        newton.conservativeResize(hessian.cols());
        gradE.conservativeResize(hessian.cols());

        Xd new_dir = -Eigen::Map<Xd>(newton.data(), cur_uv.rows(), 2); // newton direction
        std::cout << "-gradE.dot(Dx) = " << newton.dot(gradE) << "\n";
        double newton_dec2 = newton.dot(hessian * newton);
        double step_size;
        energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, compute_grad, gradE, energy, step_size);
        step_size_last_it = step_size;

        Vd gradL = gradE + AeqT * w;

        double E_avg, E_max;
        if (ii > uniform_step)
            {E_avg = compute_energy_no_plastic(cur_uv); E_max = compute_energy_max_no_plastic(cur_uv);}
        else
            {E_avg = compute_energy(cur_uv); E_max = compute_energy_max(cur_uv);}

        
        int n_flip = check_flip(cur_uv, F);
        std::cout << std::setprecision(20)
                  << "E=" << E_avg << "\t\tE_max=" << E_max
                  << "\n |new_dir|=" << new_dir.norm() << "\t|gradL|=" << gradL.norm() << std::endl;
        std::cout << "neton_dec^2 = " << newton_dec2 << std::endl;
        std::cout << "#fl = " << n_flip << std::endl;
        std::cout << "lambda = " << lambda << std::endl;

        // compare bd edge length with targets
        double trg_diff_max = -1;
        double trg_diff_sum = 0;
        double ratio_max = -1;
        double ratio_min = 1e10;
        double ratio_sum = 0;
        int count = 0;
        for (int i = 0; i < F.rows(); i++)
        {
            if (trg[i] > 0)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (TT(i, j) == -1)
                    {
                        double ratio = (cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm() / trg[i];
                        double trg_diff = fabs((cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm() - trg[i]);
                        // std::cout << ratio << " " << trg_diff << std::endl;
                        if (trg_diff > trg_diff_max)
                            trg_diff_max = trg_diff;
                        if (ratio > ratio_max)
                            ratio_max = ratio;
                        if (ratio < ratio_min)
                            ratio_min = ratio;
                        ratio_sum += ratio;
                        trg_diff_sum += trg_diff;
                        count++;
                    }
                }
            }
            else if (trg[i] == -1)
            {
                int k = 0;
                for (int j = 0; j < 3; j++)
                {
                    if (TT(i, j) == -1)
                    {
                        k++;
                        double ratio = (cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm() / trg[i + k * F.rows()];
                        double trg_diff = fabs((cur_uv.row(F(i, j)) - cur_uv.row(F(i, (j + 1) % 3))).norm() - trg[i + k * F.rows()]);
                        if (trg_diff > trg_diff_max)
                            trg_diff_max = trg_diff;
                        if (ratio > ratio_max)
                            ratio_max = ratio;
                        if (ratio < ratio_min)
                            ratio_min = ratio;
                        ratio_sum += ratio;
                        trg_diff_sum += trg_diff;
                        count++;
                    }
                }
            }
        }
        std::cout << "trg_diff_max = " << trg_diff_max << std::endl;
        std::cout << "trg_diff_avg = " << trg_diff_sum / count << std::endl;
        std::cout << "ratio_min = " << ratio_min << std::endl;
        std::cout << "ratio_max = " << ratio_max << std::endl;
        std::cout << "ratio_avg = " << ratio_sum / count << std::endl;
        writecsv << std::setprecision(20) << ii << "," << std::setprecision(20) << E_avg << "," << E_max << "," << step_size << "," << new_dir.norm() << "," << gradL.norm() << "," << newton_dec2 << "," << lambda << "," << n_flip << "," << trg_diff_max << "," << trg_diff_sum / count << ","
                 << ratio_min << "," << ratio_max << "," << ratio_sum / count << std::endl;
        if (std::abs(energy - 4) < 1e-10)
            // norm of the gradE
            // if (std::abs(energy - old_energy) < 1e-9)
            break;

        old_energy = energy;
        // save the cur_uv for each iteration
        std::string outfilename = "./serialized/cur_uv_step" + std::to_string(ii);
        igl::serialize(cur_uv, "cur_uv", outfilename, true);
    }

    // std::cout << "write mesh\n";
    // igl::writeOBJ("out.obj", V, F, CN, FN, cur_uv, F);
    // std::cout << compute_energy_all(cur_uv) << std::endl;
}
