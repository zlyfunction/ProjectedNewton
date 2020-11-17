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
    // igl::grad(V, F, G);
    igl::grad(V, F, G, true); // use uniform mesh instead of V
    // modify F1, F2 to get the right Dx, Dy
    std::cout << "F1 before" << F1 << std::endl;
    F1.col(0).setConstant(1);
    F1.col(1).setConstant(0);
    F1.col(2).setConstant(0);
    std::cout << "F1 after" << F1 << std::endl;

    std::cout << "F2 before" << F2 << std::endl;
    F2.col(0).setConstant(0);
    F2.col(1).setConstant(1);
    F2.col(2).setConstant(0);
    std::cout << "F2 after" << F2 << std::endl;
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
        std::cout << "fix " << l[0] << " " << l[1] << std::endl;
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
    Xd CN;
    Xi FN, FTC;
    Xi cut;

    if (argc > 1)
    {
        std::cout << "read model\n";
        std::string model = argv[1];
        igl::readOBJ(model, V, uv_init, CN, F, FN, FN);
        std::cout << V << std::endl;
        std::cout << uv_init << std::endl;
        std::cout << F << std::endl;
        // igl::deserialize(F, "F", model);
        // igl::deserialize(uv_init, "uv", model);
        // igl::deserialize(V, "V", model);

    }
    else
    {
        std::cout << "test single triangle\n";
        F.resize(1,3);
        F << 0, 1, 2;
        uv_init.resize(3, 2);
        uv_init << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
        // uv_init << 0.0, 0.0, 1.0, 0.0, 0.5, sqrt(3) / 2.0;

        V.resize(3, 3);
        V << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, sqrt(3) / 2.0, 0.0;
    }
    
    igl::writeOBJ("input_init.obj", V, F, CN, FN, uv_init, F);
    
    

    Vd dblarea_uv;
    igl::doublearea(uv_init, F, dblarea_uv);

    Vd dblarea;
    igl::doublearea(V, F, dblarea);
    dblarea.setConstant(1); // set area to be a constant
    mesh_area = dblarea.sum();
    
    
    spXd Dx, Dy, G;
    prepare(V, F, Dx, Dy);
    G = combine_Dx_Dy(Dx, Dy);
    std::cout << G.rows() << " " << G.cols() << std::endl;
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
        // std::cout << Ji << std::endl;
        return compute_energy_from_jacobian(Ji, dblarea);
    };

    auto compute_grad = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        spXd hessian;
        Vd gradE;
        get_grad_and_hessian(G, dblarea, aaa, gradE, hessian);
        return gradE;
    };

    auto compute_energy_max = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        // std::cout << "Jacobian:\n" << Ji << std::endl;
        auto E = symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
        double max_e = E(0);
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

    auto compute_energy_all = [&G, &dblarea, &mesh_area](Eigen::MatrixXd &aaa) {
        Xd Ji;
        jacobian_from_uv(G, aaa, Ji);
        return symmetric_dirichlet_energy(Ji.col(0), Ji.col(1), Ji.col(2), Ji.col(3));
    };

    double energy = compute_energy(cur_uv);
    std::cout << "Start Energy" << energy << std::endl;
    double old_energy = energy;

    // double lambda = 0.999;
    double lambda = 1.0;

    std::ofstream writecsv;
    writecsv.open("log.csv");
    writecsv << "step,E_avg,E_max,step_size,|dir|,|gradL|,newton_dec^2,lambda,#flip" << std::endl;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    // Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    for (int ii = start_iter + 1; ii < 100000; ii++)
    {
        spXd hessian;
        Vd gradE;
        std::cout << "\nIt" << ii << std::endl;
        get_grad_and_hessian(G, dblarea, cur_uv, gradE, hessian);

        // fix two dofs
        for (int k = 0; k < hessian.outerSize(); k++)
        {
            for (Eigen::SparseMatrix<double>::InnerIterator it(hessian, k); it; ++it)
            {
                if (it.row() == 0 || it.col()  == 0)
                {
                    if (it.col() == it.row())
                    {
                        it.valueRef() = 1.0;
                    }
                    else
                    {
                        it.valueRef() = 0;
                    }
                }
                if (it.row() == V.rows() || it.col() == V.rows())
                {
                    if (it.col() == it.row())
                    {
                        it.valueRef() = 1.0;
                    }   
                    else
                    {
                        it.valueRef() = 0;
                    }
                }
                // if (it.row() == 1 || it.col()  == 1)
                // {
                //     if (it.col() == it.row())
                //     {
                //         it.valueRef() = 1.0;
                //     }
                //     else
                //     {
                //         it.valueRef() = 0;
                //     }
                // }
                
                // if (it.row() == V.rows() + 1 || it.col() == V.rows() + 1)
                // {
                //     if (it.col() == it.row())
                //     {
                //         it.valueRef() = 1.0;
                //     }   
                //     else
                //     {
                //         it.valueRef() = 0;
                //     }
                // }
            
            }
        }
        
        gradE(0) = 0;
        gradE(V.rows()) = 0; 
        // gradE(1) = 0; 
        // gradE(V.rows() + 1) = 0;

        // add disturb to avoid singular
        // spXd Id(hessian.rows(), hessian.cols());
        // Id.setIdentity();
        // hessian = hessian + 1e-10 * Id;

        std::string hessian_filename = "hessian/hessian_" + std::to_string(ii) + ".txt";
        write_hessian_to_file(hessian, hessian_filename);
        
        std::string grad_filename = "hessian/gradE_" + std::to_string(ii) + ".txt";
        std::ofstream write_grad;
        write_grad.open(grad_filename);
        write_grad << std::setprecision(20) << gradE;

        std::string uv_filename = "hessian/uv_" + std::to_string(ii) + ".txt";
        std::ofstream write_uv;
        write_uv.open(uv_filename);
        write_uv << std::setprecision(20) << Eigen::Map<Vd>(cur_uv.data(), cur_uv.size());
        
        if (ii == start_iter + 1)
        {
            // solver.analyzePattern(hessian);
            solver.analyzePattern(hessian);
        }

        solver.factorize(hessian);
        Vd newton = solver.solve(gradE);

        Xd new_dir = -Eigen::Map<Xd>(newton.data(), V.rows(), 2); // newton
        std::cout << "-gradE.dot(Dx) = " << newton.dot(gradE) << "\n";
        double newton_dec2 = newton.dot(hessian * newton);
        // energy = wolfe_linesearch(F, cur_uv, new_dir, compute_energy, gradE, energy, use_gd);
        double step_size;
        energy = bi_linesearch(F, cur_uv, new_dir, compute_energy, compute_grad, gradE, energy, step_size);
        
        get_grad_and_hessian(G, dblarea, cur_uv, gradE, hessian);
        Vd gradL = gradE;
        double E_avg = compute_energy(cur_uv), E_max = compute_energy_max(cur_uv);
        int n_flip = check_flip(cur_uv, F);
        std::cout << std::setprecision(20)
                  << "E=" << E_avg << "\t\tE_max=" << E_max
                  << "\n |new_dir|=" << new_dir.norm() << "\t|gradL|=" << gradL.norm() << std::endl;
        std::cout << "neton_dec^2 = " << newton_dec2 << std::endl;
        std::cout << "#fl = " << n_flip << std::endl;
        std::cout << "lambda = " << lambda << std::endl;
        // std::cout << "uv = \n";
        // for (int kk = 0; kk < cur_uv.rows(); kk++)
        // {
        //     std::cout << cur_uv.row(kk) << std::endl;
        // }

        writecsv << std::setprecision(20) << ii << "," << std::setprecision(20) << E_avg << "," << E_max << "," << step_size << "," << new_dir.norm() << "," << gradL.norm() << "," << newton_dec2 << "," << lambda << "," << n_flip << std::endl;
        // if (std::abs(energy - 4) < 1e-10)
        //     // norm of the gradE
        // if (std::abs(energy - old_energy) < 1e-9)
        if (step_size == 0)
            break;

        old_energy = energy;

        // save the cur_uv
        std::string outfilename = "./serialized/cur_uv_step" + std::to_string(ii);
        igl::serialize(cur_uv, "cur_uv", outfilename, true);
    }
}
