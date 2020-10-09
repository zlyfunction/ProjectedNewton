#include <iostream>
#include <fstream>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/colormap.h>

int main(int argc, char *argv[])
{
    std::string model = argv[1];
    Eigen::VectorXd dblarea_3d, dblarea_uv;
    Eigen::MatrixXi F, Fuv, FN;
    Eigen::MatrixXd V, CN;
    Eigen::MatrixXd uv, cur_uv;
    igl::readOBJ(model, V, uv, CN, F, Fuv, FN);

    // check_flip(uv, F);
    Eigen::VectorXd Energy(F.rows());
    Eigen::MatrixXd color(F.rows(), 3);
    igl::doublearea(V, F, dblarea_3d);
    // igl::doublearea(uv, F, dblarea_uv);
    int start_iter = 0;
    if (argc > 2)
    {
        std::string s = argv[2];
        start_iter = std::atoi(argv[2]);
        std::string infilename = "./serialized/cur_uv_step" + std::to_string(start_iter);
        igl::deserialize(cur_uv, "cur_uv", infilename);
    }
    else
    {
      cur_uv = uv;
    }
    igl::doublearea(cur_uv, F, dblarea_uv);

    // compute area distortion
    // double r = dblarea_3d.sum() / dblarea_uv.sum();
    // std::cout << std::setprecision(17) << "area_3d : area_uv = " << r << std::endl;
    // for (int i = 0; i < F.rows(); i++)
    // {
    //     Energy(i) = dblarea_uv(i) / dblarea_3d(i) * r;
    //     if (Energy(i) < 0 || Energy(i) > 3)
    //     {
    //         std::cout << std::setprecision(16) << dblarea_uv(i) << "\t" << dblarea_3d(i) << "\t" << Energy(i) << std::endl;
    //     }
    // }
    
    for (int i = 0; i < F.rows(); i++)
    {
        double max_e = (cur_uv(F(i,0), 0) - cur_uv(F(i, 1), 0)) * (cur_uv(F(i,0), 0) - cur_uv(F(i, 1), 0)) + (cur_uv(F(i,0), 1) - cur_uv(F(i, 1), 1)) * (cur_uv(F(i,0), 1) - cur_uv(F(i, 1), 1));
        for (int j = 1; j < 3; j++)
        {
            double tmp = (cur_uv(F(i,j), 0) - cur_uv(F(i, (j+1)%3), 0)) * (cur_uv(F(i,j), 0) - cur_uv(F(i, (j+1)%3), 0)) + (cur_uv(F(i,j), 1) - cur_uv(F(i, (j+1)%3), 1)) * (cur_uv(F(i,j), 1) - cur_uv(F(i, (j+1)%3), 1));
            if (tmp > max_e)
            {
                max_e = tmp;
            }
        }
        std::cout << i << ' ' << std::setprecision(17) << max_e / dblarea_uv(i) << std::endl;
    }
    
    // igl::colormap(igl::COLOR_MAP_TYPE_JET, Energy, true, color);
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.data().set_uv(cur_uv * 40);
    // viewer.data().set_colors(color);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    auto key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
    {
        if (key == '2')
        {
            // viewer.data().clear();
            viewer.data().set_mesh(cur_uv, F);
            viewer.core().align_camera_center(cur_uv, F);
        }
        else if (key == '1')
        {
            // viewer.data().clear();
            viewer.data().set_mesh(V, F);
            viewer.core().align_camera_center(V, F);
        }
        else if (key == ' ')
        {
          start_iter++;
          std::cout << start_iter << std::endl;
          std::string infilename = "./serialized/cur_uv_step" + std::to_string(start_iter);
          igl::deserialize(cur_uv, "cur_uv", infilename);
          viewer.data().set_mesh(cur_uv, F);
          viewer.core().align_camera_center(cur_uv, F);
        }
        viewer.data().compute_normals();
        return false;
    };
    viewer.callback_key_down = key_down;
    viewer.launch();
    
    return 0;
}