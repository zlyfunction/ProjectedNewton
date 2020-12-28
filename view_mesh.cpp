#include <iostream>
#include <fstream>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/colormap.h>
#include <igl/triangle_triangle_adjacency.h>
#include "plot.h"
// this functio is only for knot1 model
void build_for_vis(const Eigen::MatrixXi &cut, std::vector<std::vector<std::pair<int, int>>> &all_pairs, std::vector<Eigen::MatrixXd> &all_colors)
{
    std::vector<std::pair<int, int>> edges;
    std::pair<int, int> one_edge;
    Eigen::MatrixXd colors(edges.size(), 3);
    for (int i = 0; i < cut.rows(); i++)
    {

        if (cut(i, 4) == 0)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
        else if (cut(i, 5) == 0)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
    }
    colors.resize(edges.size(), 3);
    for (int i = 0; i < colors.rows(); i++)
    {
        colors.row(i) = Eigen::RowVector3d(1, 0, 0);
    }
    all_pairs[0] = edges;
    all_colors[0] = colors;

    edges.clear();
    for (int i = 0; i < cut.rows(); i++)
    {

        if (cut(i, 4) == 1)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
        else if (cut(i, 5) == 1)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
    }
    colors.resize(edges.size(), 3);
    for (int i = 0; i < colors.rows(); i++)
    {
        colors.row(i) = Eigen::RowVector3d(0, 1, 0);
    }
    all_pairs[1] = edges;
    all_colors[1] = colors;

    edges.clear();
    for (int i = 0; i < cut.rows(); i++)
    {

        if (cut(i, 4) == 2)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
        else if (cut(i, 5) == 2)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
    }
    colors.resize(edges.size(), 3);
    for (int i = 0; i < colors.rows(); i++)
    {
        colors.row(i) = Eigen::RowVector3d(0, 0, 1);
    }
    all_pairs[2] = edges;
    all_colors[2] = colors;

    edges.clear();
    for (int i = 0; i < cut.rows(); i++)
    {

        if (cut(i, 4) == 4)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
        else if (cut(i, 5) == 4)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
    }
    colors.resize(edges.size(), 3);
    for (int i = 0; i < colors.rows(); i++)
    {
        colors.row(i) = Eigen::RowVector3d(0, 1, 1);
    }
    all_pairs[3] = edges;
    all_colors[3] = colors;

    Eigen::MatrixXd color_table(8, 3);
    color_table << 25, 25, 112,
        0, 100, 0,
        255, 69, 0,
        255, 215, 0,
        34, 139, 34,
        255, 182, 193,
        30, 144, 255,
        178, 48, 96;

    edges.clear();
    for (int i = 0; i < cut.rows(); i++)
    {

        if (cut(i, 4) == -1)
        {
            one_edge = std::make_pair(cut(i, 0), cut(i, 1));
            edges.push_back(one_edge);
            one_edge = std::make_pair(cut(i, 2), cut(i, 3));
            edges.push_back(one_edge);
        }
    }
    colors.resize(edges.size(), 3);
    for (int i = 0; i < colors.rows() / 2; i++)
    {
        colors.row(2 * i) = color_table.row(i % 8) / 255.0;
        colors.row(2 * i + 1) = color_table.row(i % 8) / 255.0;
    }
    all_pairs[4] = edges;
    all_colors[4] = colors;
}
int main(int argc, char *argv[])
{
    std::string model = argv[1];
    Eigen::VectorXd dblarea_3d, dblarea_uv;
    Eigen::MatrixXi F, Fuv, FN;
    Eigen::MatrixXd V, CN;
    Eigen::MatrixXd uv, cur_uv;
    // igl::readOBJ(model, V, uv, CN, F, Fuv, FN);

    // for visualization
    Eigen::MatrixXi cut;
    Eigen::VectorXd S;
    igl::deserialize(F, "F", model);
    igl::deserialize(uv, "uv", model);
    igl::deserialize(V, "V", model);

    // Eigen::VectorXd Energy(F.rows());
    Eigen::MatrixXd color(F.rows(), 3);
    // igl::doublearea(V, F, dblarea_3d);
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

    Eigen::MatrixXi TT;
    igl::triangle_triangle_adjacency(F, TT);
    for (int i = 0; i < F.rows(); i++)
    {
        color.row(i) << 0,1,1;
        for (int j = 0; j < 3; j++)
        {
            if (TT(i, j) == -1)
                color.row(i) << 1, 0, 0;
        }
    }
 

    std::vector<std::vector<std::pair<int, int>>> all_pairs(5);
    std::vector<Eigen::MatrixXd> all_colors(5);
    // test visualization
    if (argc > 2)
    {
        build_for_vis(cut, all_pairs, all_colors);
    }
    else
    {
        all_pairs.clear();
    }

    igl::opengl::glfw::Viewer viewer;
    // viewer.data().set_mesh(V, F);
    // viewer.data().set_uv(cur_uv * 40);
    // viewer.data().set_colors(color);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    auto key_down = [&](igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier) {
        if (key == '2')
        {
            viewer.data().clear();

            viewer.data().set_mesh(cur_uv, F);
            if (argc > 2)
            {
                for (int i = 0; i < all_pairs.size(); i++)
                {
                    // std::cout << i << std::endl;
                    // std::cout << all_pairs[0].size() << std::endl;
                    plot_edges(viewer, cur_uv, F, all_colors[i], all_pairs[i]);
                }
                plot_singularity(viewer, cur_uv, F, S, 0.2);
            }
            viewer.data().set_colors(color);
            viewer.core().align_camera_center(cur_uv, F);
            viewer.selected_data_index = 0;
        }
        else if (key == '1')
        {
            viewer.data_list.resize(1);
            viewer.selected_data_index = 0;
            viewer.data().clear();
            // viewer.data().set_mesh(V, F);
            // viewer.data().set_mesh(V, F.row(63));
            // viewer.core().align_camera_center(V, F);
            // viewer.core().align_camera_center(V, F.row(63));
        }
        else if (key == ' ')
        {
            viewer.data_list.resize(1);
            viewer.selected_data_index = 0;
            viewer.data().clear();
            start_iter++;
            std::cout << start_iter << std::endl;
            std::string infilename = "./serialized/cur_uv_step" + std::to_string(start_iter);
            igl::deserialize(cur_uv, "cur_uv", infilename);
            viewer.data().set_mesh(cur_uv, F);
            if (argc > 2)
            {
                for (int i = 0; i < all_pairs.size(); i++)
                {

                    plot_edges(viewer, cur_uv, F, all_colors[i], all_pairs[i]);
                }
                plot_singularity(viewer, cur_uv, F, S, 0.2);
            }
            viewer.data().set_colors(color);
            viewer.selected_data_index = 0;
            // viewer.core().align_camera_center(cur_uv, F);
        }
        viewer.data().compute_normals();
        return false;
    };
    viewer.callback_key_down = key_down;
    viewer.launch();

    return 0;
}