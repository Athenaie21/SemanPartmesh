#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <array>
#include <Eigen/Core>

#include <igl/readOBJ.h>
#include <igl/copyleft/comiso/miq.h>

#include <qex.h>

static bool load_cross_field(const std::string &path,
                             Eigen::MatrixXd &PD1,
                             Eigen::MatrixXd &PD2)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open cross field file: " << path << std::endl;
        return false;
    }

    std::vector<std::array<double,6>> rows;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::array<double,6> r;
        for (int i = 0; i < 6; ++i) {
            if (!(ss >> r[i])) {
                std::cerr << "Parse error at row " << rows.size() << std::endl;
                return false;
            }
        }
        rows.push_back(r);
    }

    int nf = static_cast<int>(rows.size());
    PD1.resize(nf, 3);
    PD2.resize(nf, 3);
    for (int i = 0; i < nf; ++i) {
        PD1(i, 0) = rows[i][0];
        PD1(i, 1) = rows[i][1];
        PD1(i, 2) = rows[i][2];
        PD2(i, 0) = rows[i][3];
        PD2(i, 1) = rows[i][4];
        PD2(i, 2) = rows[i][5];
    }
    return true;
}

static bool write_quad_obj(const std::string &path,
                           const qex_QuadMesh &qm)
{
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Cannot write to: " << path << std::endl;
        return false;
    }

    for (unsigned int i = 0; i < qm.vertex_count; ++i) {
        ofs << "v " << qm.vertices[i].x[0] << " "
                    << qm.vertices[i].x[1] << " "
                    << qm.vertices[i].x[2] << "\n";
    }

    for (unsigned int i = 0; i < qm.quad_count; ++i) {
        ofs << "f " << (qm.quads[i].indices[0] + 1) << " "
                    << (qm.quads[i].indices[1] + 1) << " "
                    << (qm.quads[i].indices[2] + 1) << " "
                    << (qm.quads[i].indices[3] + 1) << "\n";
    }
    return true;
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.obj> <crossfield.txt> <output_quad.obj>"
                  << " [gradient_size] [stiffness] [direct_round] [iters] [local_iters]"
                  << std::endl;
        return 1;
    }

    std::string mesh_path     = argv[1];
    std::string cf_path       = argv[2];
    std::string output_path   = argv[3];
    int argi                 = 4;
    double gradient_size      = 30.0;
    double stiffness          = 5.0;
    bool   direct_round       = false;
    int    iters              = 5;
    int    local_iters        = 5;

    auto is_flag = [](const char *arg) {
        return std::strlen(arg) >= 2 && arg[0] == '-' && arg[1] == '-';
    };

    if (argc > argi && !is_flag(argv[argi])) gradient_size = std::atof(argv[argi++]);
    if (argc > argi && !is_flag(argv[argi])) stiffness = std::atof(argv[argi++]);
    if (argc > argi && !is_flag(argv[argi])) direct_round = (std::atoi(argv[argi++]) != 0);
    if (argc > argi && !is_flag(argv[argi])) iters = std::atoi(argv[argi++]);
    if (argc > argi && !is_flag(argv[argi])) local_iters = std::atoi(argv[argi++]);

    // 1. Load triangle mesh
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::cout << "[1/4] Loading mesh: " << mesh_path << std::endl;
    if (!igl::readOBJ(mesh_path, V, F)) {
        std::cerr << "Failed to read mesh: " << mesh_path << std::endl;
        return 1;
    }
    std::cout << "       V=" << V.rows() << " F=" << F.rows() << std::endl;

    // 2. Load cross field
    Eigen::MatrixXd PD1, PD2;
    std::cout << "[2/4] Loading cross field: " << cf_path << std::endl;
    if (!load_cross_field(cf_path, PD1, PD2)) {
        return 1;
    }
    std::cout << "       cross field rows=" << PD1.rows() << std::endl;

    if (PD1.rows() != F.rows()) {
        std::cerr << "Cross field rows (" << PD1.rows()
                  << ") != face count (" << F.rows() << ")" << std::endl;
        return 1;
    }

    // 3. MIQ parameterization (cross field -> UV)
    Eigen::MatrixXd UV;
    Eigen::MatrixXi FUV;
    std::cout << "[3/4] Running MIQ parameterization (gradient_size="
              << gradient_size << ", stiffness=" << stiffness
              << ", iters=" << iters << ")..." << std::endl;

    igl::copyleft::comiso::miq(
        V, F, PD1, PD2,
        UV, FUV,
        gradient_size,
        stiffness,
        direct_round,
        static_cast<unsigned int>(iters),
        static_cast<unsigned int>(local_iters),
        true,   // doRound
        true    // singularityRound
    );

    std::cout << "       UV=" << UV.rows() << " FUV=" << FUV.rows() << std::endl;

    // 4. libQEx: extract quad mesh from UV parameterization
    std::cout << "[4/4] Extracting quad mesh with libQEx..." << std::endl;

    int nv = static_cast<int>(V.rows());
    int nf = static_cast<int>(F.rows());

    qex_TriMesh triMesh;
    triMesh.vertex_count = static_cast<unsigned int>(nv);
    triMesh.tri_count    = static_cast<unsigned int>(nf);

    triMesh.vertices = (qex_Point3*)malloc(sizeof(qex_Point3) * nv);
    triMesh.tris     = (qex_Tri*)malloc(sizeof(qex_Tri) * nf);
    triMesh.uvTris   = (qex_UVTri*)malloc(sizeof(qex_UVTri) * nf);

    for (int i = 0; i < nv; ++i) {
        triMesh.vertices[i].x[0] = V(i, 0);
        triMesh.vertices[i].x[1] = V(i, 1);
        triMesh.vertices[i].x[2] = V(i, 2);
    }

    for (int i = 0; i < nf; ++i) {
        triMesh.tris[i].indices[0] = static_cast<qex_Index>(F(i, 0));
        triMesh.tris[i].indices[1] = static_cast<qex_Index>(F(i, 1));
        triMesh.tris[i].indices[2] = static_cast<qex_Index>(F(i, 2));
    }

    for (int i = 0; i < FUV.rows(); ++i) {
        for (int k = 0; k < 3; ++k) {
            triMesh.uvTris[i].uvs[k].x[0] = UV(FUV(i, k), 0);
            triMesh.uvTris[i].uvs[k].x[1] = UV(FUV(i, k), 1);
        }
    }

    qex_QuadMesh quadMesh;
    memset(&quadMesh, 0, sizeof(quadMesh));

    qex_extractQuadMesh(&triMesh, nullptr, &quadMesh);

    std::cout << "       Quad vertices=" << quadMesh.vertex_count
              << " Quad faces=" << quadMesh.quad_count << std::endl;

    // 5. Write output
    if (!write_quad_obj(output_path, quadMesh)) {
        return 1;
    }
    std::cout << "Output saved to: " << output_path << std::endl;

    free(triMesh.vertices);
    free(triMesh.tris);
    free(triMesh.uvTris);
    free(quadMesh.vertices);
    free(quadMesh.quads);

    return 0;
}
