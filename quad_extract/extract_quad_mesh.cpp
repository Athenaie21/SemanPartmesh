#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <cmath>
#include <algorithm>

#include <Eigen/Core>

#include <igl/readOBJ.h>
#include <igl/copyleft/comiso/miq.h>

#include <qex.h>

struct ExtraOptions {
    std::string size_field_path;
    double size_strength = 0.75;
    int size_smooth_iters = 2;
};

static bool starts_with(const std::string &value, const std::string &prefix)
{
    return value.rfind(prefix, 0) == 0;
}

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

static bool load_scalar_field(
    const std::string &path,
    const Eigen::MatrixXi &F,
    Eigen::VectorXd &face_field)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open size field file: " << path << std::endl;
        return false;
    }

    std::vector<double> values;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double v = 0.0;
        if (!(ss >> v)) {
            std::cerr << "Parse error in size field file at row "
                      << values.size() << std::endl;
            return false;
        }
        values.push_back(v);
    }

    if (values.empty()) {
        std::cerr << "Size field file is empty: " << path << std::endl;
        return false;
    }

    if (static_cast<int>(values.size()) == F.rows()) {
        face_field.resize(F.rows());
        for (int i = 0; i < F.rows(); ++i) {
            face_field(i) = values[i];
        }
        std::cout << "       Loaded per-face size field (" << values.size()
                  << " values)" << std::endl;
        return true;
    }

    int max_vid = F.maxCoeff();
    int vertex_count = max_vid + 1;
    if (static_cast<int>(values.size()) == vertex_count) {
        face_field.resize(F.rows());
        for (int i = 0; i < F.rows(); ++i) {
            face_field(i) = (
                values[F(i, 0)] +
                values[F(i, 1)] +
                values[F(i, 2)]
            ) / 3.0;
        }
        std::cout << "       Loaded per-vertex size field (" << values.size()
                  << " values), averaged to faces" << std::endl;
        return true;
    }

    std::cerr << "Size field length (" << values.size()
              << ") matches neither face count (" << F.rows()
              << ") nor vertex count (" << vertex_count << ")" << std::endl;
    return false;
}

static std::vector<std::vector<int>> build_face_adjacency(const Eigen::MatrixXi &F)
{
    std::vector<std::vector<int>> adjacency(F.rows());
    std::map<std::pair<int, int>, int> edge_owner;

    for (int i = 0; i < F.rows(); ++i) {
        for (int k = 0; k < 3; ++k) {
            int a = F(i, k);
            int b = F(i, (k + 1) % 3);
            std::pair<int, int> edge = std::make_pair(std::min(a, b), std::max(a, b));
            auto it = edge_owner.find(edge);
            if (it == edge_owner.end()) {
                edge_owner[edge] = i;
            } else {
                int j = it->second;
                adjacency[i].push_back(j);
                adjacency[j].push_back(i);
            }
        }
    }
    return adjacency;
}

static void smooth_face_field(Eigen::VectorXd &field,
                              const std::vector<std::vector<int>> &adjacency,
                              int iterations)
{
    for (int it = 0; it < iterations; ++it) {
        Eigen::VectorXd smoothed = field;
        for (int i = 0; i < field.rows(); ++i) {
            const auto &neighbors = adjacency[i];
            if (neighbors.empty()) continue;

            double sum = field(i);
            for (int j : neighbors) {
                sum += field(j);
            }
            smoothed(i) = sum / static_cast<double>(neighbors.size() + 1);
        }
        field = smoothed;
    }
}

static Eigen::VectorXd make_uv_face_scales(
    const Eigen::VectorXd &size_field,
    double size_strength)
{
    Eigen::VectorXd safe_size = size_field.array().max(1e-8);
    Eigen::VectorXd density = safe_size.array().inverse();
    double mean_density = density.mean();
    if (mean_density <= 1e-12) {
        mean_density = 1.0;
    }

    Eigen::VectorXd log_density = (density.array() / mean_density).log();
    std::vector<double> abs_log_density;
    abs_log_density.reserve(log_density.rows());
    for (int i = 0; i < log_density.rows(); ++i) {
        abs_log_density.push_back(std::abs(log_density(i)));
    }
    std::sort(abs_log_density.begin(), abs_log_density.end());

    double robust = 1.0;
    if (!abs_log_density.empty()) {
        std::size_t idx = static_cast<std::size_t>(
            0.95 * static_cast<double>(abs_log_density.size() - 1));
        robust = std::max(abs_log_density[idx], 1e-6);
    }

    // Conservative local UV scaling:
    // 1) normalize log-density by a robust spread estimate
    // 2) compress outliers with tanh
    // 3) cap maximum local deviation from the global scale
    double max_deviation = std::clamp(0.18 * size_strength, 0.0, 0.25);

    Eigen::VectorXd scales(log_density.rows());
    for (int i = 0; i < log_density.rows(); ++i) {
        double normalized = log_density(i) / robust;
        double compressed = std::tanh(normalized);
        double scale = 1.0 + max_deviation * compressed;
        scales(i) = std::clamp(scale, 1.0 - max_deviation, 1.0 + max_deviation);
    }
    return scales;
}

static void apply_face_scales_to_uv_tris(
    const Eigen::MatrixXd &UV,
    const Eigen::MatrixXi &FUV,
    const Eigen::VectorXd &uv_face_scales,
    qex_TriMesh &triMesh)
{
    for (int i = 0; i < FUV.rows(); ++i) {
        Eigen::Vector2d uv0 = UV.row(FUV(i, 0));
        Eigen::Vector2d uv1 = UV.row(FUV(i, 1));
        Eigen::Vector2d uv2 = UV.row(FUV(i, 2));
        Eigen::Vector2d centroid = (uv0 + uv1 + uv2) / 3.0;
        double scale = uv_face_scales(i);

        Eigen::Vector2d scaled0 = centroid + scale * (uv0 - centroid);
        Eigen::Vector2d scaled1 = centroid + scale * (uv1 - centroid);
        Eigen::Vector2d scaled2 = centroid + scale * (uv2 - centroid);

        triMesh.uvTris[i].uvs[0].x[0] = scaled0(0);
        triMesh.uvTris[i].uvs[0].x[1] = scaled0(1);
        triMesh.uvTris[i].uvs[1].x[0] = scaled1(0);
        triMesh.uvTris[i].uvs[1].x[1] = scaled1(1);
        triMesh.uvTris[i].uvs[2].x[0] = scaled2(0);
        triMesh.uvTris[i].uvs[2].x[1] = scaled2(1);
    }
}

static ExtraOptions parse_extra_options(int argc, char *argv[], int start_idx)
{
    ExtraOptions opts;
    for (int i = start_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (starts_with(arg, "--size_field=")) {
            opts.size_field_path = arg.substr(std::strlen("--size_field="));
        } else if (starts_with(arg, "--size_strength=")) {
            opts.size_strength = std::atof(arg.substr(std::strlen("--size_strength=")).c_str());
        } else if (starts_with(arg, "--size_smooth_iters=")) {
            opts.size_smooth_iters = std::atoi(arg.substr(std::strlen("--size_smooth_iters=")).c_str());
        } else {
            std::cerr << "WARNING: Unknown extra option ignored: " << arg << std::endl;
        }
    }
    return opts;
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.obj> <crossfield.txt> <output_quad.obj>"
                  << " [gradient_size] [stiffness] [direct_round] [iters] [local_iters]"
                  << " [--size_field=path] [--size_strength=value] [--size_smooth_iters=value]"
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

    ExtraOptions extra = parse_extra_options(argc, argv, argi);

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

    Eigen::VectorXd uv_face_scales = Eigen::VectorXd::Ones(nf);
    if (!extra.size_field_path.empty()) {
        std::cout << "       Applying local size field: " << extra.size_field_path << std::endl;
        Eigen::VectorXd size_field;
        if (!load_scalar_field(extra.size_field_path, F, size_field)) {
            return 1;
        }
        auto adjacency = build_face_adjacency(F);
        smooth_face_field(size_field, adjacency, std::max(0, extra.size_smooth_iters));
        uv_face_scales = make_uv_face_scales(size_field, extra.size_strength);

        std::cout << "       size_strength=" << extra.size_strength
                  << " smooth_iters=" << extra.size_smooth_iters
                  << " uv_scale_range=[" << uv_face_scales.minCoeff()
                  << ", " << uv_face_scales.maxCoeff() << "]" << std::endl;
    }

    for (int i = 0; i < nf; ++i) {
        triMesh.tris[i].indices[0] = static_cast<qex_Index>(F(i, 0));
        triMesh.tris[i].indices[1] = static_cast<qex_Index>(F(i, 1));
        triMesh.tris[i].indices[2] = static_cast<qex_Index>(F(i, 2));
    }

    apply_face_scales_to_uv_tris(UV, FUV, uv_face_scales, triMesh);

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
