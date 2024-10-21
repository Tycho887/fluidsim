#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;

// Structure to store mesh data in a struct-of-arrays format
struct Mesh {
    // Array of vertex positions
    vector<float> x; // x-coordinates of vertices
    vector<float> y; // y-coordinates of vertices

    // Array of triangles (each triangle stores indices of three vertices)
    vector<int> triangle1; // Index of the first vertex of each triangle
    vector<int> triangle2; // Index of the second vertex of each triangle
    vector<int> triangle3; // Index of the third vertex of each triangle

    // Additional arrays for computation (e.g., velocity, pressure)
    vector<float> velocity_x;
    vector<float> velocity_y;
    vector<float> pressure;
};

// Function to read mesh from file
Mesh readMeshFromFile(const string& filename) {
    Mesh mesh;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Could not open the file: " << filename << endl;
        return mesh;
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);

        // Example for parsing vertices
        if (line.find("Vertices") != string::npos) {
            int numVertices;
            iss >> numVertices;
            for (int i = 0; i < numVertices; ++i) {
                float x, y;
                file >> x >> y;
                mesh.x.push_back(x);
                mesh.y.push_back(y);
            }
        }

        // Example for parsing triangles
        if (line.find("Triangles") != string::npos) {
            int numTriangles;
            iss >> numTriangles;
            for (int i = 0; i < numTriangles; ++i) {
                int v1, v2, v3;
                file >> v1 >> v2 >> v3;
                mesh.triangle1.push_back(v1);
                mesh.triangle2.push_back(v2);
                mesh.triangle3.push_back(v3);
            }
        }
    }

    return mesh;
}

void exportToVTK(const Mesh& mesh, const string& filename) {
    ofstream file(filename);

    // Header for the VTK file
    file << "# vtk DataFile Version 2.0\n";
    file << "2D fluid mesh\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // Write vertices
    file << "POINTS " << mesh.x.size() << " float\n";
    for (size_t i = 0; i < mesh.x.size(); ++i) {
        file << mesh.x[i] << " " << mesh.y[i] << " 0.0\n";  // Z is 0 for 2D
    }

    // Write triangles
    file << "CELLS " << mesh.triangle1.size() << " " << mesh.triangle1.size() * 4 << "\n";
    for (size_t i = 0; i < mesh.triangle1.size(); ++i) {
        file << "3 " << mesh.triangle1[i] << " " << mesh.triangle2[i] << " " << mesh.triangle3[i] << "\n";
    }

    // Write cell types (5 is for triangles)
    file << "CELL_TYPES " << mesh.triangle1.size() << "\n";
    for (size_t i = 0; i < mesh.triangle1.size(); ++i) {
        file << "5\n";
    }

    file.close();
}

int main() {
    Mesh mesh = readMeshFromFile("mesh.txt");
    exportToVTK(mesh, "mesh.vtk");

    return 0;
}


