#include <SFML/Graphics.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

// Structure to hold points
struct Point {
    float x, y;
};

// Function to parse the mesh file and extract points and elements
bool readMeshData(const std::string& filename, std::vector<Point>& points, std::vector<std::vector<int>>& elements) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return false;
    }

    std::string line;
    bool readingVertices = false;
    bool readingElements = false;

    while (std::getline(infile, line)) {
        if (line.find("# Vertices") != std::string::npos) {
            readingVertices = true;
            readingElements = false;
            continue;
        }
        if (line.find("# Elements") != std::string::npos) {
            readingVertices = false;
            readingElements = true;
            continue;
        }
        if (readingVertices) {
            // Parse vertex lines
            std::istringstream ss(line);
            int index;
            char colon;
            Point p;
            ss >> index >> colon >> p.x >> colon >> p.y;
            points.push_back(p);
        }
        if (readingElements) {
            // Parse element lines
            std::istringstream ss(line);
            int index, v0, v1, v2;
            char colon;
            ss >> index >> colon >> v0 >> colon >> v1 >> colon >> v2;
            elements.push_back({v0, v1, v2});
        }
    }

    infile.close();
    return true;
}

int main() {
    // Read the mesh data from the file
    std::vector<Point> points;
    std::vector<std::vector<int>> elements;

    if (!readMeshData("mesh_output.txt", points, elements)) {
        return -1; // Exit if reading file failed
    }

    // Setup SFML window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Mesh Visualization");
    window.setFramerateLimit(60);

    // Create a vector to hold all the triangles
    std::vector<sf::ConvexShape> triangles;

    for (const auto& tri : elements) {
        // Create a triangle using SFML ConvexShape
        sf::ConvexShape polygon;
        polygon.setPointCount(3);
        polygon.setPoint(0, sf::Vector2f(points[tri[0]].x * 50, points[tri[0]].y * 50)); // Scale to fit window
        polygon.setPoint(1, sf::Vector2f(points[tri[1]].x * 50, points[tri[1]].y * 50));
        polygon.setPoint(2, sf::Vector2f(points[tri[2]].x * 50, points[tri[2]].y * 50));

        polygon.setFillColor(sf::Color::Green);
        polygon.setOutlineColor(sf::Color::Black);
        polygon.setOutlineThickness(1);

        triangles.push_back(polygon);
    }

    // Main loop to keep the window open and render the mesh
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // Clear the window
        window.clear(sf::Color::White);

        // Draw each triangle
        for (const auto& triangle : triangles) {
            window.draw(triangle);
        }

        // Display the rendered frame
        window.display();
    }

    return 0;
}
