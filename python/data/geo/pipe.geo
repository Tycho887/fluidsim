// Pipe dimensions
L = 1;        // Length of the pipe
H = 0.1;         // Height (diameter) of the pipe
d = 0.05;       // Diameter of the obstruction

// Mesh resolution
res = 0.01;     // Element size

// Pipe Geometry
Point(1) = {0, 0, 0, res};
Point(2) = {L, 0, 0, res};
Point(3) = {L, H, 0, res};
Point(4) = {0, H, 0, res};

// Lines for the pipe
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Obstruction in the middle (circular)
xc = L/2;       // X-coordinate of the center of the obstruction
yc = H/2;       // Y-coordinate of the center of the obstruction
Point(5) = {xc, yc, 0, res};
Circle(5) = {5, d};

// Create surface for the pipe
Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Subtract obstruction from the pipe
BooleanDifference{ Surface{1}; Delete; }{ Surface{5}; }

Mesh 2;
