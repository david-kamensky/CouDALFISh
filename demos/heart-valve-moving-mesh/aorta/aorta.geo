// 3D Artery from STEP Geometry File
// Created for GMSH 4.4
// Grant Neighbor
// May 4, 2022
// Iowa State University


////////////////////////////////////////////////
// Environment Configuration
////////////////////////////////////////////////

// Viewer Properties
Geometry.Points = 1;
Geometry.Curves = 1;
Geometry.Surfaces = 1;
Geometry.Volumes = 1;
General.Axes = 1;

// Element Sizing
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 1;
Mesh.MeshSizeExtendFromBoundary = 0;
//Mesh.CharacteristicLengthFactor = 1;

// Clipping Defaults
General.Clip0A = 1;
General.Clip0B = 0;
General.Clip0C = 0;
General.Clip0D = .05;
General.ClipWholeElements = 1;
Mesh.Clip = 1;


////////////////////////////////////////////////
// Element Sizes
////////////////////////////////////////////////

// Fluid Mesh Sizes
h_min = .020; //.010; 
h_max = .160; //.160; 
h_ref_valve = .060; //.040; 
h_ref_upstream = .060; //.040;

Mesh.MeshSizeMax = h_max;
Mesh.MeshSizeMin = h_min;

// Structured Mesh Counts
n_solid_layers = 		2; //3;
n_circumferential = 	19; //41;
n_length_inlet = 		10; //15;
n_length_sinus = 		40; //110;
n_length_outlet = 		40; //50;
n_fluid_layers = 		3; //9;
n_fluid_progression = 	.88; //.88; 


n_solid_layers = 		2/Mesh.CharacteristicLengthFactor; //3;
n_circumferential = 	19/Mesh.CharacteristicLengthFactor; //41;
n_length_inlet = 		10/Mesh.CharacteristicLengthFactor; //15;
n_length_sinus = 		40/Mesh.CharacteristicLengthFactor; //110;
n_length_outlet = 		40/Mesh.CharacteristicLengthFactor; //50;
n_fluid_layers = 		3/Mesh.CharacteristicLengthFactor; //9;
n_fluid_progression = 	.88; //.88; 


////////////////////////////////////////////////
// Geometry
////////////////////////////////////////////////

// Import
SetFactory("OpenCASCADE");
Geometry.OCCScaling = 0.1;
Merge "inner-surface.step";
Coherence;

// Outer Surface
Dilate {{0, 0, 0}, {1.14, 1.14, 1}} {
  Duplicata { Surface{1:18}; }
}

// inner surface for boundary layer
Dilate {{0, 0, 0}, {.9, .9, 1}} {
  Duplicata { Surface{1:18}; }
}

Coherence;

// Fluid Inlet
Curve Loop(1) = {92, 126, 122, 104, 110, 115};
Plane Surface(55) = {1};

// Fluid Outlet
Curve Loop(2) = {86, 124, 118, 97, 106, 112};
Plane Surface(56) = {2};

// Lines connecting the inner surface to the BL separator
startPoint = 127;
numLines = 24;
For i In {1:numLines}
  Line(startPoint+i) = {i+48,i};
  Transfinite Curve {startPoint+i} = (n_fluid_layers+1) Using Progression n_fluid_progression;
EndFor

// Lines connecting the solid region
startPoint = 176;
numLines = 24;
For i In {1:numLines}
  Line(startPoint+i) = {i+24,i};
  Transfinite Curve {startPoint+i} = (n_solid_layers+1) Using Progression 1;
EndFor

// Fluid BL Inlet
Curve Loop(4) = {92, 135, -8, -134};
Plane Surface(57) = {4};
Curve Loop(5) = {42, -151, -126, 135};
Plane Surface(58) = {5};
Curve Loop(6) = {151, 38, -143, -122};
Plane Surface(59) = {6};
Curve Loop(7) = {104, 142, -20, -143};
Plane Surface(60) = {7};
Curve Loop(8) = {147, -26, -142, 110};
Plane Surface(61) = {8};
Curve Loop(9) = {115, 134, -31, -147};
Plane Surface(62) = {9};

// Solid Inlet
Curve Loop(10) = {50, 184, -8, -183};
Plane Surface(63) = {10};
Curve Loop(11) = {73, 183, -31, -196};
Plane Surface(64) = {11};
Curve Loop(12) = {68, 196, -26, -191};
Plane Surface(65) = {12};
Curve Loop(13) = {20, -191, -62, 192};
Plane Surface(66) = {13};
Curve Loop(14) = {192, -38, -200, 80};
Plane Surface(67) = {14};
Curve Loop(15) = {42, -200, -84, 184};
Plane Surface(68) = {15};

// Inlet Length
Transfinite Curve {111, 27, 69, 52, 10, 94, 51, 9, 93, 121, 37, 79, 103, 19, 61, 102, 18, 60} = n_length_inlet Using Progression 1;

// Sinus Length
Transfinite Curve {49, 7, 91, 48, 6, 90, 120, 36, 78, 99, 15, 57, 101, 17, 59, 67, 25, 109} = n_length_sinus Using Progression 1;

// Outlet Length
Transfinite Curve {45, 3, 87, 63, 21, 105, 43, 1, 85, 12, 96, 54, 116, 32, 74, 98, 14, 56} = n_length_outlet Using Progression 1;

// Circumfrential Curves
Transfinite Curve {65, 71, 29, 23, 113, 107, 11, 53, 95, 33, 75, 117, 123, 39, 81, 88, 4, 46, 47, 5, 89, 83, 41, 125, 119, 35, 77, 100, 16, 58, 66, 24, 108, 114, 30, 72, 73, 31, 115, 92, 30, 50, 8, 42, 84, 126, 122, 38, 80, 62, 20, 104, 110, 26, 68, 44, 2, 86, 124, 40, 82, 118, 34, 76, 55, 97, 13, 106, 22, 64, 70, 28, 112} = n_circumferential Using Progression 1;

// Fluid outlet BL
Curve Loop(16) = {2, -128, -86, 130};
Plane Surface(69) = {16};
Curve Loop(18) = {40, -149, -124, 128};
Plane Surface(70) = {18};
Curve Loop(20) = {118, 139, -34, -149};
Plane Surface(71) = {20};
Curve Loop(22) = {97, 138, -13, -139};
Plane Surface(72) = {22};
Curve Loop(24) = {22, -144, -106, 138};
Plane Surface(73) = {24};
Curve Loop(26) = {112, 130, -28, -144};
Plane Surface(74) = {26};


// Solid Outlet
Curve Loop(28) = {44, 177, -2, -179};
Plane Surface(75) = {28};
Curve Loop(30) = {82, 198, -40, -177};
Plane Surface(76) = {30};
Curve Loop(32) = {34, -188, -76, 198};
Plane Surface(77) = {32};
Curve Loop(34) = {13, -187, -55, 188};
Plane Surface(78) = {34};
Curve Loop(36) = {64, 193, -22, -187};
Plane Surface(79) = {36};
Curve Loop(38) = {28, -179, -70, 193};
Plane Surface(80) = {38};

// Fluid BL Sinus Root
Curve Loop(40) = {89, 133, -5, -132};
Plane Surface(81) = {40};
Curve Loop(41) = {125, 150, -41, -133};
Plane Surface(82) = {41};
Curve Loop(42) = {119, 140, -35, -150};
Plane Surface(83) = {42};
Curve Loop(43) = {100, 141, -16, -140};
Plane Surface(84) = {43};
Curve Loop(44) = {24, -146, -108, 141};
Plane Surface(85) = {44};
Curve Loop(45) = {114, 132, -30, -146};
Plane Surface(86) = {45};

// Solid Sinus Root
Curve Loop(46) = {47, 182, -5, -181};
Plane Surface(87) = {46};
Curve Loop(47) = {83, 199, -41, -182};
Plane Surface(88) = {47};
Curve Loop(48) = {199, 35, -189, -77};
Plane Surface(89) = {48};
Curve Loop(49) = {16, -190, -58, 189};
Plane Surface(90) = {49};
Curve Loop(50) = {66, 195, -24, -190};
Plane Surface(91) = {50};
Curve Loop(51) = {72, 181, -30, -195};
Plane Surface(92) = {51};

// Fluid BL top of valve area
Curve Loop(52) = {4, -131, -88, 129};
Plane Surface(93) = {52};
Curve Loop(54) = {123, 129, -39, -148};
Plane Surface(94) = {54};
Curve Loop(56) = {117, 148, -33, -137};
Plane Surface(95) = {56};
Curve Loop(58) = {11, -137, -95, 136};
Plane Surface(96) = {58};
Curve Loop(60) = {107, 136, -23, -145};
Plane Surface(97) = {60};
Curve Loop(62) = {145, -29, -131, 113};
Plane Surface(98) = {62};

// Solid top of valve area ring
Curve Loop(63) = {46, 180, -4, -178};
Plane Surface(99) = {63};
Curve Loop(65) = {81, 178, -39, -197};
Plane Surface(100) = {65};
Curve Loop(67) = {33, -197, -75, 186};
Plane Surface(101) = {67};
Curve Loop(69) = {11, -186, -53, 185};
Plane Surface(102) = {69};
Curve Loop(71) = {65, 185, -23, -194};
Plane Surface(103) = {71};
Curve Loop(73) = {71, 194, -29, -180};
Plane Surface(104) = {73};

// Solid Inlet Lengthwise Surfaces
Curve Loop(75) = {52, 183, -10, -181};
Plane Surface(105) = {75};
Curve Loop(76) = {69, 195, -27, -196};
Plane Surface(106) = {76};
Curve Loop(77) = {60, 191, -18, -190};
Plane Surface(107) = {77};
Curve Loop(79) = {192, 19, -189, -61};
Plane Surface(108) = {79};
Curve Loop(80) = {37, -200, -79, 199};
Plane Surface(109) = {80};
Curve Loop(81) = {51, 182, -9, -184};
Plane Surface(110) = {81};

// Fluid BL Inlet Lengthwise Surfaces
Curve Loop(82) = {27, -146, -111, 147};
Plane Surface(111) = {82};
Curve Loop(83) = {134, -10, -132, 94};
Plane Surface(112) = {83};
Curve Loop(84) = {135, 9, -133, -93};
Plane Surface(113) = {84};
Curve Loop(85) = {150, 37, -151, -121};
Plane Surface(114) = {85};
Curve Loop(86) = {103, 140, -19, -143};
Plane Surface(115) = {86};
Curve Loop(88) = {102, 142, -18, -141};
Plane Surface(116) = {88};

// Solid sinus region lengthwise surfaces
Curve Loop(90) = {49, 180, -7, -181};
Plane Surface(117) = {90};
Curve Loop(92) = {67, 194, -25, -195};
Plane Surface(118) = {92};
Curve Loop(93) = {59, 190, -17, -185};
Plane Surface(119) = {93};
Curve Loop(95) = {57, 186, -15, -189};
Plane Surface(120) = {95};
Curve Loop(98) = {36, -197, -78, 199};
Plane Surface(121) = {98};
Curve Loop(100) = {6, -178, -48, 182};
Plane Surface(122) = {100};

// Fluid sinus region BL lengthwise
Curve Loop(101) = {7, -131, -91, 132};
Plane Surface(123) = {101};
Curve Loop(103) = {6, -129, -90, 133};
Plane Surface(124) = {103};
Curve Loop(104) = {36, -148, -120, 150};
Plane Surface(125) = {104};
Curve Loop(106) = {99, 137, -15, -140};
Plane Surface(126) = {106};
Curve Loop(108) = {101, 141, -17, -136};
Plane Surface(127) = {108};
Curve Loop(110) = {109, 145, -25, -146};
Plane Surface(128) = {110};

// Fluid BL Lengthwise Outlet
Curve Loop(111) = {105, 145, -21, -144};
Plane Surface(129) = {111};
Curve Loop(113) = {96, 138, -12, -136};
Plane Surface(130) = {113};
Curve Loop(114) = {98, 137, -14, -139};
Plane Surface(131) = {114};
Curve Loop(115) = {116, 149, -32, -148};
Plane Surface(132) = {115};
Curve Loop(117) = {1, -129, -85, 128};
Plane Surface(133) = {117};
Curve Loop(119) = {87, 130, -3, -131};
Plane Surface(134) = {119};

// Solid lengthwise outlet surfaces
Curve Loop(121) = {45, 179, -3, -180};
Plane Surface(135) = {121};
Curve Loop(123) = {43, 178, -1, -177};
Plane Surface(136) = {123};
Curve Loop(125) = {74, 198, -32, -197};
Plane Surface(137) = {125};
Curve Loop(127) = {14, -186, -56, 188};
Plane Surface(138) = {127};
Curve Loop(128) = {54, 187, -12, -185};
Plane Surface(139) = {128};
Curve Loop(129) = {63, 194, -21, -193};
Plane Surface(140) = {129};

////////////////////////////////////////////////
// Structured Mesh Regions
////////////////////////////////////////////////

// Inner surfaces
Transfinite Surface {3} = {5, 7, 8, 6} Right;
Transfinite Surface {18} = {6, 8, 24, 23} Right;
Transfinite Surface {15} = {23, 24, 16, 13} Right;
Transfinite Surface {6} = {13, 16, 15, 14} Right;
Transfinite Surface {9} = {14, 15, 20, 19} Right;
Transfinite Surface {12} = {19, 20, 7, 5} Right;
Transfinite Surface {2} = {4, 5, 6, 2} Right;
Transfinite Surface {17} = {2, 6, 23, 21} Right;
Transfinite Surface {14} = {21, 23, 13, 10} Right;
Transfinite Surface {5} = {10, 13, 14, 9} Right;
Transfinite Surface {8} = {9, 14, 19, 18} Right;
Transfinite Surface {7} = {11, 9, 18, 17} Right;
Transfinite Surface {10} = {17, 18, 4, 3} Right;
Transfinite Surface {1} = {3, 4, 2, 1} Right;
Transfinite Surface {16} = {1, 2, 21, 22} Right;
Transfinite Surface {13} = {22, 21, 10, 12} Right;
Transfinite Surface {4} = {12, 10, 9, 11} Right;
Transfinite Surface {11} = {18, 19, 5, 4} Right;

// Outer surfaces
Transfinite Surface {21} = {29, 31, 32, 30} Right;
Transfinite Surface {20} = {28, 29, 30, 26} Right;
Transfinite Surface {19} = {27, 28, 26, 25} Right;
Transfinite Surface {36} = {30, 32, 48, 47} Right;
Transfinite Surface {35} = {26, 30, 47, 45} Right;
Transfinite Surface {34} = {25, 26, 45, 46} Right;
Transfinite Surface {33} = {47, 48, 40, 37} Right;
Transfinite Surface {32} = {45, 47, 37, 34} Right;
Transfinite Surface {31} = {46, 45, 34, 36} Right;
Transfinite Surface {24} = {37, 40, 39, 38} Right;
Transfinite Surface {23} = {34, 37, 38, 33} Right;
Transfinite Surface {22} = {36, 34, 33, 35} Right;
Transfinite Surface {27} = {38, 39, 44, 43} Right;
Transfinite Surface {26} = {33, 38, 43, 42} Right;
Transfinite Surface {25} = {35, 33, 42, 41} Right;
Transfinite Surface {30} = {43, 44, 31, 29} Right;
Transfinite Surface {29} = {42, 43, 29, 28} Right;
Transfinite Surface {28} = {41, 42, 28, 27} Right;

// Inner BL Surfaces

Transfinite Surface {39} = {53, 55, 56, 54} Right;
Transfinite Surface {38} = {52, 53, 54, 50} Right;
Transfinite Surface {37} = {51, 52, 50, 49} Right;
Transfinite Surface {54} = {54, 56, 72, 71} Right;
Transfinite Surface {53} = {50, 54, 71, 69} Right;
Transfinite Surface {52} = {49, 50, 69, 70} Right;
Transfinite Surface {51} = {71, 72, 64, 61} Right;
Transfinite Surface {50} = {69, 71, 61, 58} Right;
Transfinite Surface {49} = {70, 69, 58, 60} Right;
Transfinite Surface {42} = {61, 64, 63, 62} Right;
Transfinite Surface {41} = {58, 61, 62, 57} Right;
Transfinite Surface {40} = {60, 58, 57, 59} Right;
Transfinite Surface {45} = {62, 63, 68, 67} Right;
Transfinite Surface {44} = {57, 62, 67, 66} Right;
Transfinite Surface {43} = {59, 57, 66, 65} Right;
Transfinite Surface {48} = {67, 68, 55, 53} Right;
Transfinite Surface {47} = {66, 67, 53, 52} Right;
Transfinite Surface {46} = {65, 66, 52, 51} Right;

// Solid Outlet Circumfrential Surfaces
Transfinite Surface {80} = {41, 27, 3, 17} Left;
Transfinite Surface {75} = {27, 25, 1, 3} Left;
Transfinite Surface {76} = {25, 46, 22, 1} Left;
Transfinite Surface {77} = {12, 36, 46, 22} Left; //{22, 46, 36, 12} Left;
Transfinite Surface {78} = {11, 12, 36, 35} Left;
Transfinite Surface {79} = {35, 11, 17, 41} Left; //{41, 17, 11, 35} Left;

// Fluid BL Outlet Circumfrential surfaces
Transfinite Surface {74} = {3, 51, 65, 17} Right;
Transfinite Surface {69} = {3, 1, 49, 51} Left;
Transfinite Surface {70} = {49, 1, 22, 70} Right;
Transfinite Surface {71} = {70, 22, 12, 60} Right;
Transfinite Surface {72} = {11, 59, 60, 12} Right;
Transfinite Surface {73} = {17, 65, 59, 11} Right;

// Solid Top Sinus Circumfrential surfaces
Transfinite Surface {99} = {28, 4, 2, 26} Left; //{26, 2, 4, 28} Left;
Transfinite Surface {100} = {26, 45, 21, 2} Left;
Transfinite Surface {101} = {45, 21, 10, 34} Left;
Transfinite Surface {102} = {34, 10, 9, 33} Left;
Transfinite Surface {103} = {33, 42, 18, 9} Left;
Transfinite Surface {104} = {4, 18, 42, 28} Left;

// Fluid BL Top Sinus Circufrential Surfaces
Transfinite Surface {93} = {50, 52, 4, 2} Left;
Transfinite Surface {97} = {66, 18, 9, 57} Left;
Transfinite Surface {98} = {52, 66, 18, 4} Left;
Transfinite Surface {94} = {21, 2, 50, 69} Right;
Transfinite Surface {95} = {10, 21, 69, 58} Right;
Transfinite Surface {96} = {57, 58, 10, 9} Left;

// Solid Root Circumfrential surfaces
Transfinite Surface {87} = {6, 5, 29, 30} Left;
Transfinite Surface {88} = {23, 6, 30, 47} Left;
Transfinite Surface {89} = {47, 23, 13, 37} Left;
Transfinite Surface {90} = {37, 13, 14, 38} Left;
Transfinite Surface {91} = {19, 14, 38, 43} Left;
Transfinite Surface {92} = {5, 19, 43, 29} Left;

// Fluid BL Root Circumfrential surfaces
Transfinite Surface {81} = {5, 53, 54, 6} Left;
Transfinite Surface {82} = {6, 54, 71, 23} Left;
Transfinite Surface {83} = {23, 71, 61, 13} Left;
Transfinite Surface {84} = {62, 14, 13, 61} Left;
Transfinite Surface {85} = {67, 62, 14, 19} Left;
Transfinite Surface {86} = {19, 67, 53, 5} Left;

// Solid Inlet Circumfrential surfaces
Transfinite Surface {63} = {8, 7, 31, 32} Left;
Transfinite Surface {68} = {32, 8, 24, 48} Left;
Transfinite Surface {67} = {24, 16, 40, 48} Right;
Transfinite Surface {66} = {39, 15, 16, 40} Right;
Transfinite Surface {65} = {20, 15, 39, 44} Left;
Transfinite Surface {64} = {7, 20, 44, 31} Left;

// Fluid BL Inlet Circumfrential surfaces
Transfinite Surface {57} = {7, 55, 56, 8} Left;
Transfinite Surface {58} = {72, 56, 8, 24} Left;
Transfinite Surface {59} = {64, 72, 24, 16} Left;
Transfinite Surface {60} = {16, 64, 63, 15} Left;
Transfinite Surface {61} = {15, 63, 68, 20} Left;
Transfinite Surface {62} = {20, 68, 55, 7} Left;

// Solid Outlet Lengthwise
Transfinite Surface {136} = {2, 1, 25, 26} Right;
Transfinite Surface {135} = {28, 27, 3, 4} Left;
Transfinite Surface {140} = {18, 17, 41, 42} Right;
Transfinite Surface {139} = {11, 9, 33, 35} Left;
Transfinite Surface {138} = {10, 34, 36, 12} Right;
Transfinite Surface {137} = {21, 45, 46, 22} Right;

// Fluid BL outlet lengthwise
Transfinite Surface {131} = {12, 60, 58, 10} Right;
Transfinite Surface {132} = {22, 70, 69, 21} Right;
Transfinite Surface {133} = {1, 49, 50, 2} Right;
Transfinite Surface {134} = {3, 51, 52, 4} Right;
Transfinite Surface {129} = {17, 18, 66, 65} Right;
Transfinite Surface {130} = {11, 9, 57, 59} Right;

// Solid sinus lengthwise
Transfinite Surface {119} = {38, 33, 9, 14} Left;
Transfinite Surface {120} = {37, 34, 10, 13} Left;
Transfinite Surface {121} = {47, 45, 21, 23} Left;
Transfinite Surface {122} = {30, 26, 2, 6} Left;
Transfinite Surface {117} = {29, 28, 4, 5} Left;
Transfinite Surface {118} = {43, 42, 18, 19} Left;

// Fluid BL sinus lengthwise
Transfinite Surface {123} = {5, 4, 52, 53} Left;
Transfinite Surface {128} = {19, 18, 66, 67} Left;
Transfinite Surface {127} = {14, 9, 57, 62} Left;
Transfinite Surface {126} = {13, 10, 58, 61} Left;
Transfinite Surface {125} = {23, 21, 69, 71} Left;
Transfinite Surface {124} = {6, 2, 50, 54} Left;

// Solid inlet lengthwise
Transfinite Surface {108} = {40, 37, 13, 16} Left;
Transfinite Surface {109} = {48, 47, 23, 24} Left;
Transfinite Surface {110} = {32, 30, 6, 8} Left;
Transfinite Surface {105} = {31, 29, 5, 7} Left;
Transfinite Surface {106} = {44, 43, 19, 20} Left;
Transfinite Surface {107} = {39, 38, 14, 15} Left;

// Fluid BL inlet lengthwise
Transfinite Surface {116} = {15, 14, 62, 63} Left;
Transfinite Surface {115} = {16, 13, 61, 64} Left;
Transfinite Surface {114} = {24, 23, 71, 72} Left;
Transfinite Surface {113} = {8, 6, 54, 56} Left;
Transfinite Surface {112} = {7, 5, 53, 55} Left;
Transfinite Surface {111} = {20, 19, 67, 68} Left;

// Center Fluid Volume
Surface Loop(1) = {55, 56, 52, 37, 43, 46, 40, 49, 53, 38, 47, 44, 39, 54, 50, 41, 48, 45, 42, 51};
Volume(1) = {1};

// Fluid BL inlet volumes
Surface Loop(2) = {57, 81, 113, 112, 39, 3};
Volume(2) = {2};
Surface Loop(3) = {54, 18, 113, 82, 58, 114};
Volume(3) = {3};
Surface Loop(4) = {51, 114, 15, 83, 59, 115};
Volume(4) = {4};
Surface Loop(5) = {42, 6, 115, 116, 60, 84};
Volume(5) = {5};
Surface Loop(6) = {9, 116, 45, 111, 85, 61};
Volume(6) = {6};
Surface Loop(7) = {48, 12, 111, 62, 86, 112};
Volume(7) = {7};

Transfinite Volume{2};
Transfinite Volume{3} = {71, 54, 6, 23, 72, 56, 8, 24};
Transfinite Volume{4} = {61, 71, 23, 13, 64, 72, 24, 16};
Transfinite Volume{5} = {62, 61, 13, 14, 63, 64, 16, 15};
Transfinite Volume{6} = {67, 62, 14, 19, 68, 63, 15, 20};
Transfinite Volume{7} = {53, 67, 19, 5, 55, 68, 20, 7};

// Fluid BL Sinus Region volumes
Surface Loop(8) = {44, 8, 85, 97, 128, 127};
Volume(8) = {8};
Surface Loop(9) = {41, 5, 127, 96, 84, 126};
Volume(9) = {9};
Surface Loop(10) = {50, 14, 126, 95, 83, 125};
Volume(10) = {10};
Surface Loop(11) = {53, 17, 82, 94, 124, 125};
Volume(11) = {11};
Surface Loop(12) = {123, 47, 11, 128, 98, 86};
Volume(12) = {12};
Surface Loop(13) = {53, 17, 94, 82, 125, 124};
Volume(13) = {13};
Surface Loop(14) = {38, 2, 81, 123, 93, 124};
Volume(14) = {14};
Transfinite Volume{8} = {66, 57, 9, 18, 67, 62, 14, 19};
Transfinite Volume{9} = {57, 58, 10, 9, 62, 61, 13, 14};
Transfinite Volume{10} = {58, 69, 21, 10, 61, 71, 23, 13};
Transfinite Volume{11} = {69, 50, 2, 21, 71, 54, 6, 23};
Transfinite Volume{12} = {52, 66, 18, 4, 53, 67, 19, 5};
Transfinite Volume{13} = {69, 50, 2, 21, 71, 54, 6, 23};
Transfinite Volume{14} = {50, 52, 4, 2, 54, 53, 5, 6};

// Fluid BL Outlet volumes
Surface Loop(15) = {43, 7, 97, 73, 129, 130};
Volume(15) = {15};
Surface Loop(16) = {40, 4, 96, 72, 131, 130};
Volume(16) = {16};
Surface Loop(17) = {49, 13, 95, 131, 132, 71};
Volume(17) = {17};
Surface Loop(18) = {52, 16, 70, 133, 132, 94};
Volume(18) = {18};
Surface Loop(19) = {37, 1, 133, 134, 69, 93};
Volume(19) = {19};
Surface Loop(20) = {46, 10, 74, 134, 98, 129};
Volume(20) = {20};
Transfinite Volume{15} = {65, 59, 11, 17, 66, 57, 9, 18};
Transfinite Volume{16} = {59, 60, 12, 11, 57, 58, 10, 9};
Transfinite Volume{17} = {60, 70, 22, 12, 58, 69, 21, 10};
Transfinite Volume{18} = {70, 49, 1, 22, 69, 50, 2, 21};
Transfinite Volume{19} = {49, 51, 3, 1, 50, 52, 4, 2};
Transfinite Volume{20} = {51, 65, 17, 3, 52, 66, 18, 4};

// Solid inlet volumes
Surface Loop(21) = {27, 65, 9, 106, 91, 107};
Volume(21) = {21};
Surface Loop(22) = {24, 66, 6, 107, 108, 90};
Volume(22) = {22};
Surface Loop(23) = {33, 67, 15, 109, 108, 89};
Volume(23) = {23};
Surface Loop(24) = {36, 68, 18, 109, 88, 110};
Volume(24) = {24};
Surface Loop(25) = {21, 63, 3, 110, 105, 87};
Volume(25) = {25};
Surface Loop(26) = {105, 12, 30, 64, 106, 92};
Volume(26) = {26};
Transfinite Volume{21} = {19, 14, 38, 43, 20, 15, 39, 44};
Transfinite Volume{22} = {14, 13, 37, 38, 15, 16, 40, 39};
Transfinite Volume{23} = {13, 23, 47, 37, 16, 24, 48, 40};
Transfinite Volume{24} = {23, 6, 30, 47, 24, 8, 32, 48};
Transfinite Volume{25} = {6, 5, 29, 30, 8, 7, 31, 32};
Transfinite Volume{26} = {5, 19, 43, 29, 7, 20, 44, 31};

// Solid sinus volumes
Surface Loop(27) = {11, 29, 117, 118, 92, 104};
Volume(27) = {27};
Surface Loop(28) = {26, 8, 118, 91, 119, 103};
Volume(28) = {28};
Surface Loop(29) = {23, 5, 119, 102, 90, 120};
Volume(29) = {29};
Surface Loop(30) = {32, 14, 120, 89, 101, 121};
Volume(30) = {30};
Surface Loop(31) = {35, 17, 121, 88, 100, 122};
Volume(31) = {31};
Surface Loop(32) = {20, 2, 122, 117, 87, 99};
Volume(32) = {32};
Transfinite Volume{27} = {4, 18, 42, 28, 5, 19, 43, 29};
Transfinite Volume{28} = {18, 9, 33, 42, 19, 14, 38, 43};
Transfinite Volume{29} = {9, 10, 34, 33, 14, 13, 37, 38};
Transfinite Volume{30} = {10, 21, 45, 34, 13, 23, 47, 37};
Transfinite Volume{31} = {21, 2, 26, 45, 23, 6, 30, 47};
Transfinite Volume{32} = {2, 4, 28, 26, 6, 5, 29, 30};

// Solid outlet volumes
Surface Loop(33) = {19, 75, 1, 136, 135, 99};
Volume(33) = {33};
Surface Loop(34) = {135, 28, 80, 10, 140, 104};
Volume(34) = {34};
Surface Loop(35) = {140, 7, 25, 79, 139, 103};
Volume(35) = {35};
Surface Loop(36) = {22, 4, 139, 138, 78, 102};
Volume(36) = {36};
Surface Loop(37) = {138, 31, 77, 13, 101, 137};
Volume(37) = {37};
Surface Loop(38) = {34, 76, 16, 137, 136, 100};
Volume(38) = {38};
Transfinite Volume{33} = {1, 3, 27, 25, 2, 4, 28, 26};
Transfinite Volume{34} = {3, 17, 41, 27, 4, 18, 42, 28};
Transfinite Volume{35} = {17, 11, 35, 41, 18, 9, 33, 42};
Transfinite Volume{36} = {11, 12, 36, 35, 9, 10, 34, 33};
Transfinite Volume{37} = {12, 22, 46, 36, 10, 21, 45, 34};
Transfinite Volume{38} = {22, 1, 25, 46, 21, 2, 26, 45};


////////////////////////////////////////////////
// Unstructured Mesh Element Sizing
////////////////////////////////////////////////


// Near-valve Mesh Refinement
Field[1] = Cylinder;
Field[1].Radius = 3.0;
Field[1].VIn = h_ref_valve;
Field[1].VOut = h_max;
Field[1].ZAxis = 1.5;
Field[1].ZCenter = 1;

// Near-valve Mesh Refinement
Field[2] = Cylinder;
Field[2].Radius = 3.0;
Field[2].VIn = h_ref_upstream;
Field[2].VOut = h_max;
Field[2].ZAxis = 1.5;
Field[2].ZCenter = 2.5;

// Background Field
Field[3] = Min;
Field[3].FieldsList = {1,2};
Field[4] = Mean;
Field[4].InField = 3;
Field[4].Delta = h_max*2;
Background Field = 4;


////////////////////////////////////////////////
// Physical Groups
////////////////////////////////////////////////

Physical Volume("fluid", 1) = {1:20};
Physical Volume("solid", 2) = {21:38};
Physical Surface("interface", 3) = {1:18};
Physical Surface("fluid_inlet", 4) = {55,57:62};
Physical Surface("fluid_outlet", 5) = {56,69:74};
Physical Surface("solid_inlet", 6) = {63:68};
Physical Surface("solid_outlet", 7) = {75:80};
Physical Surface("solid_outer", 8) = {19:36};


// I double-added a volume by mistake
Delete {
  Volume{11}; 
}
