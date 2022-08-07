'''
Simulate bloodflow, aorta wall, and valve deformation for a tricuspid surgical
heart valve. A collaboration of the Hsu group at Iowa State University and the
Kamensky group at University of California San Diego.

Notes:
 - It is recomended that this script be run first on a coarse mesh in serial to
   JIT-compile the weak forms. Be aware, the forms for the valve computations
   take an extremely long time to compile and require a great deal of memory.
 - To generate the aorta mesh, use GMSH (https://gmsh.info/) with the provided
   .geo file in the ``aorta`` directory. To convert the .msh file to a
   FEniCS-compatible XDMF format (.xdmf + .h5), use the ChaMeleon utility
   provided with VarMINT.
 - The results are best visualized with a single slice in the fluid domain.
   This script also saves the shell normal vectors so that Paraview can
   smoothly light the shell surface (these must be manually specified).
   Paraview state files with sample visualizations are provided in the ``vis``
   directory.
 - This script works in parallel and in high-performance computing
   environments. The ``hpc`` directory has some instructions and samples in the
   context of the Stampede2 cluster at the Texas Advanced Computing Center
   (TACC).
'''

###############################################################
#### Housekeeping and Imports #################################
###############################################################
from dolfin import *
import ufl
import os
import sys
import datetime

# import numerical-related routines
import csv
import numpy as np

# build command-line parser and parse arguments
import argparse
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# constants
DEFAULT_LEAFLET_BC_EDGES = [[0,0],[0,1],[1,1]]

# general arguments
p = parser.add_argument_group("general")
p.add_argument("--results-folder",dest="RESULTS_FOLDER",default="./results/",
               help="Folder in which the result files are written.")
p.add_argument("--restarts-folder",dest="RESTARTS_FOLDER",
               default="./restarts/",
               help="Folder in which the restart files are written.")   
p.add_argument("--no-use-tsfc",dest="USE_TSFC",
               default=True, action='store_false',
               help='Use UFLACS instead of TSFC for a form compiler.')
p.add_argument("--flags",dest='FLAGS',nargs="+",type=int,
               default=[0,1,2,3,4,5,6,7,8,9,10,11],
               help='Mesh marker flags within the code, in the following \
                    order: none, fluid, solid, interface, fluid_inflow, \
                    fluid_outflow, solid_inflow, solid_outflow, solid_outer, \
                    stent_intersection, leaflet, stent.')
p.add_argument("--no-use-restarts",dest="USE_RESTARTS",
               default=True, action='store_false',
               help='Use the restart files if they exist.')
p.add_argument("--no-write-visualizations",dest="WRITE_VIS",
               default=True, action='store_false',
               help='Write the simulation visualization files.')
p.add_argument("--vis-skip-fluid-solid",dest="VIS_SKIP_FLUID_SOLID",
               default=20, type=int,
               help='Write the simulation visualization files every \
                    VIS_SKIP_FLUID_SOLID steps.')
p.add_argument("--vis-skip-shell",dest="VIS_SKIP_SHELL",
               default=5, type=int,
               help='Write the simulation visualization files every \
                    VIS_SKIP_SHELL steps.')
p.add_argument("--compute-fluid-stats-skip",dest="WRITE_FLUID_STATS_SKIP",
               default=10, type=int,
               help='Compute fluid stats every WRITE_FLUID_STATS_SKIP \
                   steps.')                    
p.add_argument("--no-write-restarts",dest="WRITE_RESTARTS",
               default=True, action='store_false',
               help='Write the simulation restart files.')
p.add_argument("--restart-skip",dest="WRITE_RESTARTS_SKIP",
               default=50, type=int,
               help='Write the simulation restart files every \
                    WRITE_RESTARTS_SKIP steps.')
p.add_argument("--no-log-timings",dest="LOG_TIMINGS",
               default=True, action='store_false',
               help='Log the timing map for each step.')
p.add_argument("--no-write-fluid-stats",dest="WRITE_FLUID_STATS",
               default=True, action='store_false',
               help='Write the fluid stats to a file.')
p.add_argument("--no-write-shell-intersections",dest="WRITE_SHELL_INTERSECTIONS",
               default=True, action='store_false',
               help='Write the shell intersection file.')
p.add_argument("--use-shell-intersections",dest="USE_SHELL_INTERSECTIONS",
               default=False, action='store_true',
               help='Use shell intersection file.')
p.add_argument("--outfile-fluid-solid-mesh",dest="FILE_FSM",
               default="fluid-solid-mesh-results.xdmf",
               help="The name of the fluid/solid/mesh results file.") 
p.add_argument("--outfile-shell-results",dest="FILE_SHELL_DISP",
               default="shell-results.xdmf",
               help="The name of the shell displacemnt results file.") 
p.add_argument("--outfile-qoi-data",dest="FILE_QOI_DATA",
               default="qoi-data.csv",
               help="The name of the quantity of interest (QOI) csv file.") 

# time-stepping arguments
p = parser.add_argument_group("time stepping")
p.add_argument("--start-time",dest="START_TIME",type=float,default=0.6,
               help='The start time when not using restart files.')
p.add_argument("--start-time-step",dest="START_TIME_STEP",type=int,default=1,
               help='The start time step when not using restart files.')
p.add_argument("--num-steps",dest="NUM_TIME_STEPS",type=int,default=11200,
               help='The number of time steps to take.')
p.add_argument("--delta-t",dest="DT",type=float,default=1e-4,
               help='The time step interval.')
p.add_argument("--rho-infinity",dest="RHO_INFINITY",type=float,default=0.0,
               help='The high-frequency dissipation for the \
                    Generalized-Alpha time integrator.')

# fluid-solid problem parameters
p = parser.add_argument_group("fluid-solid problem")
p.add_argument("--mesh-folder",dest="MESH_FOLDER",
               default="./mesh-M0/",
               help="Folder that contains the fluid-solid mesh and markers.")
p.add_argument("--mesh-filename",dest="MESH_FILENAME",
               default="aorta_mesh.xdmf",
               help="Filename of the fluid-solid mesh.")
p.add_argument("--subdomains-filename",dest="SUBDOMAINS_FILENAME",
               default="aorta_subdomains.xdmf",
               help="Filename of the fluid-solid subdomains markers.")
p.add_argument("--boundaries-filename",dest="BOUNDARIES_FILENAME",
               default="aorta_boundaries.xdmf",
               help="Filename of the fluid-solid boundary markers.")
p.add_argument("--markers-string",dest="MARKERS_STRING",default="markers",
               help="Mesh physical group marker string.")
p.add_argument("--polynomial-degree",dest="POLYNOMIAL_DEGREE",
               type=int,default=1,
               help="Fluid-solid problem polynomial degree.")            

# solid problem parameters
p = parser.add_argument_group("solid problem")
p.add_argument("--solid-E",dest="SOLID_E",
               type=float, default=1e7,
               help='The Youngs modulus of the solid material.')
p.add_argument("--solid-nu",dest="SOLID_NU",
               type=float, default=0.45,
               help='Poissons ratio of the solid material.')
p.add_argument("--solid-rho",dest="SOLID_RHO",
               type=float, default=1.0,
               help='The density of the solid material.')
p.add_argument("--solid-c",dest="SOLID_C",
               type=float, default=1e4,
               help='The damping coefficient for the solid material.')
p.add_argument("--freeze-solid",dest="FREEZE_SOLID",
               action='store_true', default=False,
               help='Freeze the solid problem.')

# fluid problem parameters
p = parser.add_argument_group("fluid problem")
p.add_argument("--pressure-waveform-folder",dest="PRESSURE_WAVEFORM_FOLDER",
               default="./pressure-waveforms/",
               help='The folder of the pressure waveforms.')
p.add_argument("--pressure-waveform-file",dest="PRESSURE_WAVEFORM_FILE",
               default="pressures-realistic-start-after-closure.csv",
               help='The filename of the pressure waveform in the path.')
p.add_argument("--pressure-unit-conversation",dest="PRESSURE_UNIT_CONVERSION",
               type=float, default=1333.22,
               help='A multiplier for pressure units to CGS units.')
p.add_argument("--fluid-rho",dest="FLUID_RHO",
               type=float, default=1.0,
               help='The fluid density in CGS units.')
p.add_argument("--fluid-mu",dest="FLUID_MU",
               type=float, default=3e-2,
               help='The fluid viscosity in CGS units.')
p.add_argument("--fluid-resistance-bc",dest="FLUID_RES_BC",
               type=float, default=70.0,
               help='The fluid outlet resistance boundary condition \
                    parameter.')
p.add_argument("--fluid-gamma",dest="FLUID_GAMMA",
               type=float, default=1.0,
               help='The fluid weak bc enforcement constraint.')
p.add_argument("--fluid-stability-eps",dest="FLUID_STABILITY_EPS",
               type=float, default=1e-3,
               help='The fluid stability condition scaling near an \
                    immersed shell.')
p.add_argument("--fluid-c-i",dest="FLUID_C_I",
               type=float, default=3.0,
               help='A fluid stability scaling term.')
p.add_argument("--fluid-c-t",dest="FLUID_C_T",
               type=float, default=4.0,
               help='A fluid stability scaling term.')
p.add_argument("--ksp-rel-tol",dest="KSP_REL_TOL",
               type=float, default=1e-2,
               help='The fluid Krylov solver relative tolerance.')    
p.add_argument("--ksp-max-iters",dest="KSP_MAX_ITERS",
               type=int, default=300,
               help='The fluid Krylov solver maximum iterations.') 
p.add_argument("--log-ksp",dest="LOG_KSP",
               default=False, action='store_true',
               help='Log the progress of the fluid linear solver.')

# shell problem parameters
p = parser.add_argument_group("shell problem")
p.add_argument("--solid-stent-intersections-filename",
               dest="SOLID_STENT_INTERSECTIONS_FILENAME",
               default="aorta_solid-stent-intersections.xdmf",
               help="Filename of the fluid-solid boundary markers.")
p.add_argument("--valve-folder",dest="VALVE_FOLDER",
               default="./valves/coarse/",
               help='The folder that contains the valve smesh files.')
p.add_argument("--valve-smesh-prefix",dest="VALVE_SMESH_PREFIX",
               default="smesh.",
               help='The prefix for the valve smesh files.')
p.add_argument("--valve-smesh-suffix",dest="VALVE_SMESH_SUFFIX",
               default=".txt",
               help='The suffix for the valve smesh files.')
p.add_argument("--leaflet-patches",dest="LEAFLET_PATCHES",
               type=int,nargs="+", default=[1,2,3],
               help='A list of the leaflet patches.')
p.add_argument("--stent-patches",dest="STENT_PATCHES",
               type=int,nargs="+", default=[4,5,6,7,8,9,10],
               help='A list of the stent patches.')
p.add_argument("--stentless",dest="STENTLESS",
               default=False, action='store_true',
               help='For stentless valves.')
p.add_argument("--num-patches",dest="NUM_PATCHES",
               type=int, default=10,
               help='The number of valve patches to import.')
p.add_argument("--leaflet-bc-layers",dest="LEAFLET_BC_LAYERS",
               type=int, default=2,
               help='The number of control point layers fix for a boundary \
                    condition, 1=pinned, 2=clamped, etc.')
p.add_argument("--spline-quad-deg",dest="SPLINE_QUAD_DEG",
               type=int, default=3,
               help='The k-degree quadrature rule for the spline.')
p.add_argument("--r-self",dest="R_SELF",
               type=float, default=0.0308,
               help='The self-intersection radius for shell contact.')
p.add_argument("--r-max",dest="R_MAX",
               type=float, default=0.0237,
               help='The maximum radius for shell contact.')
p.add_argument("--k-contact",dest="K_CONTACT",
               type=float, default=1e11,
               help='The contact penalty parameter.')
p.add_argument("--s-contact",dest="S_CONTACT",
               type=float, default=0.2,
               help='The contact smoothing parameter in [0,1] for no \
                    smoothing to complete smoothing.')
p.add_argument("--leaflet-bc-edge",dest="LEAFLET_BC_EDGES",
               type=int, nargs=2, default=[], action='append',
               help='The direction, side of the leaflet on which to apply \
                    the leaflet BC. Repeat for each edge.')
p.add_argument("--leaflet-material",dest="LEAFLET_MATERIAL",
               type=str, default='isotropic-lee-sacks',
               help="Leaflet material.")
p.add_argument("--leaflet-c0",dest="LEAFLET_C0",
               type=float, default=676080.0,
               help="Leaflet material parameter c0.")
p.add_argument("--leaflet-c1",dest="LEAFLET_C1",
               type=float, default=132848.0,
               help="Leaflet material parameter c1.")
p.add_argument("--leaflet-c2",dest="LEAFLET_C2",
               type=float, default=38.1878,
               help="Leaflet material parameter c2.")
p.add_argument("--leaflet-c3",dest="LEAFLET_C3",
               type=float, default=0.0,
               help="Leaflet material parameter c3.")
p.add_argument("--leaflet-w",dest="LEAFLET_W",
               type=float, default=0.0,
               help="Leaflet material parameter w.")
p.add_argument("--leaflet-thickness",dest="LEAFLET_THICKNESS",
               type=float, default=0.0386,
               help='The thickness of the leaflet.')
p.add_argument("--leaflet-rho",dest="LEAFLET_RHO",
               type=float, default=1.0,
               help='The initial leaflet density.')

# coupling problem parameters
p = parser.add_argument_group("coupling problem")
p.add_argument("--block-tol",dest="BLOCK_REL_TOL",
               type=float, default=1e-3,
               help='The block iteration relative tolerance.')
p.add_argument("--block-max-iters",dest="BLOCK_MAX_ITERS",
               type=int, default=15,
               help='The maximum number of block iterations.')
p.add_argument("--block-throw-error",dest="BLOCK_NO_ERROR",
               default=True, action='store_false',
               help='Do not stop the simulation if the block iteration \
                    does not converge, just go to the max iterations.')
p.add_argument("--dal-penalty",dest="DAL_PENALTY",
               type=float, default=5e2,
               help='The DAL coupling penalty.')
p.add_argument("--dal-r",dest="DAL_R",
               type=float, default=1e-5,
               help='The DAL coupling stabilization term.')

def safe_filename(f):
    ''' 
    Return a safe filename that is different from any other in the location by
    adding 000, 001, 002, etc to the end of the provided filename. 
    '''
    intended_folder = os.path.dirname(f)
    if intended_folder=="":
        intended_folder = "./"
    intended_basename = os.path.basename(f)
    intended_basename_stripped = os.path.splitext(intended_basename)[0]
    intended_ext = os.path.splitext(intended_basename)[1]
    
    existing = [-1,]
    if os.path.exists(intended_folder):
        for filename in os.listdir(intended_folder):
            basename = os.path.basename(filename)
            basename_stripped = os.path.splitext(basename)[0]
            ext = os.path.splitext(basename)[1]
            if basename_stripped[:-4]==intended_basename_stripped \
                and ext==intended_ext:
                existing.append(int(basename_stripped[-3:]))

    return intended_folder + "/" + intended_basename_stripped + "_" + \
           f'{max(existing)+1:03}' + intended_ext

# make list of the input arguments for logging
args = parser.parse_args()
script_parameters_message = ''
for s in sys.argv:
    if s[0]=="-":     # a cheap check to find the arguments
        script_parameters_message += "\n"
    script_parameters_message += s
    script_parameters_message += " "

# parse arguments
RESTARTS_FOLDER = args.RESTARTS_FOLDER
USE_TSFC = args.USE_TSFC
LABELS = ("none", "fluid", "solid", "interface", "fluid_inflow", 
          "fluid_outflow", "solid_inflow", "solid_outflow", "solid_outer",
          "stent_intersection", "leaflet", "stent")
FLAG = {}
for i,label in enumerate(LABELS):
    FLAG[label] = args.FLAGS[i]
USE_RESTARTS = args.USE_RESTARTS
WRITE_VIS = args.WRITE_VIS
WRITE_VIS_SKIP_FLUID_SOLID = args.VIS_SKIP_FLUID_SOLID
WRITE_VIS_SKIP_SHELL = args.VIS_SKIP_SHELL
WRITE_FLUID_STATS_SKIP = args.WRITE_FLUID_STATS_SKIP
WRITE_RESTARTS = args.WRITE_RESTARTS
WRITE_RESTARTS_SKIP = args.WRITE_RESTARTS_SKIP
WRITE_SHELL_INTERSECTIONS = args.WRITE_SHELL_INTERSECTIONS
LOG_TIMINGS = args.LOG_TIMINGS
WRITE_FLUID_STATS = args.WRITE_FLUID_STATS
USE_SHELL_INTERSECTIONS = args.USE_SHELL_INTERSECTIONS
RESULTS_FOLDER = args.RESULTS_FOLDER
FILEPATH_OUT_FSM = safe_filename(os.path.join(RESULTS_FOLDER, 
                                              args.FILE_FSM))
FILEPATH_OUT_SHELL = safe_filename(os.path.join(RESULTS_FOLDER, 
                                                     args.FILE_SHELL_DISP))
FILEPATH_OUT_QOI_DATA = safe_filename(os.path.join(RESULTS_FOLDER, 
                                                   args.FILE_QOI_DATA))

START_TIME = args.START_TIME
START_TIME_STEP = args.START_TIME_STEP
NUM_TIME_STEPS = args.NUM_TIME_STEPS
DT = Constant(args.DT)
RHO_INFINITY = Constant(args.RHO_INFINITY)

FILEPATH_IN_MESH = args.MESH_FOLDER + args.MESH_FILENAME
FILEPATH_IN_SUBDOMAINS = args.MESH_FOLDER + args.SUBDOMAINS_FILENAME
FILEPATH_IN_BOUNDARIES = args.MESH_FOLDER + args.BOUNDARIES_FILENAME
MARKER = args.MARKERS_STRING
BKG_POLYNOMIAL_DEGREE = args.POLYNOMIAL_DEGREE

SOLID = {'E': Constant(args.SOLID_E), 
         'nu': Constant(args.SOLID_NU), 
         'rho': Constant(args.SOLID_RHO),
         'c': Constant(args.SOLID_C)}
FREEZE_SOLID = args.FREEZE_SOLID
FLUID = {'rho': Constant(args.FLUID_RHO), 
         'mu': Constant(args.FLUID_MU)}
FILEPATH_IN_PRESSURES = args.PRESSURE_WAVEFORM_FOLDER + \
                        args.PRESSURE_WAVEFORM_FILE
PRESSURE_UNIT_CONVERSION = args.PRESSURE_UNIT_CONVERSION
FLUID_NUMERICAL = {'R': Constant(args.FLUID_RES_BC), 
                   'gamma': Constant(args.FLUID_GAMMA), 
                   'stab_eps': Constant(args.FLUID_STABILITY_EPS), 
                   'C_i': Constant(args.FLUID_C_I),
                   'C_t': Constant(args.FLUID_C_T), 
                   'KSP': {'tol':args.KSP_REL_TOL,
                           'max_iters':args.KSP_MAX_ITERS,
                           'log':args.LOG_KSP}}

FILEPATH_SHELL_INTERSECTIONS = args.MESH_FOLDER + \
                               args.SOLID_STENT_INTERSECTIONS_FILENAME
FILEPATH_IN_SHELL_PREFIX = args.VALVE_FOLDER + args.VALVE_SMESH_PREFIX
FILEPATH_IN_SHELL_SUFFIX = args.VALVE_SMESH_SUFFIX
LEAFLET_PATCHES = args.LEAFLET_PATCHES
STENT_PATCHES = args.STENT_PATCHES
NUM_PATCHES = args.NUM_PATCHES
STENTLESS = args.STENTLESS
LEAFLET_BC_LAYERS = args.LEAFLET_BC_LAYERS
LEAFLET_BC_EDGES = args.LEAFLET_BC_EDGES
if not LEAFLET_BC_EDGES:
    LEAFLET_BC_EDGES = DEFAULT_LEAFLET_BC_EDGES
SPLINE_QUAD_DEG = args.SPLINE_QUAD_DEG
CONTACT = {'R_SELF': args.R_SELF, 
           'R_MAX': args.R_MAX, 
           'K': args.K_CONTACT,
           'S': args.S_CONTACT}
LEAFLET = {'material':args.LEAFLET_MATERIAL,
           'c0': Constant(args.LEAFLET_C0), 
           'c1': Constant(args.LEAFLET_C1), 
           'c2': Constant(args.LEAFLET_C2),
           'c3': Constant(args.LEAFLET_C3),
           'w': Constant(args.LEAFLET_W),
           'h': Constant(args.LEAFLET_THICKNESS), 
           'rho': Constant(args.LEAFLET_RHO)}

BLOCK_MAX_ITERS = args.BLOCK_MAX_ITERS
BLOCK_REL_TOL = args.BLOCK_REL_TOL
BLOCK_NO_ERROR = args.BLOCK_NO_ERROR
DAL_PENALTY = args.DAL_PENALTY
DAL_R = args.DAL_R



# import custom machinery for ALE-FSI computations
import VarMINT as vrmt                  # fluids
import SalaMANdER as sm                 # solids
import CouDALFISh as cfsh               # FSI coupling
from tIGAr.BSplines import *            # splines
from tIGAr.timeIntegration import *     # time integration
from ShNAPr.SVK import *                # shells
from ShNAPr.contact import *            # shell contact
from ShNAPr.kinematics import *         # shell kinematics
from ShNAPr.hyperelastic import *       # hyperelastic shell materials

# define MPI communicators
comm = MPI.comm_world
rank = comm.Get_rank()
size = comm.Get_size()

# logging active on master processor only
from CouDALFISh import log
set_log_active(False)
if (rank==0):
    set_log_active(True)

# put out a title for the script log
log(80*"=")
log("  Heart Valve Simulation")
log(80*"=")

# display basic information in log
log("MPI Size: " + str(size))
log("Current time: " + str(datetime.datetime.now()))
log("Script invoked with the following command-line arguments:")
log(script_parameters_message)
log(80*"=")

# set up global timer
global_timer = Timer("Global Elapsed Time")
global_timer.start()

# use TSFC form compiler representation
if USE_TSFC:
    parameters['form_compiler']['representation'] = 'tsfc'
    sys.setrecursionlimit(10000)



###############################################################
#### Load Restart Time Step and Value #########################
###############################################################

restarts_exist = os.path.exists(RESTARTS_FOLDER+"/step.dat")
restarting = restarts_exist and USE_RESTARTS  
if restarting:
    stepFile = open(RESTARTS_FOLDER+"/step.dat", "r")
    fs = stepFile.read()
    stepFile.close()
    tokens = fs.split()
    startStep = int(tokens[0])
    t = float(tokens[1])
else:
    startStep = START_TIME_STEP
    t = START_TIME



###############################################################
#### Import aorta mesh ########################################
###############################################################

# import mesh
log("Importing mesh.")
mesh = Mesh()
with Timer("HV 00: import mesh"):
    with XDMFFile(comm, FILEPATH_IN_MESH) as f:
        f.read(mesh)

# Mesh-derived quantities:
nsd = mesh.geometry().dim()
n = FacetNormal(mesh)
I = Identity(nsd)
h = CellDiameter(mesh)

# import subdomains
log("Importing subdomains.")
mvc = MeshValueCollection('size_t', mesh, nsd)
with Timer("HV 01: import subdomains"):
    with XDMFFile(comm, FILEPATH_IN_SUBDOMAINS) as f:
        f.read(mvc, MARKER)
subdomains = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# import boundaries
log("Importing boundaries.")
mvc = MeshValueCollection('size_t', mesh, nsd-1)
with Timer("HV 02: import boundaries"):
    with XDMFFile(comm, FILEPATH_IN_BOUNDARIES) as f:
        f.read(mvc, MARKER)
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# initialize mesh connectivities
log("Building mesh connectivities.")
with Timer("HV 03: build mesh connectivities"):
    mesh.init(nsd, nsd-1)
    mesh.init(nsd-3, nsd-1)

# initialize and fill a facet function for the solid region, leaving off 
# anything touching the interface facets
log("Building solid-region interior MeshFunction.")
solid_interior_facets = MeshFunction("size_t", mesh, nsd-1)
solid_interior_facets.set_all(FLAG['solid'])
with Timer("HV 04: build interior solid markers"):
    for facet in facets(mesh):
        marker = False
        for cell in cells(facet):
            marker = marker or subdomains[cell]==FLAG["fluid"]
        if (not marker):
            for vertex in vertices(facet):
                vertex_facets = vertex.entities(nsd-1)
                for vertex_facet in vertex_facets:
                    marker = marker or \
                        (boundaries[vertex_facet]==FLAG["interface"])
        if marker:
            solid_interior_facets[facet] = FLAG["none"]

# Set up integration measures, with flags to integrate over
# subsets of the domain.
log("Creating integration measures per mesh/subdomains/boundaries.")
dy = dx(metadata={'quadrature_degree': 2*BKG_POLYNOMIAL_DEGREE},
        domain=mesh, subdomain_data=subdomains)
ds = ds(metadata={'quadrature_degree': 2*BKG_POLYNOMIAL_DEGREE},
        domain=mesh, subdomain_data=boundaries)


###############################################################
#### Import pressure waveform #################################
###############################################################

log("Importing pressure waveforms from csv files.")
time_vals = []      # times (T's) in seconds
p_in_vals = []      # pressures (P's) in dyne/cm^2
p_out_vals = []     # pressures (P's) in dyne/cm^2
with open(FILEPATH_IN_PRESSURES,newline='') as f:
    reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        # times are in seconds, pressures in mmHg in file
        time_vals.append(float(row[0]))
        p_in_vals.append(PRESSURE_UNIT_CONVERSION*float(row[1]))
        p_out_vals.append(PRESSURE_UNIT_CONVERSION*float(row[2]))


###############################################################
#### Import valve geometry ####################################
###############################################################

if not os.path.isfile(FILEPATH_IN_SHELL_PREFIX+"1"+FILEPATH_IN_SHELL_SUFFIX):
    error("Missing data files for valve geometry.")

# Load a control mesh from several files in a legacy ASCII format.
log("Importing smesh files as control mesh.")
controlMesh = LegacyMultipatchControlMesh(FILEPATH_IN_SHELL_PREFIX,
                                          NUM_PATCHES,
                                          FILEPATH_IN_SHELL_SUFFIX)

# Every processor has a full copy of the shell structure, on its
# MPI_SELF communicator.
log("Creating spline generator from control mesh.")
with Timer("HV 05:create spline generator"):
    splineGenerator = EqualOrderSpline(selfcomm,nsd,controlMesh)

# set clamped bcs on edges of leaflets
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
for patch in LEAFLET_PATCHES:
    for direction,side in LEAFLET_BC_EDGES:
        sideDofs = scalarSpline.getPatchSideDofs(patch-1,direction,side,
                    nLayers=LEAFLET_BC_LAYERS)
        for i in range(0,nsd):
            splineGenerator.addZeroDofs(i,sideDofs)

# fix stent in space and time
if not STENTLESS:
    for patch in STENT_PATCHES:
        start_dof = scalarSpline.doffsets[patch-1]
        end_dof = start_dof + scalarSpline.splines[patch-1].getNcp()
        patch_dofs = list(range(start_dof,end_dof))
        for field in range(0,nsd):
            splineGenerator.addZeroDofs(field, patch_dofs)

# Generate the extracted representation of the spline.
log("Generating extracted spline.")
spline = ExtractedSpline(splineGenerator,SPLINE_QUAD_DEG)

# Define contact context:
log("Specify shell self-contact rules.")

def phiPrime(r):
    if (r>CONTACT["R_MAX"]):
        return 0.0
    elif (r>(1-CONTACT['S'])*CONTACT["R_MAX"]):
        return -CONTACT['K']/(2*CONTACT["R_MAX"]*CONTACT['S'])*\
               (CONTACT["R_MAX"]-r)**2
    else:
        return -CONTACT['K']*(CONTACT["R_MAX"]*(1-CONTACT['S']/2)-r)
def phiDoublePrime(r):
    if (r>CONTACT["R_MAX"]):
        return 0.0
    elif (r>(1-CONTACT['S'])*CONTACT["R_MAX"]):
        return CONTACT['K']/(CONTACT["R_MAX"]*CONTACT['S'])*\
               (CONTACT["R_MAX"]-r)
    else:
        return CONTACT['K']

contactContext_sh = ShellContactContext(spline,CONTACT["R_SELF"],
                                        CONTACT["R_MAX"],
                                        phiPrime,phiDoublePrime)

###############################################################
#### Compute stent-solid intersections ########################
###############################################################

log("Computing intersection between stent and wall.")

# initiate bounding box tree for background mesh
log("  Generating bounding box tree for background mesh.")
bbt = mesh.bounding_box_tree()

# fetch dof coordinates on shell from contact context
points_list = contactContext_sh.nodeXs

# initiate intersection markers on cells
intersections = MeshFunction('bool',mesh,nsd)
intersections.set_all(False)

# compute cell intersections
log("  Computing collisions between fluid-solid mesh and shell.")
for pts in points_list:
    for i in range(np.size(pts,axis=0)):
        p = Point(pts[i,:])
        collisions = bbt.compute_entity_collisions(p)
        intersections_arr = intersections.array()
        for cell in collisions:
            intersections_arr[cell] = True

# transfer cell interseactions to a facet function
# applies marker to only the solid-stent intersections
log("  Computing intersected facets of solid cells.")
stent_intersection = MeshFunction('size_t',mesh,nsd-1)
stent_intersection.set_all(FLAG["none"])
for cell in cells(mesh):
    if (intersections[cell] is True) and \
        (subdomains[cell]==FLAG["solid"]):
        for facet in facets(cell):
            stent_intersection[facet] = FLAG["stent_intersection"]

# write or load from to filesystem if requested
if WRITE_SHELL_INTERSECTIONS:
    log("  Saving solid-stent intersections to file.")
    stent_intersection.rename(MARKER,MARKER)
    XDMFFile(comm,FILEPATH_SHELL_INTERSECTIONS).write(stent_intersection)
if USE_SHELL_INTERSECTIONS:
    log("  Loading solid-stent intersections from file.")
    mvc = MeshValueCollection('size_t', mesh, nsd-1)
    with XDMFFile(comm, FILEPATH_SHELL_INTERSECTIONS) as f:
        f.read(mvc, MARKER)
    stent_intersection = cpp.mesh.MeshFunctionSizet(mesh, mvc)


###############################################################
#### Display Timings for Imports ##############################
###############################################################
if LOG_TIMINGS:
    log("\nTimings:")
    list_timings(TimingClear.clear,[TimingType.wall])
    log("")



###############################################################
#### Elements and Function Spaces #############################
###############################################################

# Define function spaces (equal order interpolation):
cell = mesh.ufl_cell()
FSM_POLYNOMIAL_DEGREE = 1
Ve = VectorElement("Lagrange", cell, FSM_POLYNOMIAL_DEGREE)
Qe = FiniteElement("Lagrange", cell, FSM_POLYNOMIAL_DEGREE)
VQe = MixedElement((Ve,Qe))
# Mixed function space for velocity and pressure:
V_fs = FunctionSpace(mesh,VQe)
V_fs_scalar = FunctionSpace(mesh,"Lagrange",FSM_POLYNOMIAL_DEGREE)
# Function space for mesh displacement field, 
# which will be solved for separately in a 
# quasi-direct scheme:
V_m = FunctionSpace(mesh,Ve)

# report size of problem
log("Problem size:")
log("  Fluid-Solid DOFs: " + str(int(V_fs.dim())))
log("  Mesh motion DOFs: " + str(int(V_m.dim())))
log("  Shell FEM DOFs: " + str(spline.M.size(0)))
log("  Shell IGA DOFs: " + str(spline.M.size(1)))
log("  Shell Lagrange nodes: " + str(spline.V_control.\
    tabulate_dof_coordinates().shape[0]))


###############################################################
#### Time Integration Setup ###################################
###############################################################

# Mesh motion functions
duhat = TestFunction(V_m)
uhat = Function(V_m)
uhat_old = Function(V_m)
uhatdot_old = Function(V_m)
uhatdotdot_old = Function(V_m)

# Mesh motion time integrator
timeInt_m = GeneralizedAlphaIntegrator(RHO_INFINITY, DT, uhat, 
                                       (uhat_old, uhatdot_old, 
                                        uhatdotdot_old), 
                                       t=t, useFirstOrderAlphaM=True)
uhat_alpha = timeInt_m.x_alpha()
vhat_alpha = timeInt_m.xdot_alpha()

# Fluid--solid time functions
log("Initiating fluid-solid functions and time integrator")
(dv, dp) = TestFunctions(V_fs)
vp = Function(V_fs)
v, p = split(vp)
vp_old = Function(V_fs)
vpdot_old = Function(V_fs)


# get the u-part of a function
def uPart(up):
    return as_vector([up[0], up[1], up[2]])

# Fluid--solid time integrator
timeInt_fs = GeneralizedAlphaIntegrator(RHO_INFINITY, DT, vp, 
                                        (vp_old, vpdot_old), t=t)
vp_alpha = timeInt_fs.x_alpha()
v_alpha = uPart(vp_alpha)
p_alpha = vp_alpha[3]
dvp_dr = timeInt_fs.xdot()
dv_dr = uPart(dvp_dr)
dv_ds = dv_dr # Only valid in solid


# Displacement field used in the solid formulation
u_s = uhat_old + DT*v_alpha
u_s_alpha = x_alpha(timeInt_fs.ALPHA_F,u_s,uhat_old)

# This is used to match u_s to set bc on the mesh motion subproblem
u_func = Function(V_m)



###############################################################
#### Boundary Conditions ######################################
###############################################################

# helper zero vector
zero = Constant(nsd*(0,))

# BCs for the mesh motion subproblem:
bcs_m = [
    DirichletBC(V_m, u_func, solid_interior_facets, FLAG["solid"]),
    DirichletBC(V_m, u_func, boundaries, FLAG["interface"]),
    DirichletBC(V_m, zero, stent_intersection, FLAG["stent_intersection"]),
    DirichletBC(V_m.sub(2), Constant(0), boundaries, FLAG["solid_inflow"]),
    DirichletBC(V_m.sub(2), Constant(0), boundaries, FLAG["solid_outflow"]),
    DirichletBC(V_m.sub(2), Constant(0), boundaries, FLAG["fluid_inflow"]),
    DirichletBC(V_m.sub(2), Constant(0), boundaries, FLAG["fluid_outflow"]),
    ]

# Fluid-solid BCs
bcs_fs = [
    DirichletBC(V_fs.sub(0).sub(2), Constant(0), boundaries, 
                FLAG["solid_inflow"]),
    DirichletBC(V_fs.sub(0).sub(2), Constant(0), boundaries, 
                FLAG["solid_outflow"]),
    DirichletBC(V_fs.sub(0), zero, stent_intersection, 
                FLAG["stent_intersection"]),
    DirichletBC(V_fs.sub(1), Constant(0), solid_interior_facets, 
                FLAG["solid"]),
    ]

if FREEZE_SOLID:
    bcs_m.append(DirichletBC(V_m, zero, solid_interior_facets, FLAG["solid"]))
    bcs_m.append(DirichletBC(V_m, zero, boundaries, FLAG["interface"]))
    bcs_fs.append(DirichletBC(V_fs.sub(0), zero, solid_interior_facets,
                              FLAG["solid"]))
    bcs_fs.append(DirichletBC(V_fs.sub(0), zero, boundaries,
                              FLAG["solid_outflow"]))
    bcs_fs.append(DirichletBC(V_fs.sub(0), zero, boundaries,
                              FLAG["solid_inflow"]))
    bcs_fs.append(DirichletBC(V_fs.sub(0), zero, boundaries, 
                              FLAG["interface"]))



###############################################################
#### Mesh Motion Subproblem ###################################
###############################################################
mesh_model = sm.JacobianStiffening(power=Constant(3))
res_m = mesh_model.interiorResidual(uhat_alpha,duhat,dx=dy)
Dres_m = derivative(res_m, uhat)

###############################################################
#### Solid Domain Subproblem ##################################
###############################################################

dX = dy(FLAG["solid"])
solid_kappa = sm.bulkModulus(SOLID["E"],SOLID["nu"])
solid_mu = sm.shearModulus(SOLID["E"],SOLID["nu"])
solid_model = sm.NeoHookean(rho=SOLID['rho'], kappa=solid_kappa, mu=solid_mu)
res_s = solid_model.interiorResidual(u_s_alpha,dv,dx=dX)
res_s += solid_model.accelerationResidual(dv_ds,dv,dx=dX)
res_s += solid_model.massDampingResidual(v_alpha,SOLID["c"],dv,dx=dX)

###############################################################
#### Fluid Subproblem #########################################
###############################################################

p_in = Constant(0)
p_out = Constant(0) 
h_in = -p_in*n
h_out = -p_out*n
Q = Constant(0)
flowrateForm = inner(v_alpha,n)*ds(FLAG["fluid_outflow"])

cutFunc = Function(V_fs_scalar)

res_f = vrmt.interiorResidual(v_alpha,p_alpha,dv,dp,
                     FLUID["rho"],FLUID["mu"],mesh,
                     uhat=uhat_alpha,
                     vhat=vhat_alpha,
                     v_t=dv_dr,
                     Dt=DT,
                     f=None,
                     C_I=FLUID_NUMERICAL["C_i"],
                     C_t=FLUID_NUMERICAL["C_t"],
                     stabScale=cfsh.stabScale(cutFunc,
                                              FLUID_NUMERICAL["stab_eps"]),
                     dy=dy(FLAG["fluid"]))

res_f += vrmt.stableNeumannBC(h_in,FLUID["rho"],v_alpha,dv,mesh,
                                uhat=uhat_alpha,
                                vhat=vhat_alpha,
                                ds=ds(FLAG["fluid_inflow"]),
                                gamma=FLUID_NUMERICAL["gamma"])
res_f += vrmt.stableNeumannBC(h_out,FLUID["rho"],v_alpha,dv,mesh,
                                uhat=uhat_alpha,
                                vhat=vhat_alpha,
                                ds=ds(FLAG["fluid_outflow"]),
                                gamma=FLUID_NUMERICAL["gamma"])
res_f += vrmt.stableNeumannBC(-Q*FLUID_NUMERICAL["R"]*n,FLUID["rho"],
                                v_alpha,dv,mesh,
                                uhat=uhat_alpha,
                                vhat=vhat_alpha,
                                ds=ds(FLAG["fluid_outflow"]),
                                gamma=FLUID_NUMERICAL["gamma"])

###############################################################
#### Fluid-Solid Residual #####################################
###############################################################
res_fs = res_f + res_s
Dres_fs = derivative(res_fs, vp)

###############################################################
#### Shell Subproblem #########################################
###############################################################

# Set up shell structure problem using ShNAPr:
y_hom = Function(spline.V)
y_old_hom = Function(spline.V)
ydot_old_hom = Function(spline.V)
yddot_old_hom = Function(spline.V)
timeInt_sh = GeneralizedAlphaIntegrator(RHO_INFINITY,DT,y_hom,
                                        (y_old_hom,ydot_old_hom,yddot_old_hom),
                                        t=t,useFirstOrderAlphaM=True)
y_alpha_hom = timeInt_sh.x_alpha()
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)

X = spline.F
x = X + spline.rationalize(y_alpha_hom)

# Return a 3D elastic strain energy density, given E in Cartesian coordinates.
def psi_el(E):

    # isotropic Lee-Sacks material with numerical cutoff
    if LEAFLET['material']=='isotropic-lee-sacks':
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*LEAFLET['c0']*(I1 - 3.0) + \
                  0.5*LEAFLET['c1']*(exp(LEAFLET['c2']*(I1-3)**2)-1)

    # linear Saint-Venant Kirchhoff material
    elif LEAFLET['material']=='SVK':
        return 0.5*LEAFLET['c0']*(tr(E)**2) + LEAFLET['c1']*(E**2)

    # anisotropic Lee-Sacks material
    if LEAFLET['material']=='anisotropic-lee-sacks':
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        _,_,A2,_,_,_ = surfaceGeometry(spline,X)
        Z = Constant((0,0,1))
        m_global = unit(cross(Z,-A2))
        a0,a1,a2,_,_,_ = surfaceGeometry(spline,x)
        e0,e1 = orthonormalize2D(a0,a1)
        e2 = a2
        R = as_matrix(((e0[0],e1[0],e2[0]),
                       (e0[1],e1[1],e2[1]),
                       (e0[2],e1[2],e2[2])))
        m_local = R*m_global
        I4 = inner(m_local,C*m_local)
        return 0.5*LEAFLET['c0']*(I1 - 3.0) + \
               0.5*LEAFLET['c1']*( \
                LEAFLET['w']*exp(LEAFLET['c2']*(I1-3)**2) + \
                (1-LEAFLET['w'])*exp(LEAFLET['c3']*(I4-1)**2) - 1)

    # Neo-Hookean material
    elif LEAFLET['material']=='neo-hookean':
        C = 2.0*E + Identity(3)
        I1 = tr(C)
        return 0.5*LEAFLET['c0']*(I1-3.0)

    # throw an error when the material type is not implemented
    else:
        error("Requested material, " +LEAFLET['material'] 
                     + ", not implemented.")
        return None

# Obtain a through-thickness integration measure:
N_QUAD_PTS = 4
dxi2 = throughThicknessMeasure(N_QUAD_PTS,LEAFLET["h"])

# Potential energy density, including Lagrange multiplier term for
# incompressibility:
psi = incompressiblePotentialKL(spline,X,x,psi_el)

# Total internal energy:
Wint = psi*dxi2*spline.dx

yddot = spline.rationalize(timeInt_sh.xddot_alpha())
dWmass = LEAFLET["rho"]*LEAFLET["h"]*inner(yddot,z)*spline.dx
dWint = (1.0/timeInt_sh.ALPHA_F)*derivative(Wint,y_hom,z_hom)
res_sh = dWint + dWmass



###############################################################
#### Nonlinear Solvers Setup ##################################
###############################################################

# Linear solver settings for the fluid: The nonlinear solver typically
# converges quite well, even if the fluid linear solver is not converging.
# This is the main "trick" needed to make scaling of SUPG/LSIC parameters
# for mass conservation tractable in 3D problems.  
fluidLinearSolver = PETScKrylovSolver("gmres","jacobi")
fluidLinearSolver.ksp().setGMRESRestart(FLUID_NUMERICAL["KSP"]["max_iters"])
fluidLinearSolver.parameters['maximum_iterations'] = \
                            FLUID_NUMERICAL["KSP"]["max_iters"]
fluidLinearSolver.parameters['error_on_nonconvergence'] = False
fluidLinearSolver.parameters['monitor_convergence'] = \
                            FLUID_NUMERICAL["KSP"]["log"]
fluidLinearSolver.parameters['report'] = FLUID_NUMERICAL["KSP"]["log"]
fluidLinearSolver.parameters['relative_tolerance'] = \
                            FLUID_NUMERICAL["KSP"]["tol"]
fluidLinearSolver.set_norm_type(PETScKrylovSolver.norm_type.unpreconditioned)

# create mesh problem
meshProblem = cfsh.SolvedMeshMotion(timeInt_m,res_m,bcs=bcs_m,Dres=Dres_m,
                                    u_s=u_s,bc_func=u_func)

fsiProblem = cfsh.CouDALFISh(mesh, res_fs, timeInt_fs,
                              spline, res_sh, timeInt_sh,
                              meshProblem=meshProblem,
                              penalty=DAL_PENALTY, r=DAL_R,
                              blockItTol=BLOCK_REL_TOL,
                              blockIts=BLOCK_MAX_ITERS,
                              blockNoErr=BLOCK_NO_ERROR,
                              bcs_f=bcs_fs, Q=Q, flowrateForm=flowrateForm, 
                              contactContext_sh=contactContext_sh,
                              fluidLinearSolver=fluidLinearSolver,
                              cutFunc=cutFunc)



###############################################################
#### Visualization Files Setup ################################
###############################################################

# assign a quadrature degree that is exact but that isn't overkill
prms = {'quadrature_degree':2*spline.p_control}

# shell visualization
parametricCoords3D = as_vector([spline.parametricCoordinates()[0],
                                spline.parametricCoordinates()[1],
                                Constant(0)])
relative_spatial_coords = spline.spatialCoordinates()-parametricCoords3D
coordsRef = project(relative_spatial_coords,spline.V,
                    form_compiler_parameters=prms)
coordsRef.rename("coordsRef","coordsRef")


# further shell visualization
log("Build XDMF file to store shell displacements.")
outfile_sh = XDMFFile(selfcomm,FILEPATH_OUT_SHELL)
outfile_sh.parameters["flush_output"] = True
outfile_sh.parameters["functions_share_mesh"] = True
outfile_sh.parameters["rewrite_function_mesh"] = False

# fluid-solid-mesh visualization
log("Build XDMF file to store v, p, & uhat.")
outfile_fsm = XDMFFile(comm,FILEPATH_OUT_FSM)
outfile_fsm.parameters["flush_output"] = True
outfile_fsm.parameters["functions_share_mesh"] = True
outfile_fsm.parameters["rewrite_function_mesh"] = False

# quantity of interest data
if (WRITE_FLUID_STATS and rank==0):
    if not os.path.exists(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    with open(FILEPATH_OUT_QOI_DATA, 'w') as csvfile:
        writer = csv.writer(csvfile)
        row = []
        row.append("time step")
        row.append("time (s)")
        row.append("maximum CFL number")
        row.append("volumetric flowrate in (mL/s)")
        row.append("volumetric flowrate out (mL/s)")
        row.append("actual pressure in (mmHg)")
        row.append("actual pressure out (mmHg)")
        row.append("BC pressure in (mmHg)")
        row.append("BC pressure out (mmHg)")
        writer.writerow(row)
comm.barrier()




###############################################################
#### Display Current Timings ##################################
###############################################################
if LOG_TIMINGS:
    log("\nTimings:")
    list_timings(TimingClear.clear,[TimingType.wall])
    log("")
    log("Global time elapsed: " + str(global_timer.elapsed()[0]))



###############################################################
#### Time-Stepping Loop #######################################
###############################################################

# Initial conditions if restarting
if restarting:
    log("Loading restarts.")
    fsiProblem.readRestarts(RESTARTS_FOLDER, startStep)

# Time stepping loop:
for timeStep in range(startStep, NUM_TIME_STEPS):

    # keep track of times from previous step and current step
    t_old = t
    t = timeInt_fs.t
    alpha = float(timeInt_fs.ALPHA_F)
    t_alpha = alpha*t + (1.0-alpha)*t_old
    log((f"\n+{76*'='}+\n|{22*' '}Time step {timeStep:6d}, t = "
         f"{t:8.5f}{24*' '}|\n+{76*'='}+"))
    
    # set up step timer
    step_timer = Timer("HV 06: full time step")
    step_timer.start()

    # set the fluid inlet/outlet pressures
    p_in.assign(np.interp(t_alpha,time_vals,p_in_vals))
    p_out.assign(np.interp(t_alpha,time_vals,p_out_vals))

    # step forward in time in FSI problem
    (last_res_fs,last_res_sh) = fsiProblem.takeStep()

    # write the restart files
    if (WRITE_RESTARTS and timeStep%WRITE_RESTARTS_SKIP==0):
        log("Writing restarts.")
        fsiProblem.writeRestarts(RESTARTS_FOLDER, timeStep)
        if rank==0:
            stepFile = open(os.path.join(RESTARTS_FOLDER,"step.dat"), "w")
            stepFile.write(str(timeStep) + " " + str(timeInt_fs.t))
            stepFile.close()
        comm.barrier()

    # write the visualization files
    if WRITE_VIS:
        # save shell to XDMF File
        if timeStep%WRITE_VIS_SKIP_SHELL==0:
            log("Writing shell visualization files.")

            # rename residual
            last_res_sh.rename("res","res")

            # shell displacements
            y = spline.rationalize(y_hom)   
            d = project(y,spline.V,form_compiler_parameters=prms)
            d.rename("d","d")

            # shell velocity
            ydot = spline.rationalize(timeInt_sh.xdot())
            ddot = project(ydot,spline.V)
            ddot.rename("ddot","ddot") 

            # shell normals
            A0,A1,A2,deriv_A2,A,B = surfaceGeometry(spline,X)
            a0,a1,a2,deriv_a2,a,b = surfaceGeometry(spline,X+y)
            normal = project(a2,spline.V,form_compiler_parameters=prms)
            normal.rename("n","n")

            # shell MIPE
            xi2 = Constant(0)
            G = metricKL(A,B,xi2)
            g = metricKL(a,b,xi2)
            E_flat = 0.5*(g - G)
            G0,G1 = curvilinearBasisKL(A0,A1,deriv_A2,xi2)
            E = covariantRank2TensorToCartesian2D(E_flat,G,G0,G1)
            term = sqrt((0.5*tr(E))**2 - det(E))
            eig1 = 0.5*tr(E) + term
            eig2 = 0.5*tr(E) - term
            MIPE = ufl.Max(eig1,eig2)
            
            xi2.assign(-LEAFLET["h"]/2)
            mipe_neg = project(MIPE,spline.V_control,
                               form_compiler_parameters=prms)
            mipe_neg.rename("MIPE_neg_xi2","MIPE_neg_xi2")

            xi2.assign(LEAFLET["h"]/2)
            mipe_pos = project(MIPE,spline.V_control,
                               form_compiler_parameters=prms)
            mipe_pos.rename("MIPE_pos_xi2","MIPE_pos_xi2")

            # write to file
            if (rank==0):
                outfile_sh.write(coordsRef,float(t))
                outfile_sh.write(normal,float(t))
                outfile_sh.write(d,float(t))
                outfile_sh.write(ddot,float(t))
                outfile_sh.write(mipe_neg,float(t))
                outfile_sh.write(mipe_pos,float(t))
                outfile_sh.write(last_res_sh,float(t))
            comm.barrier()

        # Fluid--solid--mesh motion solution:
        if timeStep%WRITE_VIS_SKIP_FLUID_SOLID==0:
            log("Writing fluid-solid-mesh visualiation files.")

            # rename residual
            last_res_fs.rename("res_fs","res_fs")

            # split out solutions
            (v, p) = vp.split()
            v.rename("u","u")
            p.rename("p","p")
            uhat.rename("uhat","uhat")

            # save fluid-solid-mesh to file
            outfile_fsm.write(v,float(t))
            outfile_fsm.write(p,float(t))
            outfile_fsm.write(uhat,float(t))
            outfile_fsm.write(last_res_fs,float(t))

    if WRITE_FLUID_STATS and timeStep%WRITE_FLUID_STATS_SKIP==0:
        log("Writing fluid flow statistics.")
        
        # extract flow fields
        (v,p) = vp.split()

        # move mesh to current configuration
        ALE.move(mesh,uhat)

        # report maximum CFL number
        DG = FunctionSpace(mesh,"DG",0)
        cfl_number = project(sqrt(inner(v,v))*DT/h, DG,
            solver_type='gmres', preconditioner_type='jacobi')
        max_cfl = MPI.max(comm,cfl_number.vector().max())

        # report current flowrate at inlet and outlet 
        Q_in = assemble(inner(v,n)*ds(FLAG["fluid_inflow"]))
        Q_out = assemble(inner(v,n)*ds(FLAG["fluid_outflow"]))

        # report average pressure at inflow and outflow
        x = SpatialCoordinate(mesh)
        force_in = assemble(inner(vrmt.sigma(v,p,x,FLUID["mu"])*n,n)*\
                            ds(FLAG["fluid_inflow"]))
        area_in = assemble(Constant(1)*ds(FLAG["fluid_inflow"]))
        force_out = assemble(inner(vrmt.sigma(v,p,x,FLUID["mu"])*n,n)*\
                            ds(FLAG["fluid_outflow"]))
        area_out = assemble(Constant(1)*ds(FLAG["fluid_outflow"]))

        # save reported values in a csv file
        if (rank==0):
            with open(FILEPATH_OUT_QOI_DATA, 'a') as csvfile:
                writer = csv.writer(csvfile)
                row = []
                row.append(timeStep)
                row.append(t)
                row.append(max_cfl)
                row.append(Q_in)
                row.append(Q_out)
                row.append(abs(force_in/area_in/PRESSURE_UNIT_CONVERSION))
                row.append(abs(force_out/area_out/PRESSURE_UNIT_CONVERSION))
                row.append(float(p_in)/PRESSURE_UNIT_CONVERSION)
                row.append(float(p_out)/PRESSURE_UNIT_CONVERSION)
                writer.writerow(row)
        comm.barrier()

        # move mesh back to reference configuration
        uhat_neg = Function(V_fs)
        uhat_neg.assign(-uhat)
        ALE.move(mesh, uhat_neg)
        
    # display timings
    step_timer.stop()
    if LOG_TIMINGS:
        log("Timings:")
        list_timings(TimingClear.clear,[TimingType.wall])


log("Time stepping loop completed! Script finished, bye for now!")
