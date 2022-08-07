"""
This demo simulates fluid--structure interaction in a simplified model of an 
aortic heart valve.  An archive with data files defining the valve geometry 
may be downloaded from the following link:

  https://www.dropbox.com/s/ot5i568dw40h75c/leaflet-geometries.tgz?dl=1

Extract this archive into the working directory in which you want to run this
demo, using the command ``tar vzf leaflet-geometries.tgz``.  

Some notes:

- Numbers are all in the centimeter--gram--second (CGS) system.

- Systolic pressure is applied impulsively to quiescent initial conditions,
  which is why the first time step takes many iterations to converge.

- The fluid may be computed in parallel, by running this script with 
  ``mpirun -n <number of tasks> python3 <path to script> [options]``.  
  However, the valve subproblem is still solved in serial, which will 
  ultimately limit scalability.

- For simplicity and ease of distribution, the fluid mesh for this demo is 
  generated using constructive solid geometry operations, with the mshr 
  component of FEniCS, which uses CGAL to create quasi-uniform meshes.
  However, it is generally recommended to use a more fully-featured mesh 
  generator (e.g., Gmsh) for problems of this scale and complexity.

- The short duration of systole corresponds to exercise conditions, and was
  selected to reduce runtime of this demo.

- The spacing between valve leaflets has been artificially increased somewhat
  to better accommodate the nonlocal contact method implemented in ShNAPr.

- To be able to run this demo on slow or small-memory machines, manually 
  reduce the resolution using the command line argument 
  ``--resolution=<integer>``, where smaller integer values correspond to 
  lower resolutions.  (Drastic reductions may of course lead to strange or 
  unphysical results.)  The default value produces correct overall 
  qualitative behavior and allows the valve to open and close in an 
  overnight run on a typical modern desktop computer with several cores.

- Higher-than-default resolutions may also require higher-than-default 
  values for the maximum number of Krylov iterations for the fluid solver,
  which can be set via the command line argument ``--maxKSPIt=<integer>``.  

- Writing ParaView files while running is handy for one-shot runs, although 
  it doesn't play nicely with restarting, so, for larger/longer runs, 
  it is better to turn this off with the option ``--noViz`` and load and 
  visualize each set of restart files in a separate postprocessing step.
"""

from CouDALFISh import *
from tIGAr.BSplines import *
from VarMINT import *
from ShNAPr.SVK import *
from os import path

# Check whether the user downloaded and extracted the data files to the
# working directory; assume that if one is present all are, since including
# just the first would be an unlikely mistake.
fnamePrefix = "smesh."
fnameSuffix = ".dat"
fnameCheck = fnamePrefix+"1"+fnameSuffix
import os.path
if(not os.path.isfile(fnameCheck)):
    if(mpirank==0):
        print("ERROR: Missing data files for valve geometry. "
              +"Please refer to the docstring at the top of this script.")
    exit()

# Suppress excessive output in parallel:
parameters["std_out_all_processes"] = False

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--resolution',dest='resolution',default=70,
                    help="Resolution of fluid mesh.")
parser.add_argument('--maxKSPIt',dest='maxKSPIt',default=300,
                    help="Maximum number of fluid Krylov iterations.")
parser.add_argument('--fluidKSPrtol',dest='fluidKSPrtol',default=1e-2,
                    help="Relative tolerance for fluid Krylov solver.")
parser.add_argument('--DAL_penalty',dest='DAL_penalty',default=5e2,
                    help="Value of penalty for DAL method.")
parser.add_argument('--DAL_r',dest='DAL_r',default=1e-5,
                    help="Value of regularization parameter for DAL method.")
parser.add_argument('--blockItTol',dest='blockItTol',default=1e-2,
                    help="Relative tolerance for block iteration.")
parser.add_argument('--stabEps',dest='stabEps',default=1e-3,
                    help="Scaling of SUPG constant near immersed shell.")
parser.add_argument('--r_max',dest='r_max',default=0.035,
                    help="Range of nonlocal contact forces.")
parser.add_argument('--k_contact',dest='k_contact',default=1e11,
                    help="Stiffness of nonlocal contact forces.")
parser.add_argument('--R_self',dest='R_self',default=0.045,
                    help="Range of ignored self-contact in reference config.")
parser.add_argument('--Dt',dest='Dt',default=0.5e-4,
                    help="Time step.")
parser.add_argument('--rho_infty',dest='rho_infty',default=0.0,
                    help="Spectral radius of time integrator at Dt=infty.")
parser.add_argument('--outSkip',dest='outSkip',default=10,
                    help="Time steps between writing visualization files.")
parser.add_argument('--Nsteps',dest='Nsteps',default=10000,
                    help="Total number of time steps.")
parser.add_argument('--outputFileName',dest='outputFileName',
                    default="flow-rate",
                    help="File to write flow rate data to.")
parser.add_argument('--restartPath',dest='restartPath',
                    default="restarts",
                    help="Name of directory to write restart files to.")
parser.add_argument('--noViz',dest='noViz',action='store_true',
                    help='Do not write ParaView files.')

args = parser.parse_args()
resolution = int(args.resolution)
maxKSPIt = int(args.maxKSPIt)
fluidKSPrtol = float(args.fluidKSPrtol)
DAL_penalty = float(args.DAL_penalty)
DAL_r = float(args.DAL_r)
r_max = float(args.r_max)
k_contact = float(args.k_contact)
R_self = float(args.R_self)
Dt = Constant(float(args.Dt))
rho_infty = Constant(float(args.rho_infty))
outSkip = int(args.outSkip)
Nsteps = int(args.Nsteps)
stabEps = Constant(float(args.stabEps))
blockItTol = float(args.blockItTol)
outputFileName = str(args.outputFileName)
restartPath = str(args.restartPath)
viz = (not bool(args.noViz))

# Check if restarting:
restarting = path.exists("step.dat")
if(restarting):
    stepFile = open("step.dat","r")
    fs = stepFile.read()
    stepFile.close()
    tokens = fs.split()
    startStep = int(tokens[0])
    t = float(tokens[1])
else:
    startStep = 0
    t = 0.0

####### IGA foreground mesh and function space setup #######

if(mpirank==0):
    print("Generating extraction data...")
    
# Load a control mesh from several files in a legacy ASCII format.
controlMesh = LegacyMultipatchControlMesh(fnamePrefix,3,fnameSuffix)

# Every processor has a full copy of the shell structure, on its
# MPI_SELF communicator.
splineGenerator = EqualOrderSpline(selfcomm,d,controlMesh)

N_LAYERS = 2
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
for patch in range(0,3):
    for side in range(0,2):
        for direction in range(0,2):
            if(not (direction==1 and side==0)):
                sideDofs = scalarSpline\
                           .getPatchSideDofs(patch,direction,side,
                                             nLayers=N_LAYERS)
                for i in range(0,d):
                    splineGenerator.addZeroDofs(i,sideDofs)
                    
if(mpirank==0):
    print("Creating extracted spline...")

# Quadrature degree for the analysis:
QUAD_DEG = 4

# Generate the extracted representation of the spline.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

####### FEM background mesh and function space setup #######

from mshr import *
CYLINDER_RAD = 1.1
BOTTOM = -0.5
TOP = 2.0
tube = Cylinder(Point(0,0,BOTTOM),
                Point(0,0,TOP),CYLINDER_RAD,CYLINDER_RAD)

SINUS_CENTER_FAC = 0.5
SINUS_RAD_FAC = 0.8
SINUS_Z_SHIFT = -0.2
sinusRad = SINUS_RAD_FAC*CYLINDER_RAD
sinusZ = sinusRad + SINUS_Z_SHIFT
for i in range(0,3):
    sinusTheta = math.pi/3.0 + i*2.0*math.pi/3.0
    sinusCenterRad = SINUS_CENTER_FAC*CYLINDER_RAD
    sinusCenter = Point(sinusCenterRad*math.cos(sinusTheta),
                        sinusCenterRad*math.sin(sinusTheta),sinusZ)
    tube += Sphere(sinusCenter,sinusRad)

# Must not re-generate mesh when restarting, because CGAL is not
# deterministic. 
if(restarting):
    mesh = Mesh()
    f = HDF5File(worldcomm,"mesh.h5","r")
    f.read(mesh,"/mesh",True)
    f.close()
else:
    mesh = generate_mesh(tube,resolution)
    f = HDF5File(worldcomm,"mesh.h5","w")
    f.write(mesh,"/mesh")
    f.close()

# Print mesh statistics:
Nel_f = mesh.num_entities_global(mesh.topology().dim())
Nvert_f = mesh.num_entities_global(0)
if(mpirank==0):
    print("======= Fluid mesh information =======")
    print("  Number of elements in fluid mesh: "+str(Nel_f))
    print("  Number of nodes in fluid mesh: "+str(Nvert_f))

####### Formulation #######

# Set up VMS fluid problem using VarMINT:
VE = VectorElement("Lagrange",mesh.ufl_cell(),1)
QE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
VQE = MixedElement([VE,QE])
V_f = equalOrderSpace(mesh)
Vscalar = FunctionSpace(mesh,"Lagrange",1)
up = Function(V_f)
up_old = Function(V_f)
updot_old = Function(V_f)
vq = TestFunction(V_f)
timeInt_f = GeneralizedAlphaIntegrator(rho_infty,Dt,up,(up_old,updot_old),t)

# Define traction boundary condition at inflow:
xSpatial = SpatialCoordinate(mesh)
PRESSURE = Expression("((t<0.1)? 2e4 : -1e5)",t=0.0,degree=1)
inflowChar = conditional(lt(xSpatial[2],BOTTOM+1e-3),1.0,Constant(0.0))
# It's actually prefereable here to use a characteristic function instead
# of marking the inflow facets, because the "zero traction" BC at the outflow
# still requires stabilization terms.  As such, we can kill two birds with
# on stone by defining a spatially-varying traction like this and applying
# it on the full boundary with VarMINT's stable Neumann BC formulation.
inflowTraction = as_vector((0.0,0.0,PRESSURE))*inflowChar

def uPart(up):
    return as_vector([up[0],up[1],up[2]])

quadDeg = 2
dx = dx(metadata={"quadrature_degree":quadDeg})
ds = ds(metadata={"quadrature_degree":quadDeg})
rho = Constant(1.0)
mu = Constant(3e-2)
up_alpha = timeInt_f.x_alpha()
u_alpha = uPart(up_alpha)
p = timeInt_f.x[3]
v,q = split(vq)
up_t = timeInt_f.xdot_alpha()
u_t = uPart(up_t)
cutFunc = Function(Vscalar)
res_f = interiorResidual(u_alpha,p,v,q,rho,mu,mesh,v_t=u_t,Dt=Dt,dy=dx,
                         stabScale=stabScale(cutFunc,stabEps))
n = FacetNormal(mesh)
res_f += stableNeumannBC(inflowTraction,rho,u_alpha,v,n,
                         ds=ds,gamma=Constant(1.0))
# The scaled radius criterion is somewhat sloppy, and is essentially to
# compensate for the fact that the Cylinder CSG primitive in mshr is faceted,
# so some vertices on the curved boundary of the cylinder will be at radii
# significantly (relative to machine precision) less than the nominal cylinder
# radius.
bcs_f = [DirichletBC(V_f.sub(0), Constant(d*(0.0,)),
                     (lambda x, on_boundary :
                      on_boundary and
                      math.sqrt(x[0]*x[0]+x[1]*x[1])>0.98*CYLINDER_RAD)),]

# Form to evaluate net inflow:
u = uPart(up)
netInflow = -inflowChar*dot(u,n)*ds

# Set up shell structure problem using ShNAPr:
y_hom = Function(spline.V)
y_old_hom = Function(spline.V)
ydot_old_hom = Function(spline.V)
yddot_old_hom = Function(spline.V)
timeInt_sh = GeneralizedAlphaIntegrator(rho_infty,Dt,y_hom,
                                        (y_old_hom,ydot_old_hom,yddot_old_hom),
                                        t=t,useFirstOrderAlphaM=True)
y_alpha_hom = timeInt_sh.x_alpha()
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)

E = Constant(1e7)
nu = Constant(0.4)
h_th = Constant(0.04)
rho0 = Constant(1.0)
X = spline.F
x = X + spline.rationalize(y_alpha_hom)
Wint = surfaceEnergyDensitySVK(spline,X,x,E,nu,h_th)*spline.dx
yddot = spline.rationalize(timeInt_sh.xddot_alpha())
res_sh = rho0*h_th*inner(yddot,z)*spline.dx \
         + (1.0/timeInt_sh.ALPHA_F)*derivative(Wint,y_hom,z_hom)

# Define contact context:
def phiPrime(r):
    if(r>r_max):
        return 0.0
    return -k_contact*(r_max-r)
def phiDoublePrime(r):
    if(r>r_max):
        return 0.0
    return k_contact
contactContext_sh = ShellContactContext(spline,R_self,r_max,
                                        phiPrime,phiDoublePrime)

# Linear solver settings for the fluid: The nonlinear solver typically
# converges quite well, even if the fluid linear solver is not converging.
# This is the main "trick" needed to make scaling of SUPG/LSIC parameters
# for mass conservation tractable in 3D problems.  
fluidLinearSolver = PETScKrylovSolver("gmres","jacobi")
fluidLinearSolver.parameters["error_on_nonconvergence"] = False
fluidLinearSolver.ksp().setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
fluidLinearSolver.ksp().setTolerances(rtol=fluidKSPrtol,max_it=maxKSPIt)
fluidLinearSolver.ksp().setGMRESRestart(maxKSPIt)

# Couple with CouDALFISh:
fsiProblem = CouDALFISh(mesh,res_f,timeInt_f,
                        spline,res_sh,timeInt_sh,
                        DAL_penalty,r=DAL_r,
                        bcs_f=bcs_f,
                        blockItTol=blockItTol,
                        contactContext_sh=contactContext_sh,
                        fluidLinearSolver=fluidLinearSolver,
                        cutFunc=cutFunc)

####### Execution #######

if(mpirank==0):
    # For x, y, and z components of displacement:
    d0File = File(selfcomm,"results/disp-x.pvd")
    d1File = File(selfcomm,"results/disp-y.pvd")
    d2File = File(selfcomm,"results/disp-z.pvd")

    # For x, y, and z components of initial configuration:
    F0File = File(selfcomm,"results/F-x.pvd")
    F1File = File(selfcomm,"results/F-y.pvd")
    F2File = File(selfcomm,"results/F-z.pvd")

    # For weights:
    F3File = File(selfcomm,"results/F-w.pvd")

# For fluid velocity
uFile = File("results/u.pvd")
pFile = File("results/p.pvd")

# Initial conditions:
if(restarting):
    fsiProblem.readRestarts(restartPath,startStep)

# Time stepping loop:
for timeStep in range(startStep,Nsteps):
    
    PRESSURE.t = timeInt_f.t-(1.0-float(timeInt_f.ALPHA_M))*float(Dt)

    if(mpirank==0):
        print("------- Time step "+str(timeStep+1)
              +" , t = "+str(timeInt_f.t)+" -------")

    # Output fields needed for restarting and visualization.
    if(timeStep % outSkip == 0):

        # Restart data:
        fsiProblem.writeRestarts(restartPath,timeStep)
        if(mpirank==0):
            stepFile = open("step.dat","w")
            stepFile.write(str(timeStep)+" "+str(timeInt_f.t-float(Dt)))
            stepFile.close()

        if(viz):
            # Structure:
            if(mpirank==0):
                (d0,d1,d2) = y_hom.split()
                d0.rename("d0","d0")
                d1.rename("d1","d1")
                d2.rename("d2","d2")
                d0File << d0
                d1File << d1
                d2File << d2
                # (Note that the components of spline.F are rational,
                # and cannot be directly outputted to ParaView files.)
                spline.cpFuncs[0].rename("F0","F0")
                spline.cpFuncs[1].rename("F1","F1")
                spline.cpFuncs[2].rename("F2","F2")
                spline.cpFuncs[3].rename("F3","F3")
                F0File << spline.cpFuncs[0]
                F1File << spline.cpFuncs[1]
                F2File << spline.cpFuncs[2]
                F3File << spline.cpFuncs[3]

            # Fluid:
            (uout,pout) = up.split()
            uout.rename("u","u")
            pout.rename("p","p")
            uFile << uout
            pFile << pout

    # Compute the time step for the coupled problem:
    fsiProblem.takeStep()

    # Write flow rate to a file
    flowRate = assemble(netInflow)
    if(mpirank==0):
        mode = "a"
        if(timeStep==0):
            mode = "w"
        outFile = open(outputFileName,mode)
        outFile.write(str(timeInt_f.t)+" "+str(flowRate)+"\n")
        outFile.close()
