"""
This demo solves the 2D valve benchmark from Section 4.7 of
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4274080/
The benchmark is derived from one introduced in prior works, as indicated by
the citations in the linked paper, although certain details of the setup vary
between references. The exact solution is unknown, and works using this problem
(or a similar one) for verification typically plot and qualitatively-compare
time histories of displacements at the free end of the top "leaflet" for
discretizations of differing resolutions.  Tip displacement is outputted to a
file whose name defaults to 'tip-dislacement' in the working directory, with
whitespace- separated columns containing time, ``$x_1$`` displacement, and
``$x_2$`` displacement. Although the geometry is relatively simple, it is given
in external files available in the following archive
  https://www.dropbox.com/s/aa4j1byiucv5pqo/2d-valve-geometry.tgz?dl=1
because, at time of writing, that is the only input format that supports
multi-patch splines.  (The use of a fixed mesh for the structure subproblem
will ultimately limit convergence, but discretization error is overwhelmingly
due to poor approximately of discontinuous pressure and velocity gradients in
the fluid subproblem, and we are not exploring convergence rigorously here,
using Sobolev norms, so it is good enough for the present purposes.)  

This demo by default uses TSFC, which can be installed via the commands found
in the Singularity recipe for ``tIGAr``.
"""

# Miscellaneous utilities:
from numpy import array
import sys


####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()

# ------- Parameter to change for convergence study -------
parser.add_argument('--Nel_f_vert',dest='Nel_f_vert',default=32,
                    help="Number of elements across the channel.")
# ---------------------------------------------------------

# (Defaults are reasonable for remaining parameters.)
parser.add_argument('--igaQuadDeg',dest='igaQuadDeg',default=4,
                    help="Quadrature degree for IGA shell subproblem.")
parser.add_argument('--femQuadDeg',dest='femQuadDeg',default=2,
                    help="Quadrature degree for FEM fluid subproblem.")
parser.add_argument('--T',dest='T',default=3.0,
                    help="Time interval.")
parser.add_argument('--stabEps',dest='stabEps',default=1e-3,
                    help="Scaling of $\tau_M$ near immersed structure.")
parser.add_argument('--C_pen',dest='C_pen',default=1.0,
                    help="Dimensionless scaling of DAL penalty.")
parser.add_argument('--OUTPUT_SKIP',dest='OUTPUT_SKIP',default=1,
                    help="Write ParaView files every OUTPUT_SKIP steps.")
parser.add_argument('--blockItTol',dest='blockItTol',default=1e-2,
                    help="Relative tolerance for block iteration.")
parser.add_argument('--N_reg',dest='N_reg',default=2,
                    help="No. of time steps to spread BC activation over.")
parser.add_argument('--N_steps_over_Nel',dest='N_steps_over_Nel',default=32,
                    help="No. of time steps per vertical-direction element.")
parser.add_argument('--rho_infty',dest='rho_infty',default=0.0,
                    help="Spectral radius of time integrator at Dt=infty.")
parser.add_argument('--outputFileName',dest='outputFileName',
                    default="qoi-data.csv",
                    help="File to write quanity-of-interest data to.")
parser.add_argument('--r',dest='results_folder',
                    default="./results/",
                    help="Folder to write outputs to.")
parser.add_argument('--log-ksp',dest='log_ksp',
                    default=False,action="store_true",
                    help="Weather or not not log the fluid linear solver.")
parser.add_argument('--ksp_max_its',dest='ksp_max_its',type=int,
                    default=300,
                    help="Number of fluid linear solver iterations.")
parser.add_argument('--ksp_rel_tol',dest='ksp_rel_tol',type=float,
                    default=1e-3,
                    help="Relative tolerance of the fluid linear solver.")
args = parser.parse_args()

# For FSI, fluids, and shells:
import CouDALFISh as aledal
import VarMINT as alevms
from ShNAPr.SVK import *

# For spline setup:
from tIGAr.BSplines import *
from tIGAr.timeIntegration import *

# Suppress excessive output in parallel:
set_log_active(False)
if (mpirank==0):
    set_log_active(True)
parameters['form_compiler']['representation'] = 'tsfc'
sys.setrecursionlimit(10000)

from dolfin.cpp.log import LogLevel as LL
from dolfin.cpp.log import log as log

# Check whether the user downloaded and extracted the data files to the
# working directory; assume that if one is present all are, since including
# just the first would be an unlikely mistake.
fnamePrefix = "smesh."
fnameSuffix = ".dat"
fnameCheck = fnamePrefix+"1"+fnameSuffix
import os.path
if(not os.path.isfile(fnameCheck)):
    error("Missing data files for shell structure geometry. "
            +"Please refer to the docstring at the top of this script.")

Nel_f_vert = int(args.Nel_f_vert)
igaQuadDeg = int(args.igaQuadDeg)
femQuadDeg = int(args.femQuadDeg)
T = float(args.T)
stabEps = Constant(float(args.stabEps))
C_pen = Constant(float(args.C_pen))
OUTPUT_SKIP = int(args.OUTPUT_SKIP)
blockItTol = float(args.blockItTol)
N_reg = int(args.N_reg)
N_steps_over_Nel = int(args.N_steps_over_Nel)
rho_infty = Constant(float(args.rho_infty))
RESULTS_FOLDER = str(args.results_folder)
outputFileName = RESULTS_FOLDER + str(args.outputFileName)
LOG_KSP = bool(args.log_ksp)
KSP_MAX_ITS = int(args.ksp_max_its)
KSP_REL_TOL = float(args.ksp_rel_tol)


# Fixed parameters of the benchmark problem:

d = 3
L = 8.0 # Horizontal length of the domain
H = 1.61 # Vertical length of the domain
D = 1.0 # Depth of the domain in the $x_3$ direction
mu = Constant(10.0) # Dynamic viscosity of the fluid
E = Constant(5.6e7) # Young's modulus of the structure
nu = Constant(0.4) # Poisson's ratio of the structure
rho = Constant(1e2) # Mass density of the fluid
rho0 = Constant(1e2) # Mass density of the structure
h_th = Constant(0.0212) # Thickness of shell structure

# Derived parameters:

Nel_f_horiz = 5*Nel_f_vert # Number of elements in the horizontal direction
N_steps = N_steps_over_Nel*Nel_f_vert # Number of time steps
Dt = Constant(T/N_steps) # Time step size
h_f = H/Nel_f_vert # Fluid mesh length scale
penalty = C_pen*float(mu)/h_f # DAL penalty

####### Read in structure mesh and generate extracted spline #######

log(LL.INFO,"Generating extraction data...")
    
# Load a control mesh from several files in a legacy ASCII format; must use
# triangles for evaluation of tip displacement.
controlMesh = LegacyMultipatchControlMesh(fnamePrefix,2,fnameSuffix,
                                          useRect=False)

# Every processor has a full copy of the shell structure, on its
# MPI_SELF communicator.
splineGenerator = EqualOrderSpline(selfcomm,d,controlMesh)

# Clamped BCs on the fixed ends of the leaflets:
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
for patch in range(0,2):
    side = 0
    direction = 0
    sideDofs = scalarSpline.getPatchSideDofs(patch,direction,side,nLayers=2)
    for i in range(0,d):
        splineGenerator.addZeroDofs(i,sideDofs)

# Eliminate all vertical motion:
for patch in range(0,2):
    for side in range(0,2):
        direction = 1
        sideDofs = scalarSpline.getPatchSideDofs(patch,direction,side,nLayers=1)
        field = 2
        splineGenerator.addZeroDofs(field,sideDofs)
                    
log(LL.INFO,"Creating extracted spline...")

# Quadrature degree for the analysis:
QUAD_DEG = 4

# Generate the extracted representation of the spline.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

####### FE mesh and spaces #######

GEO_EPS = 1e8*DOLFIN_EPS
mesh = BoxMesh(worldcomm,
               Point(-GEO_EPS,-GEO_EPS,-GEO_EPS),
               Point(L+GEO_EPS,H+GEO_EPS,D+GEO_EPS),
               int(Nel_f_horiz/2),int(Nel_f_vert/2),1)
# the mesh refinement algorithm maintains much better symmetry in the fluid,
# which leads to symmetric results in the valve displacement
# without this, the valve response is highly asymetric
mesh = refine(mesh)
V_f = alevms.equalOrderSpace(mesh)
Vscalar = FunctionSpace(mesh,"Lagrange",1)
x_f = SpatialCoordinate(mesh)

####### Explicit Mesh Motion #######

t = Constant(0.0)
y = SpatialCoordinate(mesh)
amplitude = Constant(0.15)
frequency = Constant(8)
term = sin(2*pi*y[0]/L)*sin(2*pi*y[1]/H)
uhat = amplitude*sin(frequency*pi*t/T)*as_vector((term,term,0.0))
vhat = diff(uhat,t)
ahat = diff(vhat,t)

V_m = VectorFunctionSpace(mesh,"CG",1)
uhat_real = Function(V_m)
uhat_real_old = Function(V_m)
vhat_real_old = Function(V_m)
ahat_real_old = Function(V_m)
uhat_real_old.assign(project(uhat,V_m,solver_type='gmres'))
vhat_real_old.assign(project(vhat,V_m,solver_type='gmres'))
ahat_real_old.assign(project(ahat,V_m,solver_type='gmres'))
timeInt_m = GeneralizedAlphaIntegrator(rho_infty,Dt,uhat_real,(uhat_real_old,vhat_real_old,ahat_real_old))
uhat_alpha = timeInt_m.x_alpha()
vhat_alpha = timeInt_m.xdot_alpha()

meshProblem = aledal.ExplicitMeshMotion(timeInt_m,uhat)


####### Boundary data #######

# Boundary velocity:
def v_BCf(t):
    return as_vector([5.0*(sin(2.0*pi*t) + 1.1)*x_f[1]*(H - x_f[1]),
                      Constant(0.0),Constant(0.0)])

# With ramp-up over N_reg steps:
T_reg = Constant(N_reg)*Dt
v_BC = conditional(gt(t,T_reg),v_BCf(t),t*v_BCf(T_reg)/T_reg)


####### Formulations #######

# Shell structure subproblem, using ShNAPr:
y_hom = Function(spline.V)
y_old_hom = Function(spline.V)
ydot_old_hom = Function(spline.V)
yddot_old_hom = Function(spline.V)
timeInt_sh = GeneralizedAlphaIntegrator(rho_infty,Dt,y_hom,
                                        (y_old_hom,ydot_old_hom,yddot_old_hom),
                                        useFirstOrderAlphaM=True)
y_alpha_hom = timeInt_sh.x_alpha()
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)

X = spline.F
x = X + spline.rationalize(y_alpha_hom)
Wint = surfaceEnergyDensitySVK(spline,X,x,E,nu,h_th)*spline.dx

yddot = spline.rationalize(timeInt_sh.xddot_alpha())
res_sh = inner(rho0*h_th*yddot,z)*spline.dx \
         + (1.0/timeInt_sh.ALPHA_F)*derivative(Wint,y_hom,z_hom)

# Fluid subproblem, using VarMINT:
up = Function(V_f)
u,p = split(up)
up_old = Function(V_f)
updot_old = Function(V_f)
timeInt_f = GeneralizedAlphaIntegrator(rho_infty,Dt,up,(up_old,updot_old))
vq = TestFunction(V_f)

def uPart(up):
    return as_vector([up[0],up[1],up[2]])

dx = dx(metadata={"quadrature_degree":femQuadDeg})
ds = ds(metadata={"quadrature_degree":femQuadDeg})
v,q = split(vq)
up_alpha = timeInt_f.x_alpha()
updot_alpha = timeInt_f.xdot_alpha()
u_alpha = uPart(up_alpha)
p_alpha = up_alpha[3]
u_t_alpha = uPart(updot_alpha)
cutFunc = Function(Vscalar)
res_f = alevms.interiorResidual(u_alpha,p,v,q,rho,mu,mesh,
                                uhat=uhat_alpha,vhat=vhat_alpha,
                                v_t=u_t_alpha,Dt=Dt,dy=dx,
                                stabScale=aledal.stabScale(cutFunc,stabEps))

# Flag vertical sides for weakBCs:
weakBCDomain = CompiledSubDomain("x[0]<0.0"+
                                 " || x[1]<0.0 || x[1]>"+str(H))
weakBCIndicator = MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
WEAK_BC_FLAG = 1
ds = ds(subdomain_data=weakBCIndicator)
weakBCDomain.mark(weakBCIndicator,WEAK_BC_FLAG)
res_f += alevms.weakDirichletBC(u_alpha,p,v,q,v_BC,rho,mu,mesh,
                                uhat=uhat_alpha,
                                vhat=vhat_alpha,
                                ds=ds(WEAK_BC_FLAG))

# Slip BCs on top and bottom:
bcs_f = [DirichletBC(V_f.sub(0).sub(d-1),Constant(0.0),
                     "x[2]<0.0 || x[2]>"+str(D)),]

# GMRES fluid solver:
fluidLinearSolver = PETScKrylovSolver("gmres","jacobi")
fluidLinearSolver.ksp().setGMRESRestart(KSP_MAX_ITS)
fluidLinearSolver.parameters['maximum_iterations'] = KSP_MAX_ITS
fluidLinearSolver.parameters['error_on_nonconvergence'] = False
fluidLinearSolver.parameters['monitor_convergence'] = LOG_KSP
fluidLinearSolver.parameters['report'] = LOG_KSP
fluidLinearSolver.parameters['relative_tolerance'] = KSP_REL_TOL
fluidLinearSolver.set_norm_type(PETScKrylovSolver.norm_type.unpreconditioned)

# Coupling with CouDALFISh:
fsiProblem = aledal.CouDALFISh(mesh,res_f,timeInt_f,
                               spline,res_sh,timeInt_sh,
                               penalty=penalty,
                               bcs_f=bcs_f,
                               blockItTol=blockItTol,
                               cutFunc=cutFunc,
                               meshProblem=meshProblem,
                               fluidLinearSolver=fluidLinearSolver,
                               Dres_sh=derivative(res_sh,y_hom),
                               r=0.0)

####### Execution #######

# shell space
scalar_space = spline.cpFuncs[0].function_space()
scalar_element = scalar_space.ufl_element()
vector_element = VectorElement(scalar_element,dim=d)
shell_vis_space = FunctionSpace(scalar_space.mesh(),vector_element)
prms = {'quadrature_degree':2*shell_vis_space.ufl_element().degree()}

# shell visualization
parametricCoords3D = as_vector([spline.parametricCoordinates()[0],
                                spline.parametricCoordinates()[1],
                                Constant(0)])
relative_spatial_coords = spline.spatialCoordinates()-parametricCoords3D
coordsRef = project(relative_spatial_coords,shell_vis_space,
                    form_compiler_parameters=prms)
coordsRef.rename("coordsRef","coordsRef")

# further shell visualization
outfile_sh = XDMFFile(selfcomm,RESULTS_FOLDER+"shell-displacements.xdmf")
outfile_sh.parameters["flush_output"] = True
outfile_sh.parameters["functions_share_mesh"] = True
outfile_sh.parameters["rewrite_function_mesh"] = False

# fluid-solid-mesh visualization
outfile_fsm = XDMFFile(worldcomm,RESULTS_FOLDER+"fluid-mesh-upuhat.xdmf")
outfile_fsm.parameters["flush_output"] = True
outfile_fsm.parameters["functions_share_mesh"] = True
outfile_fsm.parameters["rewrite_function_mesh"] = False

# report size of problem
log(LL.INFO,"Fluid-Solid DOFs: "+str(V_f.dim()))
log(LL.INFO,"Shell FEM DOFs: "+str(spline.M.size(0)))
log(LL.INFO,"Shell IGA DOFs: "+str(spline.M.size(1)))
log(LL.INFO,"Shell Lagrange nodes: "+str(spline.V_control
                                                .tabulate_dof_coordinates()
                                                .shape[0]))

# Time stepping loop:
t.assign(t+Dt)
for timeStep in range(0,N_steps):
    
    log(LL.INFO,"------- Time step "
                 +str(timeStep+1)+"/"
                 +str(N_steps)+" -------")
 
    # Output fields needed for visualization.
    if(timeStep % OUTPUT_SKIP == 0):

        # Structure:
        if(mpirank==0):
            # save shell to XDMF File
            y = spline.rationalize(y_hom)   
            d = project(y,shell_vis_space,form_compiler_parameters=prms)
            d.rename("d","d")
            outfile_sh.write(d,float(t))
            outfile_sh.write(coordsRef,float(t))

        # Fluid--solid--mesh motion solution:
        (v, p) = up.split()
        v.rename("u","u")
        p.rename("p","p")
        uhat_real.rename("uhat","uhat")

        # save fluid-solid-mesh to file
        outfile_fsm.write(v,float(t))
        outfile_fsm.write(p,float(t))
        outfile_fsm.write(uhat_real,float(t))

    # Compute the time step for the coupled problem:
    fsiProblem.takeStep()


    # Write out tip displacement:
    if(mpirank==0):

        # This is not ideal, as it requires some "inside knowledge" of how
        # the spline parametric space is modified for multi-patch splines.
        xi_tip_1 = array([1.0-DOLFIN_EPS,0.5])
        xi_tip_2 = array([3.0-DOLFIN_EPS,0.5])

        tip_disp_1 = y_hom(xi_tip_1)
        tip_disp_2 = y_hom(xi_tip_2)
        mode = "a"
        if(timeStep==0):
            mode = "w"
        outFile = open(outputFileName,mode)
        # Note that time t is at the alpha level of the current step:
        outFile.write(str(float(t))+","
                      +str(tip_disp_1[0])+","
                      +str(tip_disp_1[1])+","
                      +str(tip_disp_2[0])+","
                      +str(tip_disp_2[1])+"\n")
        outFile.close()

    # Advance time:
    t.assign(t+Dt)

    # print timings
    list_timings(TimingClear.clear,[TimingType.wall])