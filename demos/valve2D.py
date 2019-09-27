"""
This demo solves the 2D valve benchmark from Section 4.7 of

  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4274080/

The benchmark is derived from one introduced in prior works, as indicated by
the citations in the linked paper, although certain details of the setup
vary between references.  

The exact solution is unknown, and works using this problem (or a similar one)
for verification typically plot and qualitatively-compare time histories of 
displacements at the free end of the top "leaflet" for discretizations of
differing resolutions.  Tip displacement is outputted to a file whose name
defaults to 'tip-dislacement' in the working directory, with whitespace-
separated columns containing time, ``$x_1$`` displacement, and ``$x_2$`` 
displacement.

Although the geometry is relatively simple, it is given in external files 
available in the following archive

  https://www.dropbox.com/s/aa4j1byiucv5pqo/2d-valve-geometry.tgz?dl=1

because, at time of writing, that is the only input format that supports 
multi-patch splines.  (The use of a fixed mesh for the structure subproblem 
will ultimately limit convergence, but discretization error is overwhelmingly
due to poor approximately of discontinuous pressure and velocity gradients
in the fluid subproblem, and we are not exploring convergence rigorously here,
using Sobolev norms, so it is good enough for the present purposes.)  
"""

# For FSI, fluids, and shells:
from CouDALFISh import *
from VarMINT import *
from ShNAPr.SVK import *

# For spline setup:
from tIGAr.BSplines import *

# Miscellaneous utilities:
from numpy import array

# Check whether the user downloaded and extracted the data files to the
# working directory; assume that if one is present all are, since including
# just the first would be an unlikely mistake.
fnamePrefix = "smesh."
fnameSuffix = ".dat"
fnameCheck = fnamePrefix+"1"+fnameSuffix
import os.path
if(not os.path.isfile(fnameCheck)):
    if(mpirank==0):
        print("ERROR: Missing data files for shell structure geometry. "
              +"Please refer to the docstring at the top of this script.")
    exit()

# Suppress excessive output in parallel:
parameters["std_out_all_processes"] = False

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
                    default="tip-displacement",
                    help="File to write tip displacement data to.")


args = parser.parse_args()
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
outputFileName = str(args.outputFileName)

# Fixed parameters of the benchmark problem:

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

if(mpirank==0):
    print("Generating extraction data...")
    
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
                    
if(mpirank==0):
    print("Creating extracted spline...")

# Quadrature degree for the analysis:
QUAD_DEG = 4

# Generate the extracted representation of the spline.
spline = ExtractedSpline(splineGenerator,QUAD_DEG)

####### FE mesh and spaces #######

GEO_EPS = 1e8*DOLFIN_EPS
mesh = BoxMesh(Point(-GEO_EPS,-GEO_EPS,-GEO_EPS),
               Point(L+GEO_EPS,H+GEO_EPS,D+GEO_EPS),
               Nel_f_horiz,Nel_f_vert,1)
V_f = equalOrderSpace(mesh)
Vscalar = FunctionSpace(mesh,"Lagrange",1)
x_f = SpatialCoordinate(mesh)

####### Boundary data #######

# Time, as an Expression, to be updated:
t = Expression("t",t=0.0,degree=0)

# Boundary velocity:
def v_BCf(T):
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
p = up[3]
u_t_alpha = uPart(updot_alpha)
cutFunc = Function(Vscalar)
res_f = interiorResidual(u_alpha,p,v,q,rho,mu,mesh,
                         u_t=u_t_alpha,Dt=Dt,dx=dx,
                         stabScale=stabScale(cutFunc,stabEps))

# Flag vertical sides for weakBCs:
weakBCDomain = CompiledSubDomain("x[0]<0.0"+
                                 " || x[1]<0.0 || x[1]>"+str(H))
weakBCIndicator = MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
WEAK_BC_FLAG = 1
ds = ds(subdomain_data=weakBCIndicator)
weakBCDomain.mark(weakBCIndicator,WEAK_BC_FLAG)
res_f += weakDirichletBC(u_alpha,p,v,q,v_BC,rho,mu,mesh,
                         ds=ds(WEAK_BC_FLAG))

# Slip BCs on top and bottom:
bcs_f = [DirichletBC(V_f.sub(0).sub(d-1),Constant(0.0),
                     "x[2]<0.0 || x[2]>"+str(D)),]

# Coupling with CouDALFISh:
fsiProblem = CouDALFISh(mesh,res_f,timeInt_f,
                        spline,res_sh,timeInt_sh,
                        penalty,
                        bcs_f=bcs_f,
                        blockItTol=blockItTol,cutFunc=cutFunc,
                        r=0.0)

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

# Time stepping loop:
t.t = float(timeInt_f.ALPHA_F)*float(Dt)
for timeStep in range(0,N_steps):
    
    if(mpirank==0):
        print("------- Time step "+str(timeStep+1)+"/"+str(N_steps)+" -------")
 
    # Output fields needed for visualization.
    if(timeStep % OUTPUT_SKIP == 0):

        # Structure:
        if(mpirank==0):
            (d0,d1,d2) = y_hom.split()
            d0.rename("d0","d0")
            d1.rename("d1","d1")
            d2.rename("d2","d2")
            d0File << d0
            d1File << d1
            d2File << d2
            # (Note that the components of spline.F are rational, and cannot be
            # directly outputted to ParaView files.)
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

    # Write out tip displacement:
    if(mpirank==0):

        # This is not ideal, as it requires some "inside knowledge" of how
        # the spline parametric space is modified for multi-patch splines.
        xi_tip = array([1.0-DOLFIN_EPS,0.5])

        tipDisp = y_hom(xi_tip)
        mode = "a"
        if(timeStep==0):
            mode = "w"
        outFile = open(outputFileName,mode)
        # Note that time t is at the alpha level of the current step:
        outFile.write(str(t.t+(1.0-float(timeInt_sh.ALPHA_F))*float(Dt))
                      +" "+str(tipDisp[0])+" "
                      +str(tipDisp[1])+"\n")
        outFile.close()

    # Advance time:
    t.t += float(Dt)
