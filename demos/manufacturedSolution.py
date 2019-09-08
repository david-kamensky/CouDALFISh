"""
This demo illustrates the convergence of DAL-based FSI under space--time
refinement, using the manufactured fluid--thin structure interaction solution
from Section 7 of

https://doi.org/10.1142/S0218202518500537

The problem is solved in 3D, due to the design of CouDALFISh, but the exact
solution is $x_3$-invariant, and refinement is only carried out in the $x_1$ 
and $x_2$ directions.
"""

# For FSI, fluids, and shells:
from CouDALFISh import *
from VarMINT import *
from ShNAPr.SVK import *

# For geometry construction:
from tIGAr.NURBS import *
from igakit.nurbs import NURBS as NURBS_ik
from igakit.io import PetIGA
from numpy import array

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse
parser = argparse.ArgumentParser()

# ------- Parameter to change for convergence study -------
parser.add_argument('--level',dest='level',default=4,
                    help="Level of mesh-doubling;"
                    +" time step refined proportionally.")
# ---------------------------------------------------------

# (Defaults are reasonable for remaining parameters.)
parser.add_argument('--igaQuadDeg',dest='igaQuadDeg',default=4,
                    help="Quadrature degree for IGA shell subproblem.")
parser.add_argument('--femQuadDeg',dest='femQuadDeg',default=2,
                    help="Quadrature degree for FEM fluid subproblem.")
parser.add_argument('--L',dest='L',default=1.0,
                    help="Side length of the fluid domain.")
parser.add_argument('--H',dest='H',default=0.25,
                    help="Depth of domain in $x_3$-direction.")
parser.add_argument('--V',dest='V',default=1.0,
                    help="Velocity scale.")
parser.add_argument('--DP',dest='DP',default=1.0,
                    help="Pressure jump across structure in exact solution.")
parser.add_argument('--T',dest='T',default=1.0,
                    help="Time interval.")
parser.add_argument('--mu',dest='mu',default=0.1,
                    help="Fluid dynamic viscosity.")
parser.add_argument('--rho',dest='rho',default=1.0,
                    help="Fluid mass density.")
parser.add_argument('--stabEps',dest='stabEps',default=1e-3,
                    help="Scaling of $\tau_M$ near immersed structure.")
parser.add_argument('--E',dest='E',default=1.0,
                    help="Shell structure Young's modulus.")
parser.add_argument('--nu',dest='nu',default=0.3,
                    help="Shell structure Poisson's ratio.")
parser.add_argument('--h_th',dest='h_th',default=0.3,
                    help="Shell structure thickness.")
parser.add_argument('--prestress',dest='prestress',default=1e1,
                    help="11 component of shell's membrane prestress.")
parser.add_argument('--rho0',dest='rho0',default=1.0,
                    help="Reference mass density of shell structure.")
parser.add_argument('--C_pen',dest='C_pen',default=1.0,
                    help="Dimensionless scaling of DAL penalty.")
parser.add_argument('--OUTPUT_SKIP',dest='OUTPUT_SKIP',default=1,
                    help="Write ParaView files every OUTPUT_SKIP steps.")
parser.add_argument('--blockItTol',dest='blockItTol',default=1e-3,
                    help="Relative tolerance for block iteration.")

args = parser.parse_args()
level = int(args.level)
igaQuadDeg = int(args.igaQuadDeg)
femQuadDeg = int(args.femQuadDeg)
L = float(args.L)
H = float(args.H)
V = Constant(float(args.V))
DP = Constant(float(args.DP))
T = float(args.T)
mu = Constant(float(args.mu))
rho = Constant(float(args.rho))
stabEps = Constant(float(args.stabEps))
E = Constant(float(args.E))
nu = Constant(float(args.nu))
h_th = Constant(float(args.h_th))
prestress = Constant(((float(args.prestress), 0.0),
                      (0.0                  , 0.0)))
rho0 = Constant(float(args.rho0))
C_pen = Constant(float(args.C_pen))
OUTPUT_SKIP = int(args.OUTPUT_SKIP)
blockItTol = float(args.blockItTol)

# Derived parameters:
Nel_f = 3*(2**(level))
#N_steps = 16*Nel_f #2*Nel_f
N_steps = 2*Nel_f
Dt = Constant(T/N_steps)
h_f = L/Nel_f
penalty = C_pen*float(mu)/h_f

####### Shell geometry creation #######

if(mpirank==0):
    print("Generating extraction data...")
    
# Open knot vectors for one Bezier-element with arc-length parameterization:
uKnots = [0.0,0.0,L,L]
vKnots = [0.0,0.0,H,H]

# Array of control points for the shell structure:
cpArray = array([[[L/2,0  ,0],[L/2,0  ,H]],
                 [[L/2,L  ,0],[L/2,L  ,H]]])

# Create initial mesh:
ikNURBS = NURBS_ik([uKnots,vKnots],cpArray)

# (Could degree-elevate here using igakit, for higher degree/continuity,
# although it is not necessary, because the problem assumes only membrane
# stiffness.)

# Refinement in $u$-direction via knot-insertion using igakit:
numNewKnots = 1
for i in range(0,level+3):
    numNewKnots *= 2
h = L/float(numNewKnots)
numNewKnots -= 1
knotList = []
for i in range(0,numNewKnots):
    knotList += [float(i+1)*h,]
newKnots = array(knotList)
ikNURBS.refine(0,newKnots)

####### tIGAr setup #######

if(mpirank==0):
    print("Generating extraction...")

# Read in the generated geometry to create a control mesh.
splineMesh = NURBSControlMesh(ikNURBS,useRect=True)
splineGenerator = EqualOrderSpline(selfcomm,d,splineMesh)

# Set Dirichlet boundary conditions. 
scalarSpline = splineGenerator.getControlMesh().getScalarSpline()
parametricDirection = 0
for side in [0,1]:
    sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
    for field in range(0,d):
        splineGenerator.addZeroDofs(field,sideDofs)
parametricDirection = 1
field = d-1
for side in [0,1]:
    sideDofs = scalarSpline.getSideDofs(parametricDirection,side)
    splineGenerator.addZeroDofs(field,sideDofs)
        
if(mpirank==0):
    print("Setting up extracted spline...")
spline = ExtractedSpline(splineGenerator,igaQuadDeg)

####### FE mesh and spaces #######

GEO_EPS = 1e8*DOLFIN_EPS
mesh = BoxMesh(Point(-GEO_EPS,-GEO_EPS,-GEO_EPS),
               Point(L+GEO_EPS,L+GEO_EPS,H+GEO_EPS),
               Nel_f,Nel_f,1)
VE = VectorElement("Lagrange",mesh.ufl_cell(),1)
QE = FiniteElement("Lagrange",mesh.ufl_cell(),1)
VQE = MixedElement([VE,QE])
V_f = FunctionSpace(mesh,VQE)
Vscalar = FunctionSpace(mesh,"Lagrange",1)
x_f = SpatialCoordinate(mesh)

####### Exact solution #######

# Manufacture the exact solution; cf. Section 7 of the linked reference from
# the docstring at the top of the script.

# Time, as an Expression, to be updated:
t = Expression("t",t=0.0,degree=0)

# Auxilliary function used in defining exact solution:
def Y(x):
    return L*x - x*x
def Yp(x):
    return L - 2.0*x
def Ypp(x):
    return -2.0

# Global Cartesian basis:
e1 = Constant((1,0,0))
e2 = Constant((0,1,0))

# Exact shell structure displacement:
def y_exact(X):
    return V*t*Y(X[1])*e1

# Exact fluid velocity:
def u_left(x):
    return V*Y(x[1])*e1
def u_shear(x):
    return (V*(x[0] - (V*t*Y(x[1])+0.5*L))/L)\
        *(V*t*Yp(x[1])*e1 + e2)
def u_right(x):
    return V*Y(x[1])*e1 + u_shear(x)
def rlcond(x,right,left):
    return conditional(gt(x[0],0.5*L+V*t*Y(x[1])),right,left)
def u_exact(x):
    return rlcond(x,u_right(x),u_left(x))

# Partial time derivative of exact fluid velocity:
def u_t_exact(x):
    return rlcond(x,
                  (V*V*(x[0]-0.5*L)*Yp(x[1])
                   - 2.0*V*V*V*t*Y(x[1])*Yp(x[1]))/L*e1
                  - V*V*Y(x[1])/L*e2,
                  Constant((0,0,0)))

# Exact pressure, assuming zero pressure at the origin for uniqueness:
def p_exact(x):
    return rlcond(x,-DP,Constant(0.0))

# Exact normal to shell structure:
def n_s(X2):
    return (1.0/sqrt(1+(V*t*Yp(X2))**2))*as_vector((1.0,-V*t*Yp(X2),0))

# Gradients of fluid solution to either side of the structure:

# These need to be evaluated at points on the shell midsurface, and thus
# simply taking the UFL grad() would not work.  Coding could be simplified
# via judicious use of diff(), but hand-calculuation here is not excessive.
def gradu_left(x):
    return as_tensor(((0 , V*Yp(x[1]) , 0),
                      (0 ,          0 , 0),
                      (0 ,          0 , 0)))
def gradu_right(x):
    return as_tensor(((V*V*t*Yp(x[1])/L ,
                       V*Yp(x[1])
                       +V*V*t/L*(x[0]*Ypp(x[1])
                                 -V*t*(Yp(x[1])*Yp(x[1])+Y(x[1])*Ypp(x[1]))
                                 -0.5*L*Ypp(x[1]))    , 0),
                      (V/L        , -V*V*t*Yp(x[1])/L , 0),
                      (0          , 0                 , 0)))

# Traction from fluid on shell structure:
def tau_visc(gradu):
    return 2.0*mu*sym(gradu)
def tau_left(x):
    return tau_visc(gradu_left(x))
def tau_right(x):
    return tau_visc(gradu_right(x))
def g11(X2):
    return 1.0 + (V*t*Yp(X2))**2
def minusLam(X2):
    x = (0.5*L + V*t*Y(X2))*e1 + X2*e2
    return tau_right(x)*n_s(X2) - tau_left(x)*n_s(X2) + DP*n_s(X2)

# Shell structure body force to manufacture y_exact:
def f_shell(X2):
    f1 = -(1.5*E*h_th/(1.0-nu*nu)
           *((V*t)**3)*((Yp(X2))**2) + prestress[0,0]*V*t)*Ypp(X2) \
           - sqrt(g11(X2))*minusLam(X2)[0]
    f2 = -E*h_th/(1.0-nu*nu)*((V*t)**2)*Yp(X2)*Ypp(X2) \
         - sqrt(g11(X2))*minusLam(X2)[1]

    #f1 = -sqrt(g11(X2))*minusLam(X2)[0]
    #f2 = -sqrt(g11(X2))*minusLam(X2)[1]
    
    return as_vector((f1,f2,0))

####### Formulations #######

# Shell structure subproblem, using ShNAPr:
y_hom = Function(spline.V)
y_old_hom = Function(spline.V)
y_alpha_hom = x_alpha(y_hom,y_old_hom)
ydot_old_hom = Function(spline.V)
z_hom = TestFunction(spline.V)
z = spline.rationalize(z_hom)

X = spline.F
x = X + spline.rationalize(y_alpha_hom)
Wint = surfaceEnergyDensitySVK(spline,X,x,E,nu,h_th,
                               membrane=True,
                               membranePrestress=prestress)*spline.dx

yddot = spline.rationalize(xddot_alpha(y_hom,y_old_hom,ydot_old_hom,Dt))
res_sh = inner(rho0*h_th*yddot - f_shell(X[1]),z)*spline.dx \
         + dx_dx_alpha*derivative(Wint,y_hom,z_hom)

# Initial condition for shell:
ydot_old_hom.assign(spline.project(u_left(X),rationalize=False))

# Fluid subproblem, using VarMINT:
up = Function(V_f)
up_old = Function(V_f)
vq = TestFunction(V_f)

dx = dx(metadata={"quadrature_degree":femQuadDeg})
ds = ds(metadata={"quadrature_degree":femQuadDeg})
u,p = split(up)
u_old,_ = split(up_old)
v,q = split(vq)
u_alpha = x_alpha(u,u_old)
u_t_alpha = xdot_alpha(u,u_old,Dt)
cutFunc = Function(Vscalar)
f_f,_ = strongResidual(u_exact(x_f),p_exact(x_f),mu,rho,u_t=u_t_exact(x_f))
f_f /= rho
res_f = interiorResidual(u_alpha,p,v,q,rho,mu,mesh,
                         u_t=u_t_alpha,Dt=Dt,dx=dx,f=f_f,
                         stabScale=stabScale(cutFunc,stabEps))
# Flag vertical sides for weakBCs:
weakBCDomain = CompiledSubDomain("x[0]<0.0 || x[0]>"+str(L)
                                 +" || x[1]<0.0 || x[1]>"+str(L))
weakBCIndicator = MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
WEAK_BC_FLAG = 1
ds = ds(subdomain_data=weakBCIndicator)
weakBCDomain.mark(weakBCIndicator,WEAK_BC_FLAG)
res_f += weakDirichletBC(u_alpha,p,v,q,u_exact(x_f),rho,mu,mesh,
                         ds=ds(WEAK_BC_FLAG))

# Slip BCs on top and bottom:
bcs_f = [DirichletBC(V_f.sub(0).sub(d-1),Constant(0.0),
                     "x[2]<0.0 || x[2]>"+str(H)),]
# Pin pressure down at the origin for a unique solution:
bcs_f += [DirichletBC(V_f.sub(1),Constant(0.0),
                      "x[0]<0.0 && x[1]<0.0 && x[2]<0.0",
                      "pointwise"),]

# Initial condition for fluid:
ue = u_exact(x_f)
up_old.assign(project(as_vector([ue[0],ue[1],ue[2],p_exact(x_f)]),V_f))

# Coupling with CouDALFISh:
fsiProblem = CouDALFISh(mesh,res_f,up,up_old,
                        spline,res_sh,y_hom,y_old_hom,ydot_old_hom,
                        Dt,penalty,
                        bcs_f=bcs_f,
                        blockItTol=blockItTol,cutFunc=cutFunc,
                        r=0.0)

# Initial condition for Lagrange multiplier:
fsiProblem.lam.fill(float(DP))

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
t.t = 0.5*float(Dt)
#t.t = float(Dt)
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
    t.t += float(Dt)

####### Check errors #######
t.t = T
import math
def L2Norm_f(u):
    return math.sqrt(assemble(inner(u,u)*dx))
def L2Norm_sh(u):
    return math.sqrt(assemble(inner(u,u)*spline.dx))
eu = u - u_exact(x_f)
ep = p - p_exact(x_f)
ey = spline.rationalize(y_hom) - y_exact(X)
print("L2 error in fluid velocity = "+str(L2Norm_f(eu)))
print("H1 error in fluid velocity = "+str(L2Norm_f(grad(eu))))
print("L2 error in fluid pressure = "+str(L2Norm_f(ep)))
print("L2 error in shell displacement = "+str(L2Norm_sh(ey)))
print("H1 error in shell displacement = "+str(L2Norm_sh(spline.grad(ey))))
