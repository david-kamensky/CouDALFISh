"""
The ``CouDALFISh`` module:
--------------------------
This module provides functionality for coupling a velocity field of some 3D
finite element problem (which is conceptually a fluid), discretized using 
FEniCS (e.g., but not necessarily, via ``VarMINT``), with the displacement 
fieldof a shell structure, discretized using the ``ShNAPr`` library for 
``tIGAr``-based isogeometric shell structure analysis.  Coupling is based 
on the dynamic augmented Lagrangian (DAL) method.
"""

import numpy as np
from ShNAPr.kinematics import *
from ShNAPr.contact import *
from mpi4py import MPI as pyMPI
from numpy import zeros, save, load
from tIGAr.timeIntegration import *
from abc import ABC, abstractmethod

# This module assumes 3D problems:
d = 3
ADD_MODE = PETSc.InsertMode.ADD

# Log the individual shell residuals
LOG_SHELL_RESIDUALS = True

# Assume some tight absolute tolerances
FLUID_ABS_TOL = 1e-11
SHELL_ABS_TOL = 1e-8

# Import logging functions for syntactic sugar
from dolfin.cpp.log import LogLevel as LL
from dolfin.cpp.log import log as log
from dolfin.cpp.log import end as end
from dolfin.cpp.log import begin as begin

def stabScale(cutFunc,eps):
    """
    Returns a UFL ``Conditional`` that is 1 everywhere except where
    ``cutFunc`` is nonzero, where it is ``eps``.  This is intended for
    use in modifying stabilization parameters in elements where
    ``cutFunc`` indicates that there is a large irrotation source term
    (e.g., from an immersed boundary).  
    """
    return conditional(gt(abs(cutFunc),0.0),eps,1.0)

class MeshMotionProblem(ABC):
    def meshMotionRestartName(self,restartPath,timeStep):
        return restartPath+"/restart_m_"+str(timeStep)+".h5"
    @abstractmethod
    def writeRestarts(self,restartPath,timeStep):
        pass
    @abstractmethod
    def readRestarts(self,restartPath,timeStep):
        pass
    @abstractmethod
    def predict(self):
        pass
    @abstractmethod
    def deform(self):
        pass
    @abstractmethod
    def undeform(self):
        pass
    @abstractmethod
    def compute_solution(self):
        pass

class NoMotion(MeshMotionProblem):
    def writeRestarts(self, restartPath, timeStep):
        super().writeRestarts(restartPath, timeStep)
    def readRestarts(self, restartPath, timeStep):
        super().readRestarts(restartPath, timeStep)
    def predict(self):
        super().predict()
    def deform(self):
        super().deform()
    def undeform(self):
        super().undeform()
    def compute_solution(self):
        super().compute_solution()

class KnownMeshMotion(MeshMotionProblem):
    def __init__(self,time_integrator):
        self.time_integrator = time_integrator
        self.V = self.time_integrator.x.function_space()
        self.mesh = self.V.mesh()
        self.deformation = Function(self.V)
        self.reverse_deformation = Function(self.V)
        self.deformed = False
        super().__init__()

    def writeRestarts(self, restartPath, timeStep):
        f = HDF5File(worldcomm,
                        self.meshMotionRestartName(restartPath,timeStep), "w")
        f.write(self.time_integrator.x, "/uhat")
        f.write(self.time_integrator.x_old, "/uhat_old")
        f.write(self.time_integrator.xdot_old, "/uhatdot_old")
        f.write(self.time_integrator.xddot_old, "/uhatdotdot_old")
        f.close()

    def readRestarts(self, restartPath, timeStep):
        f = HDF5File(worldcomm,
                        self.meshMotionRestartName(restartPath,timeStep), "r")
        f.read(self.time_integrator.x, "/uhat")
        f.read(self.time_integrator.x_old, "/uhat_old")
        f.read(self.time_integrator.xdot_old, "/uhatdot_old")
        f.read(self.time_integrator.xddot_old, "/uhatdotdot_old")
        f.close()
    
    @abstractmethod
    def predict(self,prediction):
        self.time_integrator.x.assign(prediction)

    def deform(self):
        if self.deformed:
            raise RuntimeError("Mesh problem is already deformed.")
        self.deformation.assign(self.time_integrator.x_alpha())
        self.reverse_deformation.assign(-self.deformation)
        ALE.move(self.mesh,self.deformation)
        self.deformed = True
        self.mesh.bounding_box_tree().build(self.mesh)

    def undeform(self):
        if not self.deformed:
            raise RuntimeError("Mesh problem is not yet deformed.")
        ALE.move(self.mesh,self.reverse_deformation)
        self.deformed = False
        self.mesh.bounding_box_tree().build(self.mesh)

    def compute_solution(self):
        self.time_integrator.advance()

class ExplicitMeshMotion(KnownMeshMotion):
    def __init__(self,time_integrator,explicit_motion):
        super().__init__(time_integrator)
        self.explicit_motion = explicit_motion

    def predict(self):
        super().predict(project(self.explicit_motion,self.V,solver_type='gmres'))

class SolvedMeshMotion(KnownMeshMotion):
    DEFAULT_MESH_SOLVER_PARAMETERS = {'newton_solver':{
        'linear_solver':'gmres',
        'preconditioner':'jacobi',
        'error_on_nonconvergence':False,
        'maximum_iterations':100,
        'relative_tolerance':1e-4,
        'krylov_solver':{'error_on_nonconvergence':False,
                         'relative_tolerance':1e-6,
                         'maximum_iterations':200}}}

    def __init__(self,time_integrator,res,
                 bcs=[],Dres=None,u_s=None,bc_func=None):
        super().__init__(time_integrator)
        self.res = res
        self.Dres = Dres
        if self.Dres is None:
            self.Dres = derivative(self.res,self.time_integrator.x)
        self.bcs = bcs
        self.u_s = u_s
        self.bc_func = bc_func
        
    def predict(self):
        super().predict(self.time_integrator.sameVelocityPredictor())

    def compute_solution(self,
            solver_parameters=DEFAULT_MESH_SOLVER_PARAMETERS):
        if (self.bc_func is not None and self.u_s is not None):
            self.bc_func.assign(project(self.u_s, self.V, 
                                        solver_type='gmres',
                                        preconditioner_type='jacobi'))
        problem = NonlinearVariationalProblem(self.res, 
                    self.time_integrator.x, 
                    bcs=self.bcs, 
                    J=self.Dres)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters.update(solver_parameters)
        solver.solve()
        self.time_integrator.advance()

class CouDALFISh:
    """
    This class keeps track of data used for coupling a fluid and shell
    structure.
    """
    def __init__(self,mesh_f, res_f, timeInt_f,
                 spline_sh, res_sh, timeInt_sh,
                 penalty, r=0.0,
                 blockItTol=1e-3, blockIts=100,
                 bcs_f=[],
                 Dres_f=None, Dres_sh=None,
                 contactContext_sh=None, nonmatching_sh=None,
                 fluidLinearSolver=PETScLUSolver("mumps"),
                 cutFunc=None):
        """
        ``mesh_f`` is the fluid finite element mesh.  ``res_f`` is the 
        residual of the fluid subproblem formulation, without including
        coupling terms.  ``spline_sh`` and ``res_sh`` are the shell structure's
        ``ExtractedSpline`` and residual (again without coupling terms). The
        arguments ``timeInt_sh`` and ``timeInt_f`` are ``tIGAr`` 
        generalized-alpha integrators for the shell and fluid.  ``Function``
        objects for the current and previous solution states are accessed 
        through these time integrators.  ``penalty`` is the implicit velocity 
        penalty from the DAL method.  

        The optional parameter ``r`` is the stabilization parameter denoted
        by the same character in this paper:

        https://doi.org/10.4208/cicp.150115.170415s

        ``blockItTol`` and ``blockIts`` are the relative tolerance for
        block iteration and the maximum number of block iterations allowed,
        respectively.  ``bcs_f`` is a Python list of ``DirichletBC``s to be
        applied to the fluid subproblem; these BCs are assumed to be
        homogeneous, although inhomogeneous BCs can be applied weakly.  
        ``Dres_f`` and ``Dres_sh`` are custom tangent operators that 
        may be specified for the fluid and solid subproblems, 
        defaulting to ``derivative``s of the corresponding
        residuals.  ``contactContext_sh`` is a ``ShNAPr`` 
        ``ShellContactContext`` instance.  If it is left as the default
        ``None``, there will be no contact in the shell subproblem.  
        ``fluidLinearSolver`` allows specification of a custom Krylov solver
        for the fluid subproblem, which is necessary for most realistic
        3D problems.  (Refer to the demos for appropriate settings, which
        may not be obvious.)  ``cutFunc`` is a function intended to be passed
        as an argument to the free function ``stabScale``, which, in turn
        may be used in the residual ``res_f``, to scale SUPG/PSPG/LSIC
        stabilization parameters.
        """

        # Fluid related attributes
        self.mesh_f = mesh_f
        self.timeInt_f = timeInt_f
        self.fluidFac = float(self.timeInt_f.ALPHA_F)
        self.up = self.timeInt_f.x
        self.up_old = self.timeInt_f.x_old
        self.updot_old = self.timeInt_f.xdot_old
        self.up_alpha = self.timeInt_f.x_alpha()
        self.V_f = self.up.function_space()

        # Shell related attributes
        self.spline_sh = spline_sh
        if isinstance(self.spline_sh, list):
            self.spline_shs = self.spline_sh
            self.multiPatch = True
        elif isinstance(self.spline_sh, ExtractedSpline):
            self.spline_shs = [self.spline_sh]
            self.multiPatch = False
        else:
            raise TypeError("Spline type " + Type(self.spline) 
                            + " is not supported.")
        self.num_splines = len(self.spline_shs)

        self.penalty = penalty
        self.Dt = self.timeInt_f.DELTA_T
        self.timeInt_sh = timeInt_sh
        if self.multiPatch:
            if not isinstance(self.timeInt_sh, list):
                raise Type("``timeInt_sh`` has to be type of list, "
                           "same as the type of input ``spline_sh``")
            self.timeInt_shs = self.timeInt_sh
        else:
            self.timeInt_shs = [self.timeInt_sh]

        self.shellFacs = []
        for i in range(self.num_splines):
            if self.timeInt_shs[i].systemOrder == 2:
                fac = float(self.timeInt_shs[i].ALPHA_F)\
                    *float(self.timeInt_shs[i].GAMMA)\
                    /(float(self.timeInt_shs[i].BETA)*float(self.Dt))
            else:
                fac = float(self.timeInt_shs[i].ALPHA_F)\
                    /(float(self.timeInt_shs[i].GAMMA)*float(self.Dt))
            self.shellFacs += [fac,]

        self.blockIts = blockIts
        self.blockItTol = blockItTol
        self.bcs_f = bcs_f

        if contactContext_sh is None:
            # If no contact context is passed, assume that contact is not
            # part of this FSI problem, and create a dummy context for the
            # nodal quadrature rule.  (Attempting to call ``assembleContact``
            # for this instance will cause an error, due to invalid
            # parameter values.)
            self.includeContact_sh = False
            self.contactContext_sh = ShellContactContext(self.spline_sh,
                                                         None,None,None)
        else:
            self.includeContact_sh = True
            self.contactContext_sh = contactContext_sh

        self.nNodes_shs = self.contactContext_sh.nNodes
        self.nodeX_shs = self.contactContext_sh.nodeXs
        self.r = r
        if not self.multiPatch:
            self.lam = zeros(self.nNodes_shs[0])
            self.lam_old = zeros(self.nNodes_shs[0])
        self.lams = [zeros(nNodes) for nNodes in self.nNodes_shs]

        self.res_f = res_f
        self.res_sh = res_sh
        if self.multiPatch:
            if not isinstance(self.res_sh, list):
                raise Type("``res_sh`` has to be type of list, "
                           "same as the type of input ``spline_sh``")
            self.res_shs = self.res_sh
        else:
            self.res_shs = [self.res_sh]

        self.y_homs = [self.timeInt_shs[i].x for i in range(self.num_splines)]
        self.y_old_homs = [self.timeInt_shs[i].x_old 
                           for i in range(self.num_splines)]
        self.y_alpha_homs = [self.timeInt_shs[i].x_alpha() 
                             for i in range(self.num_splines)]
        self.ydot_old_homs = [self.timeInt_shs[i].xdot_old 
                              for i in range(self.num_splines)]
        self.yddot_old_homs = [self.timeInt_shs[i].xddot_old 
                               for i in range(self.num_splines)]
        self.ydot_homs = [self.timeInt_shs[i].xdot() 
                          for i in range(self.num_splines)]
        self.ydot_alpha_homs = [self.timeInt_shs[i].xdot_alpha() 
                                for i in range(self.num_splines)]

        if Dres_f is None:
            self.Dres_f = derivative(self.res_f,self.up)
        else:
            self.Dres_f = Dres_f

        if Dres_sh is None:
            self.Dres_shs = [derivative(self.res_shs[i], self.y_homs[i]) 
                             for i in range(self.num_splines)]
        else:
            if self.multiPatch:
                if not isinstance(self.Dres_sh, list):
                    raise Type("``Dres_sh`` has to be type of list, "
                               "same as the type of input ``spline_sh``")
                self.Dres_shs = self.Dres_sh
            else:
                self.Dres_shs = [self.Dres_sh]

        self.nonmatching_sh = nonmatching_sh
        if self.nonmatching_sh is not None:
            # Set residuals for non-matching problem
            if self.nonmatching_sh.residuals is None:
                self.nonmatching_sh.set_residuals(self.res_shs, self.Dres_shs)
            # If both the non-matching problem and FSI problem have
            # contact, set the contact in non-matching problem as None
            # in order to avoid adding the contact contributions twice. 
            if self.nonmatching_sh.contact is not None and \
                self.includeContact_sh is True:
                self.nonmatching_sh.contact = None

        self.x_shs = [self.spline_shs[i].F 
                      + self.spline_shs[i].rationalize(self.y_alpha_homs[i]) 
                      for i in range(self.num_splines)]
        
        if cutFunc is None:
            self.Vscalar_f = FunctionSpace(mesh_f,"CG",1)
            self.cutFunc = Function(self.Vscalar_f)
            if(mpirank==0):
                print("***************************************************")
                print("WARNING: cutFunc not provided to CouDALFISh;\n"+
                      "         will not be updated in residuals using it.")
                print("***************************************************")
        else:
            self.cutFunc = cutFunc
            self.Vscalar_f = cutFunc.function_space()

        # UFL expression for shell normal:
        self.a2_shs = []
        for i in range(self.num_splines):
            _,_,a2_sh,_,_,_ = surfaceGeometry(self.spline_shs[i], 
                                              self.x_shs[i])
            self.a2_shs += [a2_sh,]
        
        # Nodal normal field:
        self.n_nodal_shs = [Function(sh.V) for sh in self.spline_shs]
        self.fluidLinearSolver = fluidLinearSolver
        
    def updateNodalNormals(self):
        """
        This projects the normal vector for the shell's current configuration
        onto the (homogeneous) spline space for the displacement via a lumped
        ``$L^2$`` projection.
        """
        for i in range(self.num_splines):
            self.n_nodal_shs[i].assign(self.spline_shs[i].project(
                self.a2_shs[i], lumpMass=True, rationalize=False))

    def pointInFluidMesh(self,x):
        """
        Returns a Boolean indicating whether or not the point ``x`` is in
        the fluid mesh partition for the calling MPI task.
        """
        return len(self.mesh_f.bounding_box_tree()
                   .compute_entity_collisions(Point(x)))>0
        
    def evalFluidVelocity(self,up,x):
        """
        This evaluates the velocity (i.e., first ``d`` components of) the
        ``Function`` ``up`` at point ``x``, returning a zero vector if ``x``
        is outside of the fluid subproblem domain for the calling MPI task.
        """
        if(self.pointInFluidMesh(x)):
            return up(x)[0:d]
        else:
            return d*(0.0,)

    def evalFluidVelocities(self,up,nodexs):
        """
        Given a list of array of points, ``nodexs``, this returns a 
        same-shape array of velocity evaluations from the fluid solution 
        space function ``up``. The array returned is the same on all MPI 
        tasks. For points from ``nodexs`` that are not on any task's mesh 
        partition, the corresponding velocity is zero.
        """
        uFe = [zeros((nNodes, d)) for nNodes in self.nNodes_shs]
        for i in range(0, self.num_splines):
            for j in range(0, self.nNodes_shs[i]):
                uFe[i][j,:] = self.evalFluidVelocity(up, nodexs[i][j,:])
            uFe[i] = worldcomm.allreduce(uFe[i], op=pyMPI.SUM)
        return uFe        

    def couplingForceOnFluid(self,u,ydot,lam,n,w):
        """
        Given fluid velocity ``u``, structure veloctiy ``ydot``, 
        Lagrange multiplier ``lam``, normal vector ``n``, and quadrature
        weight ``w`` at a point, return the force contribution acting on the
        fluid.  (The force acting on the shell at the same point should 
        then be in the opposite direction.)
        """
        return -(self.penalty*(u-ydot) + lam*n)*w
    
    def addFluidCouplingForces(self, u_shs, u_fs, nodexs, ns, A, b):
        """
        Given arrays of shell structure velocities ``u_shs``, fluid 
        velocities, ``u_fs``, positions, ``nodexs``, and normals ``ns`` 
        at FE nodes of the shell structure's mesh, compute coupling 
        forces and apply them to fluid LHS and RHS matrix and vector 
        ``A`` and ``b``.  
        """
        rhsMultiPointSources = []
        lhsMultiPointSources = []
        rhsPointSourceData = []
        lhsPointSourceData = []
        for i in range(0,d):
            rhsPointSourceData += [[],]
            lhsPointSourceData += [[],]

        # Add point source data to lists in serial
        if(mpirank == 0):
            for i in range(self.num_splines):
                for node in range(0,self.nNodes_shs[i]):
                    # Quadrature weight
                    w = self.contactContext_sh.quadWeights[i][node]

                    # Get node's contribution to total force
                    dF = self.couplingForceOnFluid(u_fs[i][node,:], 
                                                   u_shs[i][node,:],
                                                   self.lams[i][node], 
                                                   ns[i][node,:], w)
                    for j in range(0,d):
                        rhsPointSourceData[j] += [(Point(nodexs[i][node]),
                                                   -dF[j]),]
                        # Factor of $\alpha_f$ for time integrator:
                        lhsPointSourceData[j] += [(Point(nodexs[i][node]),
                                                   self.fluidFac
                                                   *self.penalty*w),]

        # Apply scalar point sources to sub-spaces corresponding to different
        # displacement components globally (but only rank 0's lists are
        # non-empty)
        for i in range(0,d):
            rhsMultiPointSources += [PointSource(self.V_f.sub(0).sub(i),
                                                 rhsPointSourceData[i]),]
            lhsMultiPointSources += [PointSource(self.V_f.sub(0).sub(i),
                                                 lhsPointSourceData[i]),]

        # Apply point sources in batches for each dimension
        lmVec = Function(self.V_f).vector()
        for i in range(0,d):
            rhsMultiPointSources[i].apply(b)
            lhsMultiPointSources[i].apply(lmVec)
        as_backend_type(A).mat().setDiagonal(as_backend_type(lmVec).vec(),
                                             addv=ADD_MODE)

    def addShellCouplingForces(self, u_shs, u_fs, ns, Ams, bvs):
        """
        Given arrays of shell structure velocities ``u_shs``, fluid velocities
        ``u_fs``, and normal vectors ``ns`` at FE nodes of the shell structure
        mesh, compute coupling forces and add corresponding contributions to 
        shell structure LHS and RHS matrices and vectors ``Ams`` and ``bvs``.
        """
        for i in range(self.num_splines):
            lmVec = as_backend_type(Function(self.spline_shs[i].V)
                                    .vector()).vec()
            for node in range(0, self.nNodes_shs[i]):
                w = self.contactContext_sh.quadWeights[i][node]
                dF = self.couplingForceOnFluid(u_fs[i][node,:], 
                                               u_shs[i][node,:],
                                               self.lams[i][node],
                                               ns[i][node,:], w)
                for direction in range(0,d):
                    index = nodeToDof(node, direction)
                    bvs[i].setValue(index, dF[direction], addv=ADD_MODE)
                    # Factor for tangent of displacement problem:
                    lmVec.setValue(index, self.penalty*self.shellFacs[i]*w,
                                   addv=ADD_MODE)
            Ams[i][i].setDiagonal(lmVec, addv=ADD_MODE)
        
    def updateMultipliers(self, u_shs, u_fs, ns):
        """
        Given arrays of shell structure velocities ``u_shs``, fluid velocities
        ``u_fs``, and normal vectors ``ns`` at nodes of the shell structure
        FE mesh, update the corresponding FSI Lagrange multiplier samples in
        ``self.lams``.  
        """
        for i in range(self.num_splines):
            for node in range(0, self.nNodes_shs[i]):
                contrib = 0.
                for j in range(d):
                    contrib += self.penalty*(u_fs[i][node,j] \
                             - u_shs[i][node,j])*ns[i][node,j]
                self.lams[i][node] = (self.lams[i][node]+contrib)/(1.0+self.r)


    def updateStabScale(self,nodexs):
        """
        Given an array ``nodexs`` of immersed structure nodal positions,
        this function updates ``self.cutFunc``, whose nonzero 
        values indicate elements near the immersed structure.  
        """
        self.cutFunc.assign(Function(self.Vscalar_f))
        pointSourceData = []
        for i in range(self.num_splines):
            if (mpirank == 0):
                for point in range(0, self.nNodes_shs[i]):
                    pointSourceData += [(Point(nodexs[i][point]), 1.0),]
        multiPointSouces = PointSource(self.Vscalar_f, pointSourceData)
        multiPointSouces.apply(self.cutFunc.vector())


    # These are mainly for internal use, just to ensure that file names
    # match when reading and writing restart data.
    def fluidRestartName(self,restartPath,timeStep):
        return restartPath+"/restart_f."+str(timeStep)+".h5"

    def shellRestartName(self,restartPath,shInd,timeStep):
        return restartPath+"/restart_sh_"+str(shInd)+"."+str(timeStep)+".h5"
        
    def lamRestartName(self,restartPath,shInd,timeStep):
        return restartPath+"/restart_lam_"+str(shInd)+"."+str(timeStep)+".npy"

    def writeRestarts(self,restartPath,timeStep):
        """
        Write out all data needed to restart the computation from step 
        ``timeStep``, where ``restartPath`` is the path to a directory 
        containing restart files for all time steps.  This should be 
        called before ``self.takeStep()`` within a typical time stepping 
        loop.
        """
        # Only the master task writes shell restarts.
        if(mpirank==0):
            for i in range(self.num_splines):
                f = HDF5File(selfcomm, self.shellRestartName(
                    restartPath,i,timeStep), "w")
                f.write(self.y_old_homs[i], "/y_old_hom_"+str(i))
                f.write(self.ydot_old_homs[i], "/ydot_old_hom_"+str(i))
                f.write(self.yddot_old_homs[i], "/yddot_old_hom_"+str(i))
                f.close()
                f = open(self.lamRestartName(restartPath,i,timeStep), "wb")
                save(f,self.lams[i],allow_pickle=False)
                f.close()

        f = HDF5File(worldcomm, 
                     self.fluidRestartName(restartPath,timeStep),"w")
        f.write(self.up_old,"/up_old")
        f.write(self.updot_old,"/updot_old")
        f.close()
        
    def readRestarts(self,restartPath,timeStep):
        """
        Read in data needed to restart the computation from step ``timeStep``, 
        where ``restartPath`` is the path to a directory containing restart
        files for all time steps.  This should be called prior to the 
        beginning of the time stepping loop, with ``timeStep`` equal to the 
        index of the first time step to be executed.
        """
        # Each task reads a copy of the shell state on its self communicator.
        for i in range(self.num_splines):
            f = HDF5File(selfcomm, 
                         self.shellRestartName(restartPath,i,timeStep),"r")
            f.read(self.y_old_homs[i],"/y_old_hom_"+str(i))
            f.read(self.ydot_old_homs[i],"/ydot_old_hom_"+str(i))
            f.read(self.yddot_old_homs[i],"/yddot_old_hom_"+str(i))
            f.close()
            f = open(self.lamRestartName(restartPath,i,timeStep),"rb")
            self.lams[i] = load(f)
            f.close()

        f = HDF5File(worldcomm, 
                     self.fluidRestartName(restartPath,timeStep),"r")
        f.read(self.up_old,"/up_old")        
        f.read(self.updot_old,"/updot_old")
        f.close()
        
    def takeStep(self):
        """
        Advance the ``CouDALFISh`` by one time step.
        """
        if not self.multiPatch:
            # For single extracted spline case, if the initial condition 
            # (0 s) for Lagrange multiplier ``self.lam`` is changed, change 
            # the list of Lagrange multipliers ``self.lams`` as well. This 
            # code is to keep backward compatibility for existing demo 
            # ``manufacturedSolution``.
            if np.linalg.norm(self.lam - self.lam_old) != 0:
                self.lam_old = np.copy(self.lam)
                self.lams[0] = self.lam

        # Same-velocity predictor:
        for i in range(self.num_splines):
            self.y_homs[i].assign(self.timeInt_shs[i].sameVelocityPredictor())
        self.up.assign(self.timeInt_f.sameVelocityPredictor())

        # Explicit-in-geometry:
        yFuncs = []
        nodexs = []
        for i in range(self.num_splines):
            yFuncs += [Function(self.spline_shs[i].V),]
            yFuncs[i].assign(self.y_alpha_homs[i])
            nodexs += [self.nodeX_shs[i] 
                       + self.contactContext_sh.evalFunction(yFuncs[i],i)]
        self.updateStabScale(nodexs)

        # Block iteration:
        Fnorm_sh0 = -1.0
        for blockIt in range(0,self.blockIts):

            # Assemble the fluid subproblem
            K_f = assemble(self.Dres_f)
            F_f = assemble(self.res_f)

            # Add fluid's FSI coupling terms
            upFunc = Function(self.V_f)
            upFunc.assign(self.up_alpha)
            nodeus = self.evalFluidVelocities(upFunc, nodexs)

            ydotFuncs = []
            nodeydots = []
            for i in range(self.num_splines):
                ydotFuncs += [Function(self.spline_shs[i].V),]
                ydotFuncs[i].assign(self.ydot_alpha_homs[i])
                nodeydots += [self.contactContext_sh
                              .evalFunction(ydotFuncs[i],i)]

            self.updateNodalNormals()
            nodens = []
            for i in range(self.num_splines):
                nodens += [self.contactContext_sh
                           .evalFunction(self.n_nodal_shs[i],i)]
            self.addFluidCouplingForces(nodeydots, nodeus, nodexs, nodens,
                                        K_f, F_f)
            # Apply fluid BCs
            for bc in self.bcs_f:
                bc.apply(K_f,F_f)

            # Solve for fluid increment
            dup = Function(self.V_f)
            # Avoids annoying warnings about non-convergence:
            self.fluidLinearSolver.ksp().setOperators(A=as_backend_type(K_f)
                                                      .mat())
            self.fluidLinearSolver.ksp().solve(as_backend_type(F_f).vec(),
                                               as_backend_type(dup.vector())
                                               .vec())
            self.fluidLinearSolver.ksp().reset()
            self.up.assign(self.up - dup)

            # Assemble the structure subproblem
            if self.nonmatching_sh is not None:
                Ks_FE, Fs_FE = self.nonmatching_sh.assemble_nonmatching()
             
            else:
                Fs_FE = [None for i in range(self.num_splines)]
                Ks_FE = [[None for i in range(self.num_splines)]
                         for j in range(self.num_splines)]
                for i in range(self.num_splines):
                    Fs_FE[i] = as_backend_type(assemble(self.res_shs[i]))\
                               .vec()
                    Ks_FE[i][i] = as_backend_type(assemble(self.Dres_shs[i]))\
                                  .mat()

            # Next, add on the contact contributions, assembled using the
            # function defined above.
            if self.includeContact_sh is True:
                yFuncs = []
                for i in range(self.num_splines):
                    yFuncs += [Function(self.spline_shs[i].V),]
                    yFuncs[i].assign(self.y_alpha_homs[i])
                Kcs_FE, Fcs_FE = self.contactContext_sh\
                                 .assembleContact(yFuncs, output_PETSc=True)
                if self.multiPatch:
                    for i in range(self.num_splines):
                        if Fcs_FE[i] is not None:
                            Fs_FE[i] += Fcs_FE[i]
                        for j in range(self.num_splines):
                            if Ks_FE[i][j] is None:
                                Ks_FE[i][j] = Kcs_FE[i][j]
                            elif Ks_FE[i][j] is not None and \
                                Kcs_FE[i][j] is not None:
                                Ks_FE[i][j] += Kcs_FE[i][j]
                else:
                    if Fcs_FE is not None:
                        Fs_FE[0] += Fcs_FE
                        Ks_FE[0][0] += Kcs_FE

            # Add the structure's FSI coupling forces
            upFunc = Function(self.V_f)
            upFunc.assign(self.up_alpha)
            nodeus = self.evalFluidVelocities(upFunc, nodexs)
            self.addShellCouplingForces(nodeydots, nodeus, nodens, 
                                        Ks_FE, Fs_FE)

            # Apply the extraction to an IGA function space.  (This applies
            # the Dirichlet BCs on the IGA unknowns.)
            MTF_sh_list = [None for i in range(self.num_splines)]
            MTKM_sh_list = [[None for i in range(self.num_splines)]
                            for j in range(self.num_splines)]
            for i in range(self.num_splines):
                MTF_sh_list[i] = as_backend_type(self.spline_shs[i].\
                                 extractVector(PETScVector(Fs_FE[i]))).vec()
                for j in range(self.num_splines):
                    if i == j:
                        MTKM_sh_list[i][i] = as_backend_type(
                            self.spline_shs[i].extractMatrix(
                            PETScMatrix(Ks_FE[i][i]))).mat()
                    else:
                        if Ks_FE[i][j] is not None:
                            MTKM_sh_list[i][j] = as_backend_type(
                                self.spline_shs[i].M).mat().transposeMatMult(
                                Ks_FE[i][j]).matMult(as_backend_type(
                                self.spline_shs[j].M).mat())

                            MTKM_sh_list[i][j].zeroRows(
                                self.spline_shs[i].zeroDofs, diag=0)
                            MTKM_sh_list[i][j].transpose()
                            MTKM_sh_list[i][j].zeroRows(
                                self.spline_shs[j].zeroDofs, diag=0)
                            MTKM_sh_list[i][j].transpose()
                            MTKM_sh_list[i][j].assemblyBegin()
                            MTKM_sh_list[i][j].assemblyEnd()

            if self.multiPatch:
                MTF_sh = PETSc.Vec(self.spline_shs[0].comm)
                MTF_sh.createNest(MTF_sh_list, comm=self.spline_shs[0].comm)
                MTF_sh.setUp()
                MTF_sh.assemble()
                MTKM_sh = PETSc.Mat(self.spline_shs[0].comm)
                MTKM_sh.createNest(MTKM_sh_list, comm=self.spline_shs[0].comm)
                MTKM_sh.setUp()
                MTKM_sh.assemble()
                MTKM_sh.convert('seqaij')
            else:
                MTF_sh = MTF_sh_list[0]
                MTKM_sh = MTKM_sh_list[0][0]

            # Check the nonlinear residual.
            Fnorm_sh = MTF_sh.norm()
            Fnorm_f = norm(F_f)

            if(blockIt==0):
                Fnorm_f0 = Fnorm_f
            relNorm_f = Fnorm_f/Fnorm_f0
            if(Fnorm_sh0 < 0.0 and Fnorm_sh > DOLFIN_EPS):
                Fnorm_sh0 = Fnorm_sh
                
            # This condition is to catch cases where there is an
            # impulsive change in time-dependent data, which leads to
            # a sudden increase in the shell residual from the first to
            # second iteration, such that convergence relative to the
            # initial shell residual will lead to extreme over-solving.
            if(Fnorm_sh > Fnorm_sh0 and blockIt > 0):
                if(mpirank==0):
                    print("  ........... "
                          +"*** NOTE: Shell convergence criterion reset ***")
                Fnorm_sh0 = Fnorm_sh
                
            if(Fnorm_sh0 > 0.0):
                relNorm_sh = Fnorm_sh/Fnorm_sh0
            else:
                relNorm_sh = 0.0

            if(mpirank==0):
                print("  ....... Block iteration "+str(blockIt+1)+" :")
                print("  ........... Shell relative res. = "+str(relNorm_sh))
                print("  ........... Fluid relative res. = "+str(relNorm_f))

            # Solve for the nonlinear increment, and add it to the current
            # solution guess.  (Applies BCs)
            if self.multiPatch:
                dy_list = []
                dy_IGA_list = []
                for i in range(len(self.spline_shs)):
                    dy_list += [Function(self.spline_shs[i].V),]
                    dy_IGA = PETSc.Vec(self.spline_shs[i].comm)
                    dy_IGA.createSeq(self.spline_shs[i].M.size(1), 
                                     comm=self.spline_shs[i].comm)
                    dy_IGA.setUp()
                    dy_IGA.assemble()
                    dy_IGA_list += [dy_IGA,]

                dy = PETSc.Vec(self.spline_shs[0].comm)
                dy.createNest(dy_IGA_list, comm=self.spline_shs[0].comm)
                dy.setUp()
                dy.assemble()
            else:
                dyFunc = Function(self.spline_shs[0].V)
                dy = PETSc.Vec(self.spline_shs[i].comm)
                dy.createSeq(self.spline_shs[i].M.size(1), 
                             comm=self.spline_shs[i].comm)
                dy.setUp()
                dy.assemble()

            solve(PETScMatrix(MTKM_sh), PETScVector(dy), PETScVector(MTF_sh))

            if self.multiPatch:
                for i in range(self.num_splines):
                    self.spline_shs[i].M.mat().mult(dy_IGA_list[i], 
                                                    dy_list[i].vector().vec())
                    as_backend_type(dy_list[i].vector()).vec().ghostUpdate()
                    as_backend_type(dy_list[i].vector()).vec().assemble()
                    self.y_homs[i].assign(self.y_homs[i] - dy_list[i])
                # Update mortar meshes' functions
                if self.nonmatching_sh is not None:
                    for i in range(len(
                        self.nonmatching_sh.transfer_matrices_list)):
                        for j in range(len(
                            self.nonmatching_sh.transfer_matrices_list[i])):
                            for k in range(len(self.nonmatching_sh.\
                                transfer_matrices_list[i][j])):
                                self.nonmatching_sh.transfer_matrices_list\
                                    [i][j][k].mat().mult(self.y_homs\
                                    [self.nonmatching_sh.mapping_list[i][j]].\
                                    vector().vec(), 
                                    self.nonmatching_sh.mortar_vars[i][j][k].\
                                    vector().vec())
            else:
                self.spline_shs[0].M.mat().mult(dy, dyFunc.vector().vec())
                self.y_homs[0].assign(self.y_homs[0] - dyFunc)

            if(relNorm_sh < self.blockItTol
               and relNorm_f < self.blockItTol):
                break
            if(blockIt == self.blockIts-1):
                if(mpirank==0):
                    print("ERROR: Block iteration diverged.")
                exit()

        ydotFuncs = [Function(spline.V) for spline in self.spline_shs]
        for i in range(self.num_splines):
            ydotFuncs[i].assign(self.ydot_alpha_homs[i])
            nodeydots[i] = self.contactContext_sh.evalFunction(ydotFuncs[i],i)

        self.updateNodalNormals()
        for i in range(self.num_splines):
            nodens[i] = self.contactContext_sh.\
                        evalFunction(self.n_nodal_shs[i],i)
        self.updateMultipliers(nodeydots, nodeus, nodens)

        self.timeInt_f.advance()
        for i in range(self.num_splines):
            self.timeInt_shs[i].advance()