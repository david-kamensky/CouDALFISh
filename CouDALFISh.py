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

from ShNAPr.kinematics import *
from ShNAPr.contact import *
from mpi4py import MPI as pyMPI
from numpy import zeros, save, load
from tIGAr.timeIntegration import *

# This module assumes 3D problems:
d = 3

def stabScale(cutFunc,eps):
    """
    Returns a UFL ``Conditional`` that is 1 everywhere except where
    ``cutFunc`` is nonzero, where it is ``eps``.  This is intended for
    use in modifying stabilization parameters in elements where
    ``cutFunc`` indicates that there is a large irrotation source term
    (e.g., from an immersed boundary).  
    """
    return conditional(gt(abs(cutFunc),0.0),eps,1.0)

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
                 contactContext_sh=None,
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
        self.mesh_f = mesh_f
        self.timeInt_f = timeInt_f
        self.fluidFac = float(self.timeInt_f.ALPHA_F)
        self.up = self.timeInt_f.x
        self.up_old = self.timeInt_f.x_old
        self.updot_old = self.timeInt_f.xdot_old
        self.up_alpha = self.timeInt_f.x_alpha()
        self.V_f = self.up.function_space()
        self.spline_sh = spline_sh
        self.penalty = penalty
        self.Dt = self.timeInt_f.DELTA_T
        self.timeInt_sh = timeInt_sh
        if(self.timeInt_sh.systemOrder == 2):
            fac = float(self.timeInt_sh.ALPHA_F)*float(self.timeInt_sh.GAMMA)\
                  /(float(self.timeInt_sh.BETA)*float(self.Dt))
        else:
            fac = float(self.timeInt_sh.ALPHA_F)\
                  /(float(self.timeInt_sh.GAMMA)*float(self.Dt))
        self.shellFac = fac
        self.blockIts = blockIts
        self.blockItTol = blockItTol
        self.bcs_f = bcs_f
        if(contactContext_sh==None):
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
        self.nNodes_sh = self.contactContext_sh.nNodes
        self.nodeX_sh = self.contactContext_sh.nodeX
        self.r = r
        self.lam = zeros(self.nNodes_sh)
        self.res_f = res_f
        self.res_sh = res_sh
        self.y_hom = self.timeInt_sh.x
        self.y_old_hom = self.timeInt_sh.x_old
        self.y_alpha_hom = self.timeInt_sh.x_alpha()
        self.ydot_old_hom = self.timeInt_sh.xdot_old
        self.yddot_old_hom = self.timeInt_sh.xddot_old
        self.ydot_hom = self.timeInt_sh.xdot()
        self.ydot_alpha_hom = self.timeInt_sh.xdot_alpha()
        if(Dres_sh==None):
            self.Dres_sh = derivative(self.res_sh,self.y_hom)
        else:
            self.Dres_sh = Dres_sh
        if(Dres_f==None):
            self.Dres_f = derivative(self.res_f,self.up)
        else:
            self.Dres_f = Dres_f
        self.x_sh = self.spline_sh.F \
                    + self.spline_sh.rationalize(self.y_alpha_hom)
        
        if(cutFunc==None):
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
        _,_,self.a2_sh,_,_,_ = surfaceGeometry(self.spline_sh,self.x_sh)
        
        # Nodal normal field:
        self.n_nodal_sh = Function(self.spline_sh.V)

        self.fluidLinearSolver = fluidLinearSolver
        
    def updateNodalNormals(self):
        """
        This projects the normal vector for the shell's current configuration
        onto the (homogeneous) spline space for the displacement via a lumped
        ``$L^2$`` projection.
        """
        self.n_nodal_sh.assign(self.spline_sh.project(self.a2_sh,
                                                      lumpMass=True,
                                                      rationalize=False))
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
        
    def evalFluidVelocities(self,up,nodex):
        """
        Given an array of points, ``nodex``, this returns a same-shape array
        of velocity evaluations from the fluid solution space function ``up``.
        The array returned is the same on all MPI tasks.  For points from 
        ``nodex`` that are not on any task's mesh partition, the corresponding
        velocity is zero.
        """
        uFe = zeros((self.nNodes_sh,d))
        for i in range(0,self.nNodes_sh):
            uFe[i,:] = self.evalFluidVelocity(up,nodex[i,:])
        uFe = worldcomm.allreduce(uFe,op=pyMPI.SUM)
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
    
    def addFluidCouplingForces(self, u_sh, u_f, nodex, n, A, b):
        """
        Given arrays of shell structure velocities ``u_sh``, fluid velocities,
        ``u_f``, positions, ``nodex``, and normals ``n`` at FE nodes of the
        shell structure's mesh, compute coupling forces and apply them to
        fluid LHS and RHS matrix and vector ``A`` and ``b``.  
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
            for node in range(0,self.nNodes_sh):

                # Quadrature weight
                w = self.contactContext_sh.quadWeights[node]

                # Get node's contribution to total force
                dF = self.couplingForceOnFluid(u_f[node,:], u_sh[node,:],
                                               self.lam[node], n[node,:], w)
                for j in range(0,d):
                    rhsPointSourceData[j] += [(Point(nodex[node]),-dF[j]),]
                    # Factor of $\alpha_f$ for time integrator:
                    lhsPointSourceData[j] += [(Point(nodex[node]),
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
                                             addv=PETSc.InsertMode.ADD)

    def addShellCouplingForces(self, u_sh, u_f, n, A, b):
        """
        Given arrays of shell structure velocities ``u_sh``, fluid velocities
        ``u_f``, and normal vectors ``n`` at FE nodes of the shell structure
        mesh, compute coupling forces and add corresponding contributions
        to shell structure LHS and RHS matrix and vector ``A`` and ``b``.
        """
        ADD_MODE = PETSc.InsertMode.ADD
        Am = as_backend_type(A).mat()
        bv = as_backend_type(b).vec()
        lmVec = as_backend_type(Function(self.spline_sh.V).vector()).vec()
        for node in range(0,self.nNodes_sh):
            w = self.contactContext_sh.quadWeights[node]
            dF = self.couplingForceOnFluid(u_f[node,:], u_sh[node,:],
                                           self.lam[node], n[node,:], w)
            for direction in range(0,d):
                index = nodeToDof(node,direction)
                bv.setValue(index,dF[direction],addv=ADD_MODE)
                # Factor for tangent of displacement problem:
                lmVec.setValue(index,self.penalty*self.shellFac*w,
                               addv=ADD_MODE)
        Am.setDiagonal(lmVec,addv=ADD_MODE)
        
    def updateMultipliers(self, u_sh, u_f, n):
        """
        Given arrays of shell structure velocities ``u_sh``, fluid velocities
        ``u_f``, and normal vectors ``n`` at nodes of the shell structure
        FE mesh, update the corresponding FSI Lagrange multiplier samples in
        ``self.lam``.  
        """
        for node in range(0,self.nNodes_sh):
            contrib = 0.0
            for j in range(0,d):
                contrib += self.penalty*(u_f[node,j] - u_sh[node,j])*n[node,j]
            self.lam[node] = (self.lam[node] + contrib)/(1.0 + self.r)

    def updateStabScale(self,nodex):
        """
        Given an array ``nodex`` of immersed structure nodal positions,
        this function updates ``self.cutFunc``, whose nonzero 
        values indicate elements near the immersed structure.  
        """
        self.cutFunc.assign(Function(self.Vscalar_f))
        pointSourceData = []
        if(mpirank == 0):
            for point in range(0,self.nNodes_sh):
                pointSourceData += [(Point(nodex[point]),1.0),]
        multiPointSources = PointSource(self.Vscalar_f,pointSourceData)
        multiPointSources.apply(self.cutFunc.vector())

    # These are mainly for internal use, just to ensure that file names
    # match when reading and writing restart data.
    def fluidRestartName(self,restartPath,i):
        return restartPath+"/restart_f."+str(i)+".h5"
    def shellRestartName(self,restartPath,i):
        return restartPath+"/restart_sh."+str(i)+".h5"
    def lamRestartName(self,restartPath,i):
        return restartPath+"/restart_lam."+str(i)+".npy"

    def writeRestarts(self,restartPath,i):
        """
        Write out all data needed to restart the computation from step ``i``, 
        where ``restartPath`` is the path to a directory containing restart
        files for all time steps.  This should be called before
        ``self.takeStep()`` within a typical time stepping loop.
        """
        # Only the master task writes shell restarts.
        if(mpirank==0):
            f = HDF5File(selfcomm,self.shellRestartName(restartPath,i),"w")
            f.write(self.y_old_hom,"/y_old_hom")
            f.write(self.ydot_old_hom,"/ydot_old_hom")
            f.write(self.yddot_old_hom,"/yddot_old_hom")
            f.close()
            f = open(self.lamRestartName(restartPath,i),"wb")
            save(f,self.lam,allow_pickle=False)
            f.close()

        f = HDF5File(worldcomm,self.fluidRestartName(restartPath,i),"w")
        f.write(self.up_old,"/up_old")
        f.write(self.updot_old,"/updot_old")
        f.close()
        
    def readRestarts(self,restartPath,i):
        """
        Read in data needed to restart the computation from step ``i``, 
        where ``restartPath`` is the path to a directory containing restart
        files for all time steps.  This should be called prior to the 
        beginning of the time stepping loop, with ``i`` equal to the index
        of the first time step to be executed.
        """
        # Each task reads a copy of the shell state on its self communicator.
        f = HDF5File(selfcomm,self.shellRestartName(restartPath,i),"r")
        f.read(self.y_old_hom,"/y_old_hom")
        f.read(self.ydot_old_hom,"/ydot_old_hom")
        f.read(self.yddot_old_hom,"/yddot_old_hom")
        f.close()

        f = open(self.lamRestartName(restartPath,i),"rb")
        self.lam = load(f)
        f.close()
        f = HDF5File(worldcomm,self.fluidRestartName(restartPath,i),"r")
        f.read(self.up_old,"/up_old")        
        f.read(self.updot_old,"/updot_old")
        f.close()
        
    def takeStep(self):
        """
        Advance the ``CouDALFISh`` by one time step.
        """
        # Same-velocity predictor:
        self.y_hom.assign(self.timeInt_sh.sameVelocityPredictor())
        self.up.assign(self.timeInt_f.sameVelocityPredictor())

        # Explicit-in-geometry:
        yFunc = Function(self.spline_sh.V)
        yFunc.assign(self.y_alpha_hom)
        nodex = self.nodeX_sh + self.contactContext_sh.evalFunction(yFunc)
        self.updateStabScale(nodex)

        # Block iteration:
        Fnorm_sh0 = -1.0
        for blockIt in range(0,self.blockIts):

            # Assemble the fluid subproblem
            K_f = assemble(self.Dres_f)
            F_f = assemble(self.res_f)

            # Add fluid's FSI coupling terms
            upFunc = Function(self.V_f)
            upFunc.assign(self.up_alpha)
            nodeu = self.evalFluidVelocities(upFunc,nodex)
            ydotFunc = Function(self.spline_sh.V)
            ydotFunc.assign(self.ydot_alpha_hom)
            nodeydot = self.contactContext_sh.evalFunction(ydotFunc)
            self.updateNodalNormals()
            noden = self.contactContext_sh.evalFunction(self.n_nodal_sh)
            self.addFluidCouplingForces(nodeydot, nodeu, nodex, noden,
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
            K_sh = assemble(self.Dres_sh)
            F_sh = assemble(self.res_sh)

            # Next, add on the contact contributions, assembled using the
            # function defined above.
            if(self.includeContact_sh):
                yFunc = Function(self.spline_sh.V)
                yFunc.assign(self.y_alpha_hom)
                Kc, Fc = self.contactContext_sh.assembleContact(yFunc)
                K_sh += Kc
                F_sh += Fc

            # Add the structure's FSI coupling forces
            upFunc = Function(self.V_f)
            upFunc.assign(self.up_alpha)
            nodeu = self.evalFluidVelocities(upFunc,nodex)
            self.addShellCouplingForces(nodeydot, nodeu, noden, K_sh, F_sh)

            # Apply the extraction to an IGA function space.  (This applies
            # the Dirichlet BCs on the IGA unknowns.)
            MTKM_sh = self.spline_sh.extractMatrix(K_sh)
            MTF_sh = self.spline_sh.extractVector(F_sh)

            # Check the nonlinear residual.
            Fnorm_sh = norm(MTF_sh)
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
            dy_hom = Function(self.spline_sh.V)
            self.spline_sh.solveLinearSystem(MTKM_sh,MTF_sh,dy_hom)
            self.y_hom.assign(self.y_hom-dy_hom)

            if(relNorm_sh < self.blockItTol
               and relNorm_f < self.blockItTol):
                break
            if(blockIt == self.blockIts-1):
                if(mpirank==0):
                    print("ERROR: Block iteration diverged.")
                exit()

        # Update Lagrange multiplier using most recent structure solution.
        ydotFunc = Function(self.spline_sh.V)
        ydotFunc.assign(self.ydot_alpha_hom)
        nodeydot = self.contactContext_sh.evalFunction(ydotFunc)
        self.updateNodalNormals()
        noden = self.contactContext_sh.evalFunction(self.n_nodal_sh)
        self.updateMultipliers(nodeydot, nodeu, noden)

        # Move to the next time step.
        self.timeInt_f.advance()
        self.timeInt_sh.advance()
