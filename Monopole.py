#This model is used to verify monopole antenna radiation pattern

import sys
import numpy as np
from mpi4py import MPI as nMPI
from petsc4py import PETSc
import gmsh
import meshio
import ufl
import basix.ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io import gmshio
from dolfinx import fem, default_scalar_type
from dolfinx import io
from dolfinx.fem.petsc import LinearProblem

class Hinc:
    def __init__(self, x0, y0, H):
        self.x0 = x0 
        self.y0 = y0
        self.H = H
    def eval(self, x):
        hx = -self.H * (x[1] - self.y0) / (2.0 * np.pi * ((x[0] - self.x0)**2 + (x[1] - self.y0)**2))
        hy = self.H * (x[0] - self.x0) / (2.0 * np.pi * ((x[0] - self.x0)**2 + (x[1] - self.y0)**2))
        hz = np.full_like(hy, 0.0+0.0j, dtype=np.complex128)
        return(hx, hy, hz)
    

comm = nMPI.COMM_WORLD
mpiRank = comm.rank
modelRank = 0

def Model(x):

    rs = 12.0 # Radiation sphere radius
#    h = 2.0 # antenna height
    h = 3.0

    lc = 1.0 # coax length
    rc = 0.08625 # coax shield radius
    cc = 0.0375  # coax center conductor radius

    eta0 = 377.0
    eps = 1.0e-4 # Geom tolerance

    k0 = 0.225 * 2.0 * np.pi / h
    lm = 0.8 # background mesh density
    le = 0.15 # edge mesh density
    ls = 0.02 # center pin mesh density

    SymmType = 1
    NoPattern = 1

    if mpiRank == modelRank:
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.model.add("Monopole Over Ground Plane")
# radiation sphere 1 truncated to 3
        gmsh.model.occ.addSphere(0, 0, 0, rs, 1)
        gmsh.model.occ.addBox(0, -rs, 0, rs, 2*rs, 2*rs, 2)
        gmsh.model.occ.intersect([(3,1)], [(3,2)], 3, removeObject=True, removeTool=True)
# coax shield 3, center 4 and monopole antenna of height h
        xc = 0.0
        yc = 0.0
        gmsh.model.occ.addCylinder(xc, yc, -lc, 0, 0, lc, rc, 4, 2*np.pi)
        gmsh.model.occ.addCylinder(xc, yc, -lc, 0, 0, lc+h, cc, 5, 2*np.pi)
# Unify coax shield and rad sphere
        gmsh.model.occ.fuse([(3,3)],[(3,4)], 6, removeObject=True, removeTool=True)
# cut out center conductor/antenna
        gmsh.model.occ.cut([(3,6)],[(3,5)], 7, removeObject=True, removeTool=True)
# truncate rad sphere deleted
        gmsh.model.occ.addBox(0, -rs, -rs, rs, 2*rs, 2*rs, 8)
        gmsh.model.occ.intersect([(3,7)], [(3,8)], 9, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()  
        # set background mesh density
        pt = gmsh.model.getEntities(0)
        gmsh.model.mesh.setSize(pt, lm)
        pt = gmsh.model.getEntitiesInBoundingBox(-eps, -rc-eps, -lc-eps, rc+eps, rc+eps, h+eps)
        gmsh.model.mesh.setSize(pt, ls)
        gmsh.model.occ.synchronize()  
                
        pec = []
        symm = []
        rad = []
        inp = []
        bb = gmsh.model.getEntities(dim=3)
        pt = gmsh.model.getBoundary(bb, combined=True, oriented=False, recursive=False)
        for bx in pt:
            CoM = gmsh.model.occ.getCenterOfMass(bx[0], bx[1])
            if (np.abs(CoM[0]) < eps) and SymmType == 1: # Set to PMC if Symmtype is 1
                symm.append(bx[1])
            elif np.abs(CoM[2] + lc) < eps:
                inp.append(bx[1])
            elif (CoM[2] > eps) and (CoM[0] - rc  > eps):
                rad.append(bx[1])
            else:
                pec.append(bx[1])
        print(pec, symm, rad, inp)


        gmsh.model.addPhysicalGroup(3, [9], 1)
        gmsh.model.setPhysicalName(3, 1, "Rad Region")
        gmsh.model.addPhysicalGroup(2, pec, 1)
        gmsh.model.setPhysicalName(2, 1, "PEC")
        gmsh.model.addPhysicalGroup(2, symm, 2)
        gmsh.model.setPhysicalName(2, 2, "SYMM")
        gmsh.model.addPhysicalGroup(2, rad, 3)
        gmsh.model.setPhysicalName(2, 3, "Rad")
        gmsh.model.addPhysicalGroup(2, inp, 4)
        gmsh.model.setPhysicalName(2, 4, "Input")

        gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01)
        gmsh.option.setNumber('Mesh.MeshSizeMax', 1.2)
        gmsh.option.setNumber('Mesh.Algorithm', 6) #1=Mesh Adapt, 2=Auto, 3=Initial mesh only, 5=Delaunay, 6=Frontal-Delaunay
        gmsh.option.setNumber('Mesh.MinimumCirclePoints', 36)
        gmsh.option.setNumber('Mesh.CharacteristicLengthFromCurvature', 1)
        gmsh.option.setNumber("Mesh.Tetrahedra", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 10)
        
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Gmsh")

        gmsh.fltk.run()

# mesh = 3d topology
# ct = volume tags
# fm = surface tags
    mesh, ct, fm = gmshio.model_to_mesh(gmsh.model, comm, modelRank, gdim=3)
    if modelRank == mpiRank:
        gmsh.finalize()

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    elem = basix.ufl.element('Nedelec 1st kind H(curl)', mesh.basix_cell(), degree=2)
    V = fem.functionspace(mesh, elem)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

# Print out BC tags
#    with io.XDMFFile(mesh.comm, "BCs.xdmf", "w") as xx:
#        xx.write_mesh(mesh)
#        xx.write_meshtags(fm, mesh.geometry)

# Dirichlet BCs
    facets = fm.find(1) # PEC facets
    ubc = fem.Function(V)
    dofs = fem.locate_dofs_topological(V, mesh.topology.dim-1, facets)
    ubc.interpolate(lambda x: np.array([0*x[0], 0*x[1], 0*x[2]], dtype=np.complex128))
    bc = fem.dirichletbc(ubc, dofs)

    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=fm)
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    num_dofs_global = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"Number of dofs (owned) by rank {mpiRank}: {num_dofs_local}")
    if mpiRank == modelRank:
        print(f"Number of dofs global: {num_dofs_global}")
# Build RHS
    uh = fem.Function(V, dtype=np.complex128)
    f = Hinc(0.0, 0.0, 1.0+0j)
    uh.interpolate(f.eval)
    finc = 2.0j * k0 * eta0 * ufl.inner(ufl.cross(n, uh), v) * ds(4)
    uh.name = "Hinc"

# Weak form
    L = (ufl.inner(ufl.curl(u), ufl.curl(v)) - k0 * k0 * ufl.inner(u, v)) * ufl.dx
# Outgoing at inport
    bc1 = 1j * k0 * ufl.inner(ufl.cross(n, u), ufl.cross(n, v)) * ds(4)
# Radiation
#    bc2 = ((1.0j * k0 - 1.0 / rs) * ufl.inner(ufl.cross(n, u), ufl.cross(n, v))) * ds(3)
    bc2 = 1j * k0 * ufl.inner(ufl.cross(n, u), ufl.cross(n, v)) * ds(3) 
    beta = 1.0 / (2j * k0 + 2 / rs)
    bc3 = beta * ufl.inner(ufl.dot(n, ufl.curl(u)), ufl.dot(n, ufl.curl(v))) * ds(3)
    bc4 = beta * ufl.inner(ufl.div(ufl.cross(n,u)), ufl.div(ufl.cross(n,v))) * ds(3)
    L += bc1 + bc2 + bc3 - bc4
# Solve system of equations
    Lin_system = LinearProblem(L, finc, bcs=[bc], petsc_options={"ksp_type":"preonly", "pc_type":"lu"})
    E = Lin_system.solve()
    lu_solver = Lin_system.solver
    lu_solver.view()
    
# Plot baby, plot!
#    Vw = basix.ufl.element('DG', mesh.basix_cell(), 0, shape=(mesh.geometry.dim, ))
#    W = fem.functionspace(mesh, Vw)
#    Et = fem.Function(W)
#    Et.interpolate(E)
#    Et.name = "ElectricField"
#    with io.XDMFFile(mesh.comm, "Efield_{0}_{1}.xdmf".format(0, SymmType), "w") as xx:
#        xx.write_mesh(mesh)
#        xx.write_function(Et)
#    H_expr = fem.Expression(ufl.curl(E), W.element.interpolation_points())
#    Et.interpolate(H_expr)
#    Et.name = "MagneticField"
 #   with io.XDMFFile(mesh.comm, "Hfield_{0}_{1}.xdmf".format(0, SymmType), "w") as xx:
#        xx.write_mesh(mesh)
#        xx.write_function(Et)
        
    Prad = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(E, ufl.cross(ufl.curl(E), n)) * ds(3))) / (2.0j * k0 * eta0), op=nMPI.SUM)
    Pinc = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds(4))) * eta0 / 2.0 , op=nMPI.SUM)
    Pref = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(E, ufl.cross(ufl.curl(E), n)) * ds(4))) / (-2.0j * k0 * eta0), op=nMPI.SUM)

# Generate reflection coefficient at feed
    Rx = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(E - eta0 * ufl.cross(n, uh), ufl.cross(n, uh)) * ds(4))), op=nMPI.SUM)
    Dx = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds(4))), op=nMPI.SUM) * eta0 
     
#        print(Rx / Dx)
 
    Rho = Rx * np.exp(2j*k0*lc)/ Dx
    Zin = 50.0 * (1.0 + Rho) / (1.0 - Rho)
    if mpiRank == 0:
        print("Pinc = {0}, Pref = {1}, Prad = {2}".format(Pinc, np.real(Pinc-Pref), np.real(Prad)))
        print("k0 = {0}, Symm = {1}".format(k0, SymmType))
        print("Gamma at feed = {0}, Gamma at coax input = {1}".format(Rx * np.exp(2j*k0*lc)/ Dx, Rx / Dx))
        print("Zin at feed = {0}, Zin at coax input = {1}".format(Zin, 50.0 * (1 + Rx/Dx)/(1 - Rx/Dx)))

    if NoPattern == 0:
        return np.real(Pinc - Pref)
# Scalar "phase" function space
    Q = fem.functionspace(mesh, ('CG', 2))
    vv = fem.Function(Q)
    S = fem.Function(V)
    T = fem.Function(V)
    U = fem.Function(V)

# Generate integral over quarter sphere surface, using symmetry to get value over full hemisphere surface for z>0
    Ex = np.zeros(4, dtype=np.complex128)
    Ey = np.zeros(4, dtype=np.complex128)
    Ez = np.zeros(4, dtype=np.complex128)
    Hx = np.zeros(4, dtype=np.complex128)
    Hy = np.zeros(4, dtype=np.complex128)
    Hz = np.zeros(4, dtype=np.complex128)
# The field symmetry conditions for E and H on the PEC ground plane and PMC symmetry wall
    PMCsymmE = np.array([[-1, 0, 0],[0, 1, 0],[0, 0, 1]])
    PECsymmE = np.array([[-1, 0, 0],[0, -1 ,0],[0 ,0 ,1]])
    PMCsymmH = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])
    PECsymmH = np.array([[1, 0, 0],[0, 1 ,0],[0 ,0 ,-1]])

    fp = open("Pattern1.txt", "w")
#Loop over theta-phi angles
    for p in range(-26, 26, 1):
        theta = np.pi * p / 50
        for q in range(1):
           phi = np.pi * q / 50
# Transformation from rectangular to spherical coordinates
           Rot = np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)],\
                [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],\
                [-np.sin(phi), np.cos(phi), 0.0]])   
# Observation vector at position theta, phi (usual sphetrical coords)
           rr = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
# S, T, U contains constant unit vectors along x-y-z axes for integration
           S.interpolate(lambda x: (1.0 + 0.0 * x[0], x[1] * 0.0, 0.0 * x[2]))
           T.interpolate(lambda x: (0.0 * x[0], 1.0 + x[1] * 0.0, 0.0 * x[2]))
           U.interpolate(lambda x: (0.0 * x[0], 0.0 * x[1], 1.0 + 0.0 * x[2]))
# This is the scalar phase term
           vv.interpolate(lambda x: np.exp(1j * k0 * (rr[0] * x[0] + rr[1] * x[1] + rr[2] * x[2])))
# Add to integrand original quarter sphere for x>0, z>0 for all three components of current sources
           Hx[0] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), S) * ds(3))), op = nMPI.SUM)
           Hy[0] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), T) * ds(3))), op = nMPI.SUM)
           Hz[0] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), U) * ds(3))), op = nMPI.SUM)
           Ex[0] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), S) * ds(3))), op = nMPI.SUM)
           Ey[0] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), T) * ds(3))), op = nMPI.SUM)
           Ez[0] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), U) * ds(3))), op = nMPI.SUM)
# The other quarter sphere using symmetry.  Phase factor for x < 0, z > 0. PMC symmetry
           vv.interpolate(lambda x: np.exp(1j * k0 * (-rr[0] * x[0] + rr[1] * x[1] + rr[2] * x[2]))) 
# Add to integrand    
           Hx[1] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), S) * ds(3))), op = nMPI.SUM)
           Hy[1] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), T) * ds(3))), op = nMPI.SUM) 
           Hz[1] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), U) * ds(3))), op = nMPI.SUM)
           Ex[1] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), S) * ds(3))), op = nMPI.SUM)
           Ey[1] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), T) * ds(3))), op = nMPI.SUM)
           Ez[1] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), U) * ds(3))), op = nMPI.SUM)
# The third quarter sphere using symmetry.  Phase factor for x > 0, z < 0. PEC symmetry
           vv.interpolate(lambda x: np.exp(1j * k0 * (rr[0] * x[0] + rr[1] * x[1] - rr[2] * x[2])))  
# Add to integrand 
           Hx[2] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), S) * ds(3))), op = nMPI.SUM)
           Hy[2] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), T) * ds(3))), op = nMPI.SUM) 
           Hz[2] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), U) * ds(3))), op = nMPI.SUM)
           Ex[2] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), S) * ds(3))), op = nMPI.SUM)
           Ey[2] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), T) * ds(3))), op = nMPI.SUM)
           Ez[2] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), U) * ds(3))), op = nMPI.SUM)
# The last quarter sphere using symmetry.  Phase factor for x < 0, z < 0. PEC and PMC symmetry
           vv.interpolate(lambda x: np.exp(1j * k0 * (-rr[0] * x[0] + rr[1] * x[1] - rr[2] * x[2]))) 
# Add to integrand  
           Hx[3] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), S) * ds(3))), op = nMPI.SUM)
           Hy[3] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), T) * ds(3))), op = nMPI.SUM) 
           Hz[3] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,E), U) * ds(3))), op = nMPI.SUM) 
           Ex[3] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), S) * ds(3))), op = nMPI.SUM)
           Ey[3] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), T) * ds(3))), op = nMPI.SUM)
           Ez[3] = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(vv * ufl.cross(n,ufl.curl(E)), U) * ds(3))), op = nMPI.SUM)
#     
# Do integration of the four sphere slices over quarter sphere surface.  Ev should contain the azimuthal component of total field at point theta,phi
           Erad =\
              -1j * k0 * np.matmul(Rot, np.cross(rr, np.array([Hx[0], Hy[0], Hz[0]]) + np.matmul(PMCsymmH , np.array([Hx[1], Hy[1], Hz[1]])) +\
              np.matmul(PECsymmH , np.array([Hx[2], Hy[2], Hz[2]])) +\
              np.matmul(PECsymmH , np.matmul(PMCsymmH , np.array([Hx[3], Hy[3], Hz[3]]))))) +\
              np.matmul(Rot, np.array([Ex[0], Ey[0], Ez[0]]) + np.matmul(PMCsymmE , np.array([Ex[1], Ey[1], Ez[1]])) +\
              np.matmul(PECsymmE , np.array([Ex[2], Ey[2], Ez[2]])) +\
              np.matmul(PECsymmE , np.matmul(PMCsymmE , np.array([Ex[3], Ey[3], Ez[3]]))))
# Print out result
           Gvert = np.absolute(Erad[1])**2 * np.pi / ( 27.0 * np.pi * np.absolute(Prad * 2 * eta0))
           Ghoriz = np.absolute(Erad[2])**2 * np.pi / (27.0 * np.pi * np.absolute(Prad * 2 * eta0)) # 27*pi is normalization constant
           if mpiRank == 0:
               print(theta, phi, Gvert, Ghoriz, Erad[1], Erad[2])        
           print("{0} {1} {2} {3}".format(theta, phi, np.absolute(Gvert), np.absolute(Ghoriz)), file=fp)
               
    fp.close()
    return np.real(Pinc -Pref)

r = Model([0, 0])
if mpiRank == 0:
    print(r)
sys.exit(0)



