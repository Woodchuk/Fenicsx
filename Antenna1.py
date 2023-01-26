# Generate basic cavity-backed bowtie antenna.

import sys
import numpy as np
from scipy import optimize as opt
from mpi4py import MPI as nMPI
from petsc4py import PETSc
import gmsh
import meshio
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.io import gmshio
from dolfinx import fem

def Model(x):
   comm = nMPI.COMM_WORLD
   mpiRank = comm.rank
   modelRank = 0 # Where GMSH runs
   h = x[0] # cavity depth
   r = x[1] # cavity radius
   rgp = x[2] # Ground plane (and rad sphere) radius
   bl = x[3] # Bowtie length
   bt = x[4] # Metal thickness
   ba = x[5] # Bowtie width
   bx = x[6] # x offset of bowtie
   hd = x[7] # Dielectric thickness
   k0 = x[8] # Wavenumber
   eps = x[9] # Dk of dielectric
   lc = x[10] # Coax len
   rc = x[11] # coax shield rad
   rcc = x[12] # coax center cond rad
   rco = x[13] # coax center cond offset
   eta0 = 377.0
   mmin = 0.02
   mmax = 1.0
   err = 1.0e-3
   lm = 0.5
   ls = 0.025
   lq = 0.1

   if mpiRank == modelRank:
       print("Cavity height = {0:<f}, Cavity radius = {1:<f}, Rad shpere radius = {2:<f}".format(h, r, rgp))
       print("Bowtie Len = {0:<f}, Bowtie metal thickness = {1:<f}, Bowtie width = {2:<f}".format(bl, bt, ba))
       print("Bowtie x offset = {0:<f}, Diel thickness = {1:<f}, Wavenumber = {2:<f}, Dk = {3:<f}".format(bx, hd, k0, eps))

       gmsh.initialize()
       gmsh.option.setNumber('General.Terminal', 0)
       gmsh.model.add("BowtieAnt")
       gmsh.model.setCurrent("BowtieAnt")

       RadSphere = gmsh.model.occ.addSphere(0, 0, 0, rgp, 1)
       Box1 = gmsh.model.occ.addBox(0, -rgp, 0, rgp, 2*rgp, rgp, 2)
       PCB = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, hd, r, 7, angle=np.pi)
       Cavity = gmsh.model.occ.addCylinder(0, 0, -h, 0, 0, h, r, 8, angle=np.pi)
       CoaxShield = gmsh.model.occ.addCylinder(rco, 0, -h-lc, 0, 0, lc, rc, 9, angle=2*np.pi)
       CoaxCenter = gmsh.model.occ.addCylinder(rco, 0, -h-lc, 0, 0, h+lc+hd+bt/2, rcc, 10, angle=2*np.pi)
       gmsh.model.occ.rotate([(3,7),(3,8)], 0, 0, 0, 0, 0, 1, -np.pi/2)
       RadZone = gmsh.model.occ.intersect([(3,1)], [(3,2)], 3, removeObject=True, removeTool=True)
       print(RadZone)
       Bowtie1= gmsh.model.occ.addWedge(0, 0, hd, bl, ba, bt, 4, ltx=0)
       gmsh.model.occ.mirror([(3,4)], 0, 1, 0, 0)
       Bowtie2= gmsh.model.occ.addWedge(0, 0, hd, bl, ba, bt, 5, ltx=0)
       Bowtie = gmsh.model.occ.fuse([(3,4)], [(3,5)], 6, removeObject=True, removeTool=True)
       gmsh.model.occ.rotate([(3,6)], 0, 0, 0, 0, 0, 1, np.pi)
       gmsh.model.occ.translate([(3,6)], bl+bx, 0, 0) # Bowtie metal finished
       Metal = gmsh.model.occ.fuse([(3,6)], [(3,10)], 11, removeObject=True, removeTool=True)

       RadZoneFinal = gmsh.model.occ.cut([(3,3)], [(3,11)], 12, removeObject=True, removeTool=False)
       PCBFinal = gmsh.model.occ.cut([(3,7)],[(3,11)], 13, removeObject=True, removeTool=False)
       CavityAndCoax = gmsh.model.occ.fuse([(3,8)], [(3,9)], 14, removeObject=True, removeTool=True)
       CavityAndCoaxFinal = gmsh.model.occ.cut([(3,14)], [(3,11)], 15, removeObject=True, removeTool=True)
       gmsh.model.occ.fragment([(3,12),(3,13),(3,15)], [], -1)

       gmsh.model.occ.synchronize()
       gmsh.option.setNumber('Mesh.MeshSizeMin', mmin)
       gmsh.option.setNumber('Mesh.MeshSizeMax', mmax)
       gmsh.option.setNumber('Mesh.Algorithm', 6) #1=Mesh Adapt, 2=Auto, 3=Initial mesh only, 5=Delaunay, 6=Frontal-Delaunay
       gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
       gmsh.option.setNumber('Mesh.Format', 1)
       #gmsh.option.setNumber('Mesh.Smoothing', 100)
       gmsh.option.setNumber('Mesh.MinimumCirclePoints', 36)
       gmsh.option.setNumber('Mesh.CharacteristicLengthFromCurvature', 1)

#Set max mesh size
       pt = gmsh.model.getEntities(0)
       gmsh.model.mesh.setSize(pt, lm)
       print(pt)
# Set denser mesh along bowtie metal edges
       pt = gmsh.model.getEntitiesInBoundingBox(bx-err, -ba-err, hd-err, bl+bx+err, ba+err, hd+bt+err)
       print(pt, ls)
       gmsh.model.mesh.setSize(pt,ls)
# Set mesh density along coax
       pt = gmsh.model.getEntitiesInBoundingBox(rco-rcc-err, -rcc-err, hd-err, rcc+rco+err, rcc+err, hd+err)
       print(pt, lq)
       gmsh.model.mesh.setSize(pt,lq)
       pt = gmsh.model.getEntitiesInBoundingBox(rco-rc-err, -rc-err, -lc-h-err, rc+rco+err, rc+err, err)
       print(pt, lq)
       gmsh.model.mesh.setSize(pt,lq)

# set mesh density along dielectric
       gmsh.model.mesh.setTransfiniteCurve(13, 200, meshType="Progression", coef=1.0)
       gmsh.model.mesh.setTransfiniteCurve(15, 200, meshType="Progression", coef=1.0)
       
# Reduce mesh density on bowtie surface
       P1 = gmsh.model.occ.addPoint(bx+bl/3, 0, hd, lm, tag=100)
       P2 = gmsh.model.occ.addPoint(bx+bl/3, 0, hd+bt, lm, tag=101)
       P3 = gmsh.model.occ.addPoint(bx+2*bl/3, -ba/3, hd, lm, tag=102)
       P4 = gmsh.model.occ.addPoint(bx+2*bl/3, ba/3, hd, lm, tag=103)
       P5 = gmsh.model.occ.addPoint(bx+2*bl/3, -ba/3, hd+bt, lm, tag=104)
       P6 = gmsh.model.occ.addPoint(bx+2*bl/3, ba/3, hd+bt, lm, tag=105)
       gmsh.model.occ.synchronize()
       gmsh.model.mesh.embed(0, [P2,P5,P6], 2, 10)
       gmsh.model.mesh.embed(0, [P1,P3,P4], 2, 14)
       gmsh.model.occ.synchronize()

       bb = gmsh.model.getEntities(dim=3)
       pt = gmsh.model.getBoundary(bb, combined=True, oriented=False, recursive=False)
       print(pt)
       inportA = []
       radA = []
       ZBnd = []
       pecA =[]
       for bnd in pt:
          CoM = gmsh.model.occ.getCenterOfMass(bnd[0], bnd[1])
          print(CoM)
          if np.allclose(CoM, [rco, 0, -h-lc]): # Coax input port
              inportA.append(bnd[1])
          elif np.abs(CoM[2] + h) < err :
              ZBnd.append(bnd[1])
          elif (np.abs(CoM[0]) < err) or (CoM[2] < hd + bt + err):
              pecA.append(bnd[1])
          else:
              radA.append(bnd[1])
       print(inportA, pecA, radA, ZBnd)
       gmsh.model.addPhysicalGroup(2, pecA, 1)
       gmsh.model.setPhysicalName(2, 1, "PEC")
       gmsh.model.addPhysicalGroup(2, inportA, 2)
       gmsh.model.setPhysicalName(2, 2, "Inlet")
       gmsh.model.addPhysicalGroup(2, radA, 3)
       gmsh.model.setPhysicalName(2, 3, "Radiation")
       gmsh.model.addPhysicalGroup(2, ZBnd, 4)
       gmsh.model.setPhysicalName(2, 4, "Absorber")
       gmsh.model.addPhysicalGroup(3, [16], 1)
       gmsh.model.addPhysicalGroup(3, [13], 2)
       gmsh.model.addPhysicalGroup(3, [15], 3)
       gmsh.model.setPhysicalName(3, 1, "RadZone")
       gmsh.model.setPhysicalName(3, 2, "PCB")
       gmsh.model.setPhysicalName(3, 3, "Cavity")




       gmsh.model.mesh.generate(3)
       gmsh.model.mesh.optimize("Gmsh")
       gmsh.fltk.run()
   mesh, ct, fm = gmshio.model_to_mesh(gmsh.model, comm, modelRank, gdim=3)

   if mpiRank == modelRank:
       gmsh.finalize()

   mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
   elem = ufl.FiniteElement('Nedelec 1st kind H(curl)', mesh.ufl_cell(), degree=2)
   V = fem.FunctionSpace(mesh, elem) # For field solutions
   Q = fem.FunctionSpace(mesh, ("DG", 0)) # For Dk in regions

   u = ufl.TrialFunction(V)
   v = ufl.TestFunction(V)
   Dk = fem.Function(Q)
   RZone = ct.find(1)
   Diel = ct.find(2)
   Cav = ct.find(3)
   Dk.x.array[RZone] = np.full_like(RZone, 1.0, dtype=PETSc.ScalarType)
   Dk.x.array[Diel] = np.full_like(Diel, eps, dtype=PETSc.ScalarType)
   Dk.x.array[Cav] = np.full_like(Cav, 1.0, dtype=PETSc.ScalarType)

   from dolfinx import io
   with io.XDMFFile(mesh.comm, "Dk.xdmf", "w") as xdmf:
      xdmf.write_mesh(mesh)
      xdmf.write_function(Dk)

   with io.XDMFFile(mesh.comm, "BCs.xdmf", "w") as xdmf:
      xdmf.write_mesh(mesh)
      xdmf.write_meshtags(fm)

   n = ufl.FacetNormal(mesh)
   ds = ufl.Measure("ds", domain=mesh, subdomain_data=fm)

# Incident field
   class Hinc:
      def __init__(self, a, w, H):
          self.a = a
          self.w = w
          self.H = H
      def eval(self, x):
          hx = -self.H * (x[1]-self.w)/(2.0*np.pi*np.sqrt((x[0]-self.a)**2.0 + (x[1]-self.w)**2.0))
          hz = np.full_like(hx, 0.0+0j, dtype=np.complex128)
          hy = self.H * (x[0]-self.a) / (2.0 * np.pi*np.sqrt((x[0]-self.a)**2.0+(x[1]-self.w)**2.0))
          return(hx, hy, hz)

# Dirichlet n x E = 0
   facets = fm.find(1) # PEC facets
   ubc = fem.Function(V)
   ubc.x.set(0+0j)
   dofs = fem.locate_dofs_topological(V, mesh.topology.dim-1, facets)
   bc = fem.dirichletbc(ubc, dofs)

# Build RHS
   uh = fem.Function(V, dtype=np.complex128)
   f = Hinc(rco, 0.0, 1.0+0j)
   uh.interpolate(f.eval)
   finc = 2.0 * k0 * eta0 * ufl.inner(ufl.cross(n, uh), v) * ds(2) # Integrate over inlet
   uh.name = "Hinc"

# Set up weak form
   L = (ufl.inner(ufl.curl(u), ufl.curl(v)) - k0 * k0 * Dk * ufl.inner(u, v)) * ufl.dx
# Outgoing  at inport
   bc1 = 1j * k0 * ufl.inner(ufl.cross(n, u), ufl.cross(n,v)) * ds(2)
   L += bc1
# Rad boundary
   bc2 = (1j * k0 - 1.0 / rgp) * ufl.inner(ufl.cross(n, u), ufl.cross(n, v)) * ds(3)
   L += bc2
# Absorber boundary
   bc3 = 1j * k0 * ufl.inner(ufl.cross(n, u), ufl.cross(n, v)) * ds(4)
   L += bc3

   Lin_system = fem.petsc.LinearProblem(L, finc, bcs = [bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
   E = Lin_system.solve()
   lu_solver = Lin_system.solver
   lu_solver.view()

   W = fem.VectorFunctionSpace(mesh, ('DG', 2))
   Et = fem.Function(W)
   Et.interpolate(E)
   Et.name = "ElectricField"
   with io.XDMFFile(mesh.comm, "Er.xdmf", "w") as xdmf:
      xdmf.write_mesh(mesh)
      xdmf.write_function(Et)
   
   with io.VTKFile(mesh.comm, "Efield.pvd", "w") as xx:
       xx.write_mesh(mesh)
       xx.write_function([Et._cpp_object], 0)


#   with io.VTXWriter(comm, "Er.bp", Et) as xx:
#       xx.write(0.0)

# Print out time steps
   Et.name = "AnimEfield"
   with io.XDMFFile(mesh.comm, "Movie.xdmf", "w") as xx:
       xx.write_mesh(mesh)
       for t1 in range(50):
          Et.interpolate(E)
          Et.vector.array = Et.vector.array * np.exp(1j * np.pi * t1 /25.0)
          xx.write_function(Et, t1)

   Prad = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(E, ufl.cross(ufl.curl(E), n)) * ds(3))) / (2.0j * k0 * eta0), op=nMPI.SUM)
   Pinc = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds(2))) * eta0 / 2.0 , op=nMPI.SUM)
   Pref = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(E, ufl.cross(ufl.curl(E), n)) * ds(2))) / (-2.0j * k0 * eta0), op=nMPI.SUM)
   Pabs = mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(E, ufl.cross(ufl.curl(E), n)) * ds(4))) / (2.0j * k0 * eta0), op=nMPI.SUM)
 

   print("Pinc = {0}, Prad = {1}, Pref = {2}, Pabs = {3}".format(Pinc, np.real(Prad), np.real(Pinc-Pref), np.real(Pabs)))
   return 0

xin = np.array([2.5, 4.0, 8.0, 3.0, 0.05, 1.0, 0.20, 0.25, 1.04, 3.6, 1.0, 0.55, 0.1, 0.6])
res = Model(xin)

sys.exit(0)

