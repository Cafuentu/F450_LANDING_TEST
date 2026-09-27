import sys, glob, os
from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
src = "/mnt/user-data/uploads/Proyecto Drones/F450_modelo/Genesis drone F450"
def bbox(s):
    b = Bnd_Box(); BRepBndLib.Add_s(s, b); mn=b.CornerMin(); mx=b.CornerMax(); x0,y0,z0,x1,y1,z1 = mn.X(),mn.Y(),mn.Z(),mx.X(),mx.Y(),mx.Z()
    return [round(v,1) for v in (x0,y0,z0,x1,y1,z1)]
for f in sorted(glob.glob(src + "/*.step")):
    name = os.path.basename(f).split(" - ")[0].replace(" ", "_")
    r = STEPControl_Reader(); r.ReadFile(f); r.TransferRoots(); sh = r.OneShape()
    ex = TopExp_Explorer(sh, TopAbs_SOLID); solids = []
    while ex.More(): solids.append(ex.Current()); ex.Next()
    print(name, "solids:", len(solids), "bbox:", bbox(sh))
    if "assembly" in name:
        for i, s in enumerate(solids):
            g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g); c = g.CentreOfMass()
            print("   ", i, "vol", round(g.Mass()), "com", [round(c.X(),1), round(c.Y(),1), round(c.Z(),1)], bbox(s))
    BRepMesh_IncrementalMesh(sh, 0.15, False, 0.35, True)
    w = StlAPI_Writer(); w.ASCIIMode = False; w.Write(sh, name + ".stl")
    print("   ->", os.path.getsize(name + ".stl")//1024, "KB")
