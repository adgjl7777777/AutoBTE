from AutoBTE.BoltzTrap.BTE import btp2
import sys, os
import numpy as np
base_path = "/nas/transcendence/2024_2_J3/CNT_maker/"
structure_name = sys.argv[1]
work_path = os.path.join(base_path, structure_name)
btp = btp2(os.path.join(work_path,"vasp_vacuum"))
btp.interpolate(n=2400)  # sample n irreducible k points for each one that is present in the input

print(len(btp.equivalences),flush=True)
print(f'{btp.get_fermi_energy()} eV',flush=True)

os.makedirs(f"./bandstructure/{structure_name}/",exist_ok=True)
os.makedirs(f"./dos/{structure_name}/",exist_ok=True)
os.makedirs(f"./T/{structure_name}",exist_ok=True)
os.makedirs(f"./mu/{structure_name}",exist_ok=True)

os.makedirs(f"./data/dos/{structure_name}",exist_ok=True)
os.makedirs(f"./data/T/{structure_name}",exist_ok=True)
os.makedirs(f"./data/mu/{structure_name}",exist_ok=True)
#btp.BandStructure.set_bandpath("GXHCHY")

btp.BandStructure.plot(save=True,path=f"./bandstructure/{structure_name}.png")
x, y = btp.DOS.plot(save=True,path=f"./dos/{structure_name}.png")
np.save(f"./data/dos/{structure_name}/x.npy",x)
np.save(f"./data/dos/{structure_name}/y.npy",y)
for components in ["x","y","z",None]:

    os.makedirs(f"./data/T/{structure_name}/{components}",exist_ok=True)
    os.makedirs(f"./data/mu/{structure_name}/{components}",exist_ok=True)
    x, y = btp.FermiIntegrals.plot(x="temperature",y="sigma",component=components, temp_range=(200,1000),save=True,path=f"./T/{structure_name}/{components}.png")
    np.save(f"./data/T/{structure_name}/{components}/x.npy",x)
    np.save(f"./data/T/{structure_name}/{components}/y.npy",y)
    x, y = btp.FermiIntegrals.plot(x="chemical_potential",y="sigma",component=components, mu_range=(-5,5),save=True,path=f"./mu/{structure_name}/{components}.png")
    np.save(f"./data/mu/{structure_name}/{components}/x.npy",x)
    np.save(f"./data/mu/{structure_name}/{components}/y.npy",y)