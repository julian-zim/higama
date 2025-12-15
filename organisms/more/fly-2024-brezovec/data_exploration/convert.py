import os
from pynwb import NWBHDF5IO


in_path = os.path.join(os.path.expanduser('~'), '/lisc/scratch/neurobiology/zimmer/zimmermann/DATA/fly-2024-brezovec/orig')
out_path = os.path.join(os.path.expanduser('~'), '/lisc/scratch/neurobiology/zimmer/zimmermann/DATA/fly-2024-brezovec/proc')

for filename in os.listdir(in_path):
	with NWBHDF5IO(os.path.join(in_path, filename), mode='r', load_namespaces=True) as io_in:
		nwbfile = io_in.read()

		for name in list(nwbfile.acquisition.keys()):
			del nwbfile.acquisition[name]
			
		for name in list(nwbfile.processing.keys()):
			if name == 'ophys': del nwbfile.processing[name]

		with NWBHDF5IO(os.path.join(out_path, filename), mode='w') as io_out:
			io_out.export(io_in, nwbfile)
