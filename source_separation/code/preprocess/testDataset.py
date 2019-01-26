#!usr/bin/python3

from SPLdataset import SPLdataset as SPL
import numpy as np
from presvld import *

# int (mapped with) name of instruments.
# 0: accordion
# 1: acoustic_guitar
# 2: cello
# 3: flute
# 4: saxophone
# 5: trumpet
# 6: violin
# 7: xylophone
ins1 = [0,1,2,3,4,5,6,7]
ins2 = [0,1,2,3,4,5,6,7]

epoch = 8
save_npz = True

spl = SPL(ins1, ins2, val=False)

for i in range(epoch):
	print("Epoch "+str(i+1))
	image_left, image_right, spec_left, spec_right, spec_cb, phase_left, phase_right, phase_cb = spl.get_data()
	print("Batch size of Epoch "+str(i+1)+" is "+str(np.shape(image_left)[0]))
	if save_npz:
		save_data(i, image_left, image_right, spec_left, spec_right, spec_cb, phase_left, phase_right, phase_cb)
