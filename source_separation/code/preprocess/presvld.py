import torch
import numpy as np

root_dir = '../../npz_data'

def save_data(epoch, image_left, image_right, spec_left, spec_right, spec_cb, phase_left, phase_right, phase_cb):
	np.savez(root_dir+'/'+str(epoch)+'.npz', image_left=image_left.cpu().numpy(), image_right=image_right.cpu().numpy(), spec_left=spec_left.cpu().numpy(), spec_right=spec_right.cpu().numpy(), spec_cb=spec_cb.cpu().numpy(), phase_left=phase_left, phase_right=phase_right, phase_cb=phase_cb)
