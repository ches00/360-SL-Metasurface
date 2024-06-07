import numpy as np 
import OpenEXR
import Imath

def read_exr_as_np(path):
	""" Read exr file as numpy array
	"""

	f = OpenEXR.InputFile(path)
	channels = f.header()['channels']

	dw = f.header()['dataWindow']
	size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

	ch_names = [] 
	image = np.zeros((size[1], size[0], len(channels) - 1))

	for i, ch_name in enumerate(channels):
		ch_names.append(ch_name)
		ch_dtype = channels[ch_name].type  
		ch_str = f.channel(ch_name, ch_dtype)

		if ch_dtype == Imath.PixelType(Imath.PixelType.FLOAT):
			np_dtype = np.float32
		else:
			np_dtype = np.float16   
		image_ch = np.fromstring(ch_str, dtype=np_dtype)
		image_ch.shape = (size[1], size[0])

		if ch_name == "A":
			continue
		else:
			image[:, :, 3-i] = image_ch   

	return image



