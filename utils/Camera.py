#import bpy
import math
import torch, gc
import numpy as np

class Camera:
	def __init__(self, cam_type, fov, focal_length, location, rotation, name, sensor_width, sensor_height, resolution_x, resolution_y, device):
		self.type = cam_type # 'Fisheye', 'Pano'
		self.fov = fov # camera.angle
		self.focal_length = focal_length
		self.location = location 
		self.rotation = rotation 
		self.name = name
		self.device = device

		self.sensor_width = sensor_width
		self.sensor_height = sensor_height

		self.resolution_x = resolution_x 
		self.resolution_y = resolution_y 

		self.create_camera()
		
	def set_device(self, device):
		self.device = device

	def set_extrinsic(self, extrinsic):
		self.extrinsic = extrinsic

	def create_camera(self):
		'''
		camera_data = bpy.data.cameras.new(name=self.name)
		camera_obj = bpy.data.objects.new(self.name, camera_data)
		self.obj = camera_obj

		self.obj.location = self.location 
		self.obj.rotation_euler = self.rotation 
		self.obj.data.type = 'PANO'

		self.obj.data.lens_unit = 'MILLIMETERS' 
		self.obj.data.angle = self.focal_length

		self.obj.data.sensor_width = self.sensor_width
		self.obj.data.sensor_height = self.sensor_height

		self.set_lens_type()

		bpy.context.scene.collection.objects.link(self.obj)
		bpy.context.view_layer.update()
		'''
		self.extrinsic = None

		'''
		if self.device :
			self.extrinsic = torch.Tensor(self.obj.matrix_world).to(self.device)
		else:
			self.extrinsic = np.zeros((4, 4))
		'''

	def set_cam_type(self, cam_type):
		self.type = cam_type
		self.set_lens_type()

	def set_lens_type(self):
		if self.type == 'Fisheye':
			self.obj.data.cycles.panorama_type = 'FISHEYE_EQUISOLID'
			self.obj.data.cycles.fisheye_lens = self.focal_length 

		else:
			self.obj.data.cycles.panorama_type = 'EQUIRECTANGULAR'

	def get_name(self):
		return self.name

	def get_cam_obj(self):
		return self.obj

	def get_extrinsic(self):
		if self.extrinsic is None:
			x, y, z = self.rotation
			cos_x, sin_x = math.cos(x), math.sin(x)
			Rx = torch.Tensor([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
			cos_y, sin_y = math.cos(y), math.sin(y)
			Ry = torch.Tensor([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
			cos_z, sin_z = math.cos(z), math.sin(z)
			Rz = torch.Tensor([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

			rot = Rz @ Ry @ Rx
			t = torch.Tensor(self.location).reshape(3, 1)

			mat = torch.hstack([rot, t]) # 3x4 
			line = torch.Tensor([[0, 0, 0, 1]])
			mat = torch.vstack([mat, line])

			if self.device :
				mat = mat.to(self.device)

			self.extrinsic = mat
			
		return self.extrinsic

	def get_whole_pts(self):
		return 0 

	def whole_pixel2world(self):
		return 0 

	def pixel2world(self, pixels):
		return 0

	def get_resolution(self):
		return (self.resolution_x, self.resolution_y)



class FisheyeCam(Camera):
	def __init__(self, opt, location, rotation, name, device, config_path):
		super().__init__("Fisheye", opt.fov, opt.focal_length, location, rotation, name, opt.sensor_width, opt.sensor_height, opt.fisheye_resolution_x, opt.fisheye_resolution_y, device)

		self.device = device
		self.config_path = config_path
		self.pts = None

		self.read_config_from_file()
  
	def update_intrinsic(self, poly_coef, inv_poly_coef, center, affine):
		self.poly_coef = poly_coef 
		self.inv_poly_coef = inv_poly_coef
		self.center_x = center[0]
		self.center_y = center[1]
		self.c, self.d, self.e = affine
  

	
	def read_config_from_file(self):

		with open(self.config_path) as f:
			lines = f.readlines() 

		calib_data = [] 
		for line in lines:
			if (line[0] == '#' or line[0] == '\n'):
				continue
			calib_data.append([float(i) for i in line.split()])

		# polynopmial coeeficients for the DIRECT mapping function 
		self.num_poly = int(calib_data[0][0])
		self.poly_coef = torch.tensor(calib_data[0][1:]).to(self.device)

		# polynomial coefficients for the inverse mapping function 
		self.inv_poly_coef = torch.tensor(calib_data[1][1:])#.to(self.device)
		self.num_inv_poly = int(calib_data[1][0])

		# center:
		self.center_x = calib_data[2][1]
		self.center_y = calib_data[2][0]

		# affine parameters "c", "d", "e"
		self.c, self.d, self.e = calib_data[3]


	def whole_pixel2world(self):
		# torch array of pixels # 2 X N 
		x = torch.range(0, self.resolution_x-1).to(self.device) 
		y = torch.range(0, self.resolution_y-1).to(self.device)
		grid_x, grid_y = torch.meshgrid(y, x)
		pixels = torch.stack((grid_y, grid_x)).reshape(2, -1) # 2 x N 


		self.pts = self.pixel2world(pixels)


	def get_whole_pts(self):
		if self.pts is None:
			self.whole_pixel2world()

		return self.pts 


	# unproject pixel coordinate to world (spherical ray)
	def pixel2world(self, input_pixels):
		"""
		pixel2world unprojects a 2D pixel point onto the unit sphere
		you can find out the world coordinate point by multiplying depth value
		"""
		# pixel * 3 

  
		pixels = (input_pixels) - torch.Tensor([[self.center_x], [self.center_y]]).to(self.device)

		affine_mat = torch.tensor([[self.c, self.d], [self.e, 1.]]).to(self.device)
		inv_affine_mat = torch.inverse(affine_mat)
		sensor_coord = torch.matmul(inv_affine_mat, pixels) 

		r = torch.sqrt(sensor_coord[0]**2 + sensor_coord[1]**2) # N 
		r_poly = torch.stack([r**i for i in range(self.num_poly)]) 

		z = -1 * torch.matmul(self.poly_coef, r_poly) 

		z = torch.unsqueeze(z, 0)

		pts = torch.cat([pixels, z], dim=0)
		pts_norm = pts / torch.linalg.norm(pts, axis=0)

		rot_axis = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).to(self.device)

		result = torch.matmul(rot_axis, pts_norm)

		del r_poly

		return result


	def world2pixel(self, pts):
		#pts : 3xN
		# rotate axis (x->x, y->-y, z-> -z)
		rot_axis = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).to(pts.device)
		rot_pts = torch.matmul(rot_axis, pts)

		x = rot_pts[0]
		y = rot_pts[1]
		z = rot_pts[2]

		norm = torch.hypot(x, y)
		theta = torch.arctan2(-z, norm)

		# test 
		rho = 0
		for i in range(self.num_inv_poly):
			rho += theta**i * self.inv_poly_coef[i].to(pts.device)

		u = x/norm * rho 
		v = y/norm * rho 

		pixels = torch.stack([u, v]) # 2 x N 
		

		affine_mat = torch.tensor([[self.c, self.d], [self.e, 1.]]).to(pts.device)
		pixel_affine = (torch.matmul(affine_mat, pixels) + torch.Tensor([[self.center_x], [self.center_y]]).to(pts.device)) #/ 3.
		
		max_theta = math.radians(self.fov) / 2.0
		theta = theta + math.pi/2 # zenith angle from optical axis 

		valid = torch.ones(theta.shape).to(pts.device)
		valid[theta > max_theta] = 0

		valid[pixel_affine[0] >= self.resolution_x] = 0
		valid[pixel_affine[0] <0] = 0
		valid[pixel_affine[1] >= self.resolution_y] = 0
		valid[pixel_affine[1] <0] = 0
		
		pixel_affine[0][valid == 0] = 0 
		pixel_affine[1][valid == 0] = 0

		
		pixel_result = torch.cat([pixel_affine, valid.unsqueeze(0)]) #valid.unsqueeze(0)])

		return pixel_result# 3 x N [u, v, valid(valid fov)] 

	def world2sensor(self, pts):
		pixel = self.world2pixel(pts)

		u = pixel[0]
		v = pixel[1]

		u = u - self.center_x 
		v = v - self.center_y 

		u = 2 * u / self.resolution_x 
		v = 2 * v / self.resolution_y

		pixel[0] = u 
		pixel[1] = v

		return pixel  





class PanoramaCam(Camera):
	def __init__(self, opt, location, rotation, name, device):
		super().__init__("Pano", opt.fov, opt.focal_length, location, rotation, name, opt.sensor_width, opt.sensor_height, opt.pano_resolution_x, opt.pano_resolution_y, device)

		self.device = device
		self.pts = None

	

	def whole_pixel2world(self):
		# torch array of pixels # 2 X N 
		x = torch.range(0, self.resolution_x-1).to(self.device) 
		y = torch.range(0, self.resolution_y-1).to(self.device)
		grid_x, grid_y = torch.meshgrid(y, x)
		pixels = torch.stack((grid_y, grid_x)).reshape(2, -1) # 2 x N 

		self.pts = self.pixel2world(pixels)


	def get_whole_pts(self):
		if self.pts is None:
			self.whole_pixel2world()

		return self.pts 


	# unproject pixel coordinate to world (spherical ray)
	def pixel2world(self, pixels):
		"""
		pixel2world unprojects a 2D pixel point onto the unit sphere
		you can find out the world coordinate point by multiplying depth value
		"""
		u = pixels[0]
		v = pixels[1]

		# normalized to [-0.5, 0.5]
		u_norm = u / self.resolution_x - 0.5 
		v_norm = v / self.resolution_y - 0.5

		phi = u_norm * 2 * math.pi
		theta = v_norm * math.pi 

		z = -torch.sin(theta)

		r = torch.cos(theta)
		x = r * torch.sin(phi)
		y = r * torch.cos(phi)

	
		result = torch.stack([x, y, z])		

		return result

	def world2pixel(self, pts):
		x = pts[0]
		y = pts[1]
		z = pts[2]
		

		phi = torch.arctan2(x, y) 
		theta = -torch.arctan(z/torch.hypot(x, y))

		u_norm = phi/(math.pi*2) + 0.5 
		v_norm = theta/math.pi + 0.5

		u = u_norm * self.resolution_x
		v = v_norm * self.resolution_y

		result = torch.stack([u, v])

		return result 