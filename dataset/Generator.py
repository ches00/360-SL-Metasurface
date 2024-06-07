import bpy
import os, math, random

from utils.Camera import *

class Generator:
	def __init__(self, arg):
		self.N_scene = arg.N_scene
		self.N_obj = arg.N_obj

		self.max_dist_bg = arg.max_dist_bg
		self.min_dist_bg = arg.min_dist_bg 

		self.obj_path = arg.obj_path
		self.tex_path = arg.tex_path

		self.save_fisheye_path = arg.save_fisheye_path
		self.save_pano_path = arg.save_pano_path

		self.fisheye_resolution_x = arg.fisheye_resolution_x
		self.fisheye_resolution_y = arg.fisheye_resolution_y

		self.pano_resolution_x = arg.pano_resolution_x
		self.pano_resolution_y = arg.pano_resolution_y

		self.fov = arg.fov 
		self.focal_length = arg.focal_length

		self.sensor_width = arg.sensor_width
		self.sensor_height = arg.sensor_height
		self.baseline = arg.baseline

		self.arg = arg




	def gen(self):
		self.scene_init()
		
		for i in range(self.N_scene):
			self.clean_up_scene()
			
			bg_size = self.create_background()
			
			if i%2 == 0:
				self.create_random_object(bg_size, 'TEXTURE')
			else:
				self.create_random_object(bg_size, 'NONE')

	
			self.render_init()

			# PNG file
			self.render('RGB', i)
			self.links.clear() 
			self.render('Albedo', i)
			self.links.clear()
			self.render('Occlusion', i)
			self.links.clear()

			# OpenEXR file
			self.scene.render.image_settings.file_format = 'OPEN_EXR'
			self.render('Depth', i)
			self.links.clear()
			self.render('Normal', i)
			self.links.clear()			


	def scene_init(self):
		self.scene = bpy.context.scene

		# to reduce noise - fixed value
		self.scene.cycles.samples = 1000
		self.scene.cycles.sample_clamp_indirect = 1.0

		# to use node for rendering
		self.scene.use_nodes = True 
		self.nodes = self.scene.node_tree.nodes 
		self.links = self.scene.node_tree.links


		############################
		## Camera Setting - share
		############################ 

		radians_90 = math.radians(90)
		unit = self.baseline / 2

		Camera1 = FisheyeCam(self.arg, (unit, unit, 0),(radians_90, 0, math.radians(-45)),'cam1', None)
		Camera2 = FisheyeCam(self.arg, (-unit, unit, 0),(radians_90, 0, math.radians(45)), 'cam2', None)
		Camera3 = FisheyeCam(self.arg, (-unit, -unit, 0),(radians_90, 0, math.radians(135)), 'cam3', None)
		Camera4 = FisheyeCam(self.arg, (unit, -unit, 0),(radians_90, 0, math.radians(-135)), 'cam4', None)

		Camera_pano = Camera('Pano', self.fov, self.focal_length, (0, 0, 0), (radians_90, 0, 0), 'pano_cam', self.sensor_width, self.sensor_height, self.pano_resolution_x, self.pano_resolution_y, None)
		self.cam_list = [Camera1, Camera2, Camera3, Camera4]
		self.pano_cam_list = [Camera_pano]

		#for cam in self.cam_list:
			#self.scene.collection.objects.link(cam.get_cam_obj())

		#for cam in self.pano_cam_list:
			#self.scene.collection.objects.link(cam.get_cam_obj())

		############################
		## Light Setting - share
		############################
		light = bpy.data.objects['Light']
		#light.location = (1, 3, 2)
		light.location = (0, 0, 0)
		light.data.shadow_soft_size = 0
		#light.data.cycles.cast_shadow = False # doesn't project shadow on the scene


	def clean_up_scene(self):

		for node in self.nodes:
			self.nodes.remove(node)

		for mat in bpy.data.materials:
			mat.user_clear()
			bpy.data.materials.remove(mat)

		for texture in bpy.data.textures:
			texture.user_clear()
			bpy.data.textures.remove(texture)

		bpy.ops.object.select_all(action='DESELECT')

		for item in bpy.data.objects:
			if item.type == "MESH":
				bpy.data.objects[item.name].select_set(True)
				bpy.ops.object.delete()

		for item in bpy.data.meshes:
			bpy.data.meshes.remove(item)




	def create_background(self):
		""" randomly create background (sphere or cube)

		Return:
			size : the maximum size of background

		"""
		bg_type = random.choice(['SPHERE', 'CUBE']) # randomly choose the background type
		size = random.uniform(self.min_dist_bg, self.max_dist_bg)
		#size=10
		#bg_type = 'CUBE'#
		tex_list = os.listdir(self.tex_path)
		tex_name = random.choice(tex_list)
		imported_tex_path = os.path.join(self.tex_path, tex_name)

		if bg_type == 'SPHERE':
			bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
			sphere = bpy.data.objects['Sphere']
			sphere.select_set(True)
			bpy.ops.object.shade_smooth()
			sphere.scale = (size, size, size)
			self.apply_texture_to_object(imported_tex_path, sphere)

		else:
			bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
			cube = bpy.data.objects['Cube']
			cube.select_set(True)
			bpy.ops.object.shade_smooth()
			cube.scale = (size, size, size)
			self.apply_texture_to_object(imported_tex_path, cube)	


		return size


	def create_random_object(self, size, type='NONE'):
		""" randomly import obj files and put/rotate/resize them on the scene

		Args:
			size : the maximum size of the background
			type: ['TEXTURE', 'NONE'] TEXTURE - draw a texture image onto object

		"""
		obj_list = os.listdir(self.obj_path)
		obj_list = [file for file in obj_list if file.endswith(".obj")]
		obj_choice = random.sample(obj_list, self.N_obj)

		tex_list = os.listdir(self.tex_path)
		tex_choice = random.sample(tex_list, self.N_obj)

		radian_360 = math.radians(360)

		for (obj_name, tex_name) in zip(obj_choice, tex_choice):
			prev_objects = set(self.scene.objects.keys())

			path = os.path.join(self.obj_path, obj_name)
			bpy.ops.import_scene.obj(filepath=path)
			imported_obj_name = list(set(self.scene.objects.keys()) - prev_objects)[0]
			imported_obj = bpy.data.objects[imported_obj_name]

			# object location range [-size, -1.5] and [1.5, size], to guarantee the clear view of the camera (camera is located at (0, 0, 0))
			loc_x, loc_y, loc_z = random.uniform(0.4, 2), random.uniform(0.4, 2), random.uniform(0.4, 2)
			imported_obj.location = (random.choice([1, -1])*loc_x, random.choice([1, -1])*loc_y, random.choice([1, -1])*loc_z)
			
			#mag = math.sqrt(loc_x**2 + loc_y**2 + loc_z**2)
			#scale = random.uniform(mag/80, mag/120) 
			scale_x, scale_y, scale_z = random.uniform(0.8, 2), random.uniform(0.8, 2), random.uniform(0.8, 2)

			imported_obj.scale = (scale_x, scale_y, scale_z)
			imported_obj.rotation_euler = (random.uniform(0, radian_360), random.uniform(0, radian_360), random.uniform(0, radian_360))

			if type == 'TEXTURE':
				imported_tex_path = os.path.join(self.tex_path, tex_name)
				self.apply_texture_to_object(imported_tex_path, imported_obj)



	def apply_texture_to_object(self, tex_path, obj):
		""" randomly import image texture and add to the object

		Args:
			tex_path : directory path containing texture img files
			obj : a target bpy object to use the texture
			
		"""
		# Create material for texture
		mat = bpy.data.materials.new(obj.data.name + '_texture')
		obj.data.materials.clear()
		obj.data.materials.append(mat)

		mat.use_nodes = True 
		mat_nodes = mat.node_tree.nodes 

		bsdf_node = mat_nodes['Principled BSDF']
		mat_out_node = mat_nodes['Material Output']

		tex_node = mat_nodes.new('ShaderNodeTexImage')
		tex_img = bpy.data.images.load(tex_path)
		tex_node.image = tex_img 

		# Material shading node linking
		mat.node_tree.links.new(tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
		mat.node_tree.links.new(bsdf_node.outputs['BSDF'], mat_out_node.inputs['Surface'])

	def render_init(self):
		self.scene.render.image_settings.file_format = 'PNG'
		self.scene.render.engine = 'CYCLES'

		# using GPU
		
		self.scene.cycles.device = 'GPU'
		bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

		for device in bpy.context.preferences.addons['cycles'].preferences.devices:
			if device.type == 'CUDA':
				device.use = True 
			print("Device '{}' type {} : {}".format(device.name, device.type, device.use))		


		# rendering image resolution
		self.render_layer_node = self.nodes.new('CompositorNodeRLayers')
		self.compositor_node = self.nodes.new('CompositorNodeComposite')
		self.map_value_node = self.nodes.new('CompositorNodeMapValue')

		self.alpha_node = self.nodes.new("CompositorNodeSetAlpha")

		# for occlusion map 
		self.math_node = self.nodes.new('CompositorNodeMath')
		self.math_node.operation = "GREATER_THAN"
		self.math_node.inputs[1].default_value = 0.5


		# Those values would be paramaterized 
		self.map_value_node.offset[0] = 0
		self.map_value_node.size[0] = 0.1
		self.map_value_node.use_min = True 
		self.map_value_node.use_max = True 
		self.map_value_node.min[0] = 0.0
		self.map_value_node.max[0] = 1.

		# for rendering normal map
		 # normal map nodes

		self.seperate_RGBA_node = self.nodes.new("CompositorNodeSepRGBA")
		self.add_node_R = self.nodes.new("CompositorNodeMath")
		self.add_node_R.operation = "ADD"
		self.add_node_R.inputs[1].default_value = 1

		self.add_node_G = self.nodes.new("CompositorNodeMath")
		self.add_node_G.operation = "ADD"
		self.add_node_G.inputs[1].default_value = 1

		self.add_node_B = self.nodes.new("CompositorNodeMath")
		self.add_node_B.operation = "ADD"
		self.add_node_B.inputs[1].default_value = 1

		self.divide_node_R = self.nodes.new("CompositorNodeMath")
		self.divide_node_R.operation = "DIVIDE"
		self.divide_node_R.inputs[1].default_value = 2

		self.divide_node_G = self.nodes.new("CompositorNodeMath")
		self.divide_node_G.operation = "DIVIDE"
		self.divide_node_G.inputs[1].default_value = 2

		self.divide_node_B = self.nodes.new("CompositorNodeMath")
		self.divide_node_B.operation = "DIVIDE"
		self.divide_node_B.inputs[1].default_value = 2

		self.combine_RBGA_node = self.nodes.new("CompositorNodeCombRGBA")



	def render(self, render_type, i_th):
		"""
		Arg:
			render_type: ['RGB', 'Depth', 'Albedo']
		"""

		if render_type == "RGB":
			self.links.new(self.render_layer_node.outputs[0], self.compositor_node.inputs[0])
		elif render_type == "Albedo":
			bpy.context.view_layer.use_pass_diffuse_color = True 			
			self.links.new(self.render_layer_node.outputs['DiffCol'], self.alpha_node.inputs['Image'])
			self.links.new(self.render_layer_node.outputs['Alpha'], self.alpha_node.inputs['Alpha'])
			self.links.new(self.alpha_node.outputs['Image'], self.compositor_node.inputs[0])
		elif render_type == "Depth":
			bpy.context.view_layer.use_pass_z = True 
			self.links.new(self.render_layer_node.outputs['Depth'], self.map_value_node.inputs[0])
			self.links.new(self.map_value_node.outputs[0], self.compositor_node.inputs[0])
		elif render_type == "Occlusion":
			bpy.context.view_layer.use_pass_shadow = True 
			self.links.new(self.render_layer_node.outputs['Shadow'], self.math_node.inputs[0])
			self.links.new(self.math_node.outputs[0], self.compositor_node.inputs[0])

		elif render_type == "Normal":
			bpy.context.view_layer.use_pass_normal = True
			# R
			self.links.new(self.render_layer_node.outputs["Normal"], self.seperate_RGBA_node.inputs[0])
			self.links.new(self.seperate_RGBA_node.outputs['R'], self.add_node_R.inputs["Value"])
			self.links.new(self.add_node_R.outputs["Value"], self.divide_node_R.inputs['Value'])
			self.links.new(self.divide_node_R.outputs["Value"], self.combine_RBGA_node.inputs['R'])
			self.links.new(self.combine_RBGA_node.outputs[0], self.compositor_node.inputs[0])

			# G
			self.links.new(self.seperate_RGBA_node.outputs['G'], self.add_node_G.inputs["Value"])
			self.links.new(self.add_node_G.outputs["Value"], self.divide_node_G.inputs['Value'])
			self.links.new(self.divide_node_G.outputs["Value"], self.combine_RBGA_node.inputs['G'])
			self.links.new(self.combine_RBGA_node.outputs[0], self.compositor_node.inputs[0])

			#B
			self.links.new(self.seperate_RGBA_node.outputs['B'], self.add_node_B.inputs["Value"])
			self.links.new(self.add_node_B.outputs["Value"], self.divide_node_B.inputs['Value'])
			self.links.new(self.divide_node_B.outputs["Value"], self.combine_RBGA_node.inputs['B'])
			self.links.new(self.combine_RBGA_node.outputs[0], self.compositor_node.inputs[0])

		else:
			print("Unsupported function!")


		self.scene.render.resolution_x = self.fisheye_resolution_x
		self.scene.render.resolution_y = self.fisheye_resolution_y

		img_name = '{0:05d}'.format(i_th)

		for cam in self.cam_list:
			img_path = os.path.join(render_type, os.path.join(cam.get_name(), img_name))
			self.scene.render.filepath = os.path.join(self.save_fisheye_path, img_path)
			self.scene.camera = cam.get_cam_obj()
			bpy.ops.render.render(write_still=True)


		self.scene.render.resolution_x = self.pano_resolution_x
		self.scene.render.resolution_y = self.pano_resolution_y
		
		if render_type == "Occlusion":
			return 

		for cam in self.pano_cam_list:
			img_path = os.path.join(render_type, os.path.join(cam.get_name(), img_name))
			self.scene.render.filepath = os.path.join(self.save_pano_path, img_path)
			self.scene.camera = cam.get_cam_obj()
			bpy.ops.render.render(write_still=True)




