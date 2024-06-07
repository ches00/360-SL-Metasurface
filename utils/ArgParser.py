import argparse



class Argument:
	def __init__(self):
		self.parser = argparse.ArgumentParser()

		# for init scene
		self.parser.add_argument('--N_scene', type=int, default=2)
		self.parser.add_argument('--N_obj', type=int, default=60)
		self.parser.add_argument('--max_dist_bg', type=int, default=6)
		self.parser.add_argument('--min_dist_bg', type=int, default=4)

		# data directory path6.6
		self.parser.add_argument('--abs')
		self.parser.add_argument('--obj_path', type=str, default="models-OBJ\\models\\models")
		self.parser.add_argument('--tex_path', type=str, default="texture\\models-textures\\textures")
		self.parser.add_argument('--save_fisheye_path', type=str, default="/data/fisheye")
		self.parser.add_argument('--save_pano_path', type=str, default="/data/pano")

		# resolution for rendering
		self.parser.add_argument('--fisheye_resolution_x', type=int, default=640)
		self.parser.add_argument('--fisheye_resolution_y', type=int, default=400)

		self.parser.add_argument('--pano_resolution_x', type=int, default=800)
		self.parser.add_argument('--pano_resolution_y', type=int, default=400)

		# for Camera calibaration
		self.parser.add_argument('--fov', type=float, default=185.0)
		self.parser.add_argument('--focal_length', type=float, default=1.80)
		self.parser.add_argument('--sensor_width', type=float, default=6.17) # 1/2.3 inch
		self.parser.add_argument('--sensor_height', type=float, default=4.55)
		self.parser.add_argument('--baseline', type=float, default=0.1) # 10cm


		# for image formation 
		self.parser.add_argument('--device', type=str, default="cuda:1")
		self.parser.add_argument('--ambient_light_off', type=bool, default=False)
		self.parser.add_argument('--noise_gaussian_stddev', type=float, default=2e-2)
		self.parser.add_argument('--ambient_power_max', type=float, default=0.6)
		self.parser.add_argument('--ambient_power_min', type=float, default=0.6)
		self.parser.add_argument('--laser_power_min', type=float, default=1e-1, help='previous default: 5e-1')
		self.parser.add_argument('--laser_power_max', type=float, default=1.5e-0, help='previous default: 5e-1')

		self.parser.add_argument('--cam_config_path', type=str, default="./calib_results.txt")


		# for Metasurface
		self.parser.add_argument('--N_phase', type=int, default=1000)
		self.parser.add_argument('--N_supercell', type=int, default=10)
		self.parser.add_argument('--N_theta', type=int, default=300)
		self.parser.add_argument('--N_alpha', type=int, default=100)
		self.parser.add_argument('--wave_length', type=float, default=532e-9) # mono-chromatic structured light
		self.parser.add_argument('--pixel_pitch', type=float, default=260e-9) # Metasurface pixel pitch

#
		# for stereo matching
		self.parser.add_argument('--N_depth_candidate', type=int, default=90)
		self.parser.add_argument('--max_depth', type=float, default=5.0) # unit: [m]
		self.parser.add_argument('--min_depth', type=float, default=0.3)

		# for optimization
		self.parser.add_argument('--lr', type=float, default=3e-4)
		self.parser.add_argument('--momentum', type=float, default=0.9)
		self.parser.add_argument('--input_path', type=str, default='./data')
		self.parser.add_argument('--log', type=str, default="./log/") 
		self.parser.add_argument('--batch_size', type=int, default=8)

		self.parser.add_argument('--train_path', type=str, default="./data/train")
		self.parser.add_argument('--valid_path', type=str, default="./data/test")



		# for test 
		self.parser.add_argument('--num_gpu', type=int, default=1)
		self.parser.add_argument('--pattern_path', type=str, default="./checkpoint/pattern.mat") 
		self.parser.add_argument('--test_path', type=str, default='./data/test') 
		self.parser.add_argument('--single_img', type=bool, default=True)
		self.parser.add_argument('--test_save_path', type=str, default='./log/inference') 
		self.parser.add_argument('--chk_path', type=str, default='./checkpoint/model.pth')
		self.parser.add_argument('--front_right_config', type=str, default='./front_cam.npy')
		self.parser.add_argument('--back_right_config', type=str, default='./back_cam.npy')






	def parse(self):
		return self.parser.parse_args()

