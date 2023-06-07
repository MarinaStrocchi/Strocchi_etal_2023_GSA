import numpy as np
import pyvista as pv

from GSA_library.mesh_utils import *

def print_screenshot(plt_msh,
					 screenshot_name,
					 fig_w=400,
					 fig_h=400,
					 camera_azimuth=None,
					 camera_elevation=None,
					 camera_roll=None,
					 meshcolor='lightgray'):

	plotter = pv.Plotter(off_screen=True)
	plotter.background_color = "white"

	msh = plotter.add_mesh(plt_msh,color=meshcolor)

	if camera_azimuth is not None: plotter.camera.azimuth = camera_azimuth 
	if camera_elevation is not None: plotter.camera.elevation = camera_elevation
	if camera_roll is not None: plotter.camera.roll = camera_roll

	plotter.screenshot(filename=screenshot_name, 
						transparent_background=None, 
						   return_img=True,
						   window_size=[fig_w,fig_h])
	plotter.close()

def visualise_motion(displacement_file,
					 meshname,
					 screenshot_basename,
					 framerate=25,
					 camera_settings=None,
					 window_size=400,
					 meshcolor='lightgray'):

	pts,elem = read_binary_mesh(meshname)
	header,u = read_IGB_file(displacement_file)
	
	nt = u.shape[0]
	np = u.shape[1]

	if np!=pts.shape[0]:
		raise Exception("Mesh and displacement file dimensions do not match.")

	# initialise mesh
	pv_msh = carp_to_pyvista(pts,elem)

	for t in range(nt):
		print("Processing time step "+str(t)+"/"+str(nt-1)+"...")

		pv_msh.points = u[t,:,:]

		if not os.path.exists(screenshot_basename+str(t)+".png"):
			if camera_settings is None:
				print_screenshot(pv_msh,
								 screenshot_basename+str(t)+".png",
								 fig_w=window_size,
								 fig_h=window_size)
			else:
				print_screenshot(pv_msh,
								 screenshot_basename+str(t)+".png",
								 fig_w=window_size,
								 fig_h=window_size,
								 camera_azimuth=camera_settings["azimuth"],
								 camera_elevation=camera_settings["elevation"],
								 camera_roll=camera_settings["roll"],
								 meshcolor=meshcolor)

	cmd = ["ffmpeg -r",str(framerate),"-i",screenshot_basename+"%d.png"]
	cmd += ["-vcodec","libx264","-vf","scale="+str(window_size)+":"+str(window_size),screenshot_basename+".avi"]
	cmd_str = " ".join(cmd)
	os.system(cmd_str)

def concatenate_images(displacement_file,
					   meshname,
					   screenshot_basename,
					   camera_settings,
					   n_frames=5,
					   views=None,
					   window_size=400,
					   meshcolor='lightgray'):

	pts,elem = read_binary_mesh(meshname)
	header,u = read_IGB_file(displacement_file)
	
	nt = u.shape[0]
	nx = u.shape[1]

	if nx!=pts.shape[0]:
		raise Exception("Mesh and displacement file dimensions do not match.")

	# initialise mesh
	pv_msh = carp_to_pyvista(pts,elem)

	if views is None:
		views = list(camera_settings.keys())
	else:
		settings_views = list(camera_settings.keys())
		for v in views:
			if v not in settings_views:
				raise Exception("View "+v+" not in camera settings. Add settings for this view.")

	t_vector = np.linspace(0,nt-1,num=n_frames,endpoint=True,dtype=int)

	img_structure = []
	for i in range(len(views)):
		img_structure.append([])

	for i,t in enumerate(t_vector):

		print("Processing time step "+str(t)+"/"+str(nt-1)+"...")

		pv_msh.points = u[t,:,:]

		for j,v in enumerate(views):

			screenshot_name = screenshot_basename+v+"_"+str(t)+".png"
			img_structure[j].append(screenshot_name)

			if not os.path.exists(screenshot_name):
				print_screenshot(pv_msh,
								 screenshot_name,
								 fig_w=window_size,
								 fig_h=window_size,
								 camera_azimuth=camera_settings[v]["azimuth"],
								 camera_elevation=camera_settings[v]["elevation"],
								 camera_roll=camera_settings[v]["roll"],
								 meshcolor=meshcolor)

	output_fig = screenshot_basename+".png"
	if os.path.exists(output_fig):
		cmd="rm "+output_fig	
		
	cmd="convert -size "+str(window_size*n_frames)+"x"+str(window_size*len(img_structure))+" -background white xc:white -colorspace srgb "+output_fig
	os.system(cmd)		

	for i in range(len(img_structure)):
		for j in range(len(img_structure[i])):		

			figname = img_structure[i][j]
			composite_sh(window_size*j,
						 window_size*i,
						figname,
						output_fig)		
	
def composite_sh(shift_H,
				 shift_V,
				 panel_name,
				 figname):

	cmd="composite  -colorspace srgb -geometry +"+str(shift_H)+"+"+str(shift_V)+" "+panel_name+" "+figname+" "+figname
	os.system(cmd)

