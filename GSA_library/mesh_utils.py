import os
import sys

import numpy as np
import pyvista as pv
import vtk

def read_pts(filename):
	print('Reading '+filename+'...')
	return np.loadtxt(filename, dtype=float, skiprows=1)

def read_elem(filename,el_type='Tt',tags=True):
	print('Reading '+filename+'...')

	if el_type=='Tt':
		if tags:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4,5))
		else:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4))
	elif el_type=='Tr':
		if tags:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3,4))
		else:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3))
	elif el_type=='Ln':
		if tags:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2,3))
		else:
			return np.loadtxt(filename, dtype=int, skiprows=1, usecols=(1,2))
	else:
		raise Exception('element type not recognised. Accepted: Tt, Tr, Ln')

def read_IGB_file(igbfname:str):
	header_size = 256
	parsed_header = {}
	try:
		with open(igbfname,'rb') as f:
			header = f.read(header_size)
		header = header.decode("utf-8")
		for jj in header.strip().split():
			[key,val]=jj.split(':')
			if(val.isdigit()):
				parsed_header[key]=int(val)
			else:
				parsed_header[key]=val
		#now read the data and create an array
		with open(igbfname,'rb') as f:
			y = np.fromfile(f,'f4')
		y		= y[header_size:]
		nt	   = parsed_header['t']
		nx	   = parsed_header['x']
		nentries = y.shape[0]

		ntot	 = nt*nx
		if parsed_header['type']=='vec3f':
			ntot *= 3
			if(nentries==ntot):
				y = np.reshape(y,(nt,nx,3))

		else:
			if(nentries==ntot):
				y = np.reshape(y,(nt,nx))
			elif(nentries>ntot):
				print('Warning: discarding the last {0} elements (problems in igb file)'.format(nentries-ntot))
				y = y[:ntot]
			else:  #nentries<ntot
				nt=nentries//nx
				if(nt==0):
					print('ERROR: y too short! ({0} elements; expected {1} (problems in igb file)'.format(nentries,ntot))
					sys.exit()
				else:
					ntot	 = nt*nx
					y		= y[:ntot]
					print('Warning: missing {0} elements to reach {1}(problems in igb file); reshaping to {2} time steps'.format(nentries%nx,parsed_header['t'], nt))
					parsed_header['t'] = nt

			y = np.reshape(y,(nt,nx))

		return parsed_header,y
	except ValueError:
		print('error with {0}'.format(igbfname) )

def read_binary_mesh(meshname):

	cmd = ["meshtool convert","-imsh="+meshname,"-omsh="+meshname,"-ifmt=carp_bin","-ofmt=carp_txt"]
	cmd_str = ' '.join(cmd)
	os.system(cmd_str)

	pts = read_pts(meshname+".pts")
	elem = read_elem(meshname+".elem",el_type='Tt',tags=True)

	os.system("rm "+meshname+".pts")
	os.system("rm "+meshname+".elem")
	os.system("rm "+meshname+".lon")

	return pts,elem

def carp_to_pyvista(pts,elem):

	if elem.shape[1]==5:
		elem = elem[:,:4]

	tets = np.column_stack((np.ones((elem.shape[0],),dtype=int)*4,elem)).flatten()
	cell_type = np.ones((elem.shape[0],),dtype=int)*vtk.VTK_TETRA	

	plt_msh = pv.UnstructuredGrid(tets,cell_type,pts)

	return plt_msh