from mpi4py import MPI
import numpy as np
import h5py
import time
import sys
import os
from tempfile import *

tempdir=gettempdir()
filename=tempdir+'/collective.h5'

#default is collective write
col_idp='col' 

# initilize MPI communcation world
comm =MPI.COMM_WORLD

# obtain the total number of MPI processes/ranks
nproc = comm.Get_size()

# get the current rank id
rank = comm.Get_rank()

# specify the dimension length
length_y = 512*512
length_x = 1024

# read in execution flags
if len(sys.argv)>2:
	col_idp = str(sys.argv[1])
	dim_along=str(sys.argv[2])
else:
	print 'please give 2 flags'
	exit(1)

# split x dimenion over all MPI processes
length_x_rank=length_x / nproc
length_x_last_rank=length_x -length_x_rank*(nproc-1)
start_x=rank*length_x_rank
end_x=start_x+length_x_rank
if rank==nproc-1:
    end_x=start_x+length_x_last_rank

# split y dimension over all MPI processes    
length_y_rank=length_y / nproc
length_y_last_rank=length_y -length_y_rank*(nproc-1)
start_y=rank*length_y_rank
end_y=start_y+length_y_rank
if rank==nproc-1:
    end_y=start_y+length_y_last_rank

# open the file in parallel write mode    
f = h5py.File(filename, 'w', driver='mpio', comm=MPI.COMM_WORLD)

# create the variable
dset = f.create_dataset('test', (length_y,length_x), dtype='f8')

# disable the atomic access
f.atomic = False

# synchronization point of all MPI ranks
comm.Barrier()

# begin time profiling
timestart=MPI.Wtime()

# write variable along x dimension, which is the contiguous access
if dim_along=='x':
	access_type='contiguous'
	temp=np.random.random((end_y-start_y,length_x))
	if col_idp=='col':
		with dset.collective:
			dset[start_y:end_y,:] = temp
	else :
		dset[start_y:end_y,:] = temp
	f.close()

# write variable along y dimension, which is the non-contiguous access    
elif dim_along=='y':
	access_type='non-contiguous'
	temp=np.random.random((length_y,end_x-start_x))
	if col_idp=='col':
		with dset.collective:
			dset[:,start_x:end_x] = temp
	else :
		dset[:,start_x:end_x] = temp

#       
# close the file    
	f.close()
    
# synchronization point of all MPI ranks    
comm.Barrier()

# end of the time profiling
timeend=MPI.Wtime()

# print results
if rank==0:
    if col_idp=='col':
    	print "collective "+access_type+" write time %f" %(timeend-timestart)
    else :
	print "independent "+access_type+" write time %f" %(timeend-timestart)
#    print "data size x: %d y: %d" %(length_x, length_y)
#    print "file size ~%d GB" % (length_x*length_y/1024.0/1024.0/1024.0*8.0)
#    print "number of processes %d" %nproc

# open the file in parallel read mode
f = h5py.File(filename, 'r', driver='mpio', comm=MPI.COMM_WORLD)

# link the variable test to dset array
dset=f['test']

# disable atomic access
f.atomic = False

# synchronization point of all MPI ranks
comm.Barrier()

# start the time profiling
timestart=MPI.Wtime()

# read data long x dimension, which is contiguous access
if dim_along=='x':
	access_type='contiguous'
	if col_idp=='col':
		with dset.collective:
			temp=dset[start_y:end_y,:]
	else :
		temp= dset[start_y:end_y,:]
        
# read data long y dimension, which is non-contiguous access        
elif dim_along=='y':    
	access_type='non-contiguous'
	if col_idp=='col':
		with dset.collective:
			temp=dset[:,start_x:end_x]
	else :
		temp= dset[:,start_x:end_x]
        
# synchronization point of all MPI ranks     
comm.Barrier()

# end of time profiling
timeend=MPI.Wtime()

# print out the results
if rank==0:
    if col_idp=='col':
    	print "collective "+access_type+" read time %f" %(timeend-timestart)
    else :
	print "independent "+access_type+" read time %f" %(timeend-timestart)
#    print "data size x: %d y: %d" %(length_x, length_y)
#    print "file size ~%d GB" % (length_x*length_y/1024.0/1024.0/1024.0*8.0)
#    print "number of processes %d" %nproc
f.close()

