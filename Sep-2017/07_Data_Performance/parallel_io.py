from mpi4py import MPI
import numpy as np
import h5py
import time
import sys

global comm

def par_rd(colw,filename,varname,dim_along):
#parallel read the variable from the file
    f = h5py.File(filename, 'r', driver='mpio', comm=comm)

# get the length of dimension x,y and t.
    length_x=len(f['x'])
    length_y=len(f['y'])
    length_t=len(f['time'])

# divide each dimension over all processes.
    length_x_rank= length_x/ nproc
    length_x_last_rank=length_x -length_x_rank*(nproc-1)

    length_y_rank=length_y / nproc
    length_y_last_rank=length_y -length_y_rank*(nproc-1)

    length_t_rank=length_t / nproc
    length_t_last_rank=length_t -length_t_rank*(nproc-1)
    
# get the current rank id.
    rank = comm.Get_rank()

# link the variable to dset array.
    dset=f[varname]

# synchronization point of all MPI ranks.    
    comm.Barrier()

# start time profile.    
    timestart=MPI.Wtime()

# split along t dimension    
    if dim_along == 0:

# calculate start and end point along t dimension for the current rank.        
        start=rank*length_t_rank
        end=start+length_t_rank
        if rank==nproc-1:
            end=start+length_t_last_rank
            
        print "MPI Rank ",rank,"reading [",start,":",end,",",length_y,",",length_x,"]"
                
        if colw==1:
# conduct the collective read
            with dset.collective:
                temp=dset[start:end,:,:]
# conduct the independent read     
        else:
            temp= dset[start:end,:,:]
                        
# split along y dimension            
    elif dim_along==1:
        start=rank*length_y_rank
        end=start+length_y_rank
        if rank==nproc-1:
            end=start+length_y_last_rank
        print "MPI Rank #",rank,"reading [",length_t,",",start,":",end,",",length_x,"]"
        if colw==1:
            with dset.collective:
                temp=dset[:,start:end,:]
        else :
            temp= dset[:,start:end,:]

# split along x dimension             
    elif dim_along==2:
        start=rank*length_x_rank
        end=start+length_x_rank
        if rank==nproc-1:
            end=start+length_x_last_rank
        print "MPI Rank ",rank,"reading [",length_t,",",length_y,",",start,":",end,"]"
        if colw==1:
            with dset.collective:
                temp=dset[:,:,start:end]
        else :
            temp= dset[:,:,start:end]

# synchronization point of all MPI ranks.            
    comm.Barrier()
    timeend=MPI.Wtime()

    if rank==0:
        if colw==1:
            print "collective read time %f" %(timeend-timestart)
        else :
            print "independent read time %f" %(timeend-timestart)
    f.close()
    print 'MPI rank ',rank,' read data in size ', np.shape(temp)
    return temp

 
if __name__ == '__main__':
    
# initilize MPI communcation world
    comm =MPI.COMM_WORLD

# obtain the total number of MPI processes/ranks
    nproc = comm.Get_size()
    
# read in the flags
    if len(sys.argv)>4:
        colw = int(sys.argv[1])
        filename=str(sys.argv[2])
        varname=str(sys.argv[3])
        dim_along=int(sys.argv[4])

# conduct the parallel read
    temp=par_rd(colw,filename,varname,dim_along)

    
