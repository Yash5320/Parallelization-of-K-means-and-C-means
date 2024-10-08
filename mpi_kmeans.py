import csv, time, random, math
# from mpi4py import MPI
import mpi4py

def eucl_distance(point_one, point_two):# the function to find distance between 2 points
	if(len(point_one) != len(point_two)):
		raise Exception("Error: non comparable points")

	sum_diff = 0.0
	for i in range(len(point_one)):
		diff = pow((float(point_one[i]) - float(point_two[i])), 2)
		sum_diff += diff
	final = math.sqrt(sum_diff)
	return final

def main():
        comm = mpi4py.MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()	
	
        global cutoff, dim, dataset, num_clusters, data
        if rank == 0:
                print ("Enter the number of clusters you want to make:")
                num_clusters = int(input())# entering k values
                with open('modified.csv', 'r') as f:
                        reader = csv.reader(f)
                        dataset = list(reader)
                initial = []
                dataset.pop(0)		#removing headers
                data = dataset
                for i in range(num_clusters):
                        initial.append(dataset[i])#appending first k points as cluster centers
                #	dataset.pop(0)
                num_points = len(dataset)
                dim = len(dataset[0])
        else:
                initial = None# cluster centers and points with thread other than root is set to empty
                data = None 	
        cutoff = 0.2
        loop = 0
        compare_cutoff = True# to check if loop termination condition reached
        while compare_cutoff:
                loop += 1	
                clusters = []#declaring empty clusters
                strpt = comm.bcast(initial, root = 0)#root broadcasting cluster center to all threads
                recv = comm.scatter(data, root = 0)# root scattering/sending and splitting data to rest of the processes
                least = eucl_distance(strpt[0], recv)#taking random min value just for comparision
                for i in range(len(strpt)):
                        clusters.append([])	#initialisng each cluster as empty
                lpoint = 0 
                for i in range(len(strpt)):#comparing datapoint to each cluster center
                        a = eucl_distance(strpt[i], recv)
                        if a < least :
                                least = a
                                lpoint = i
                clusters[lpoint]= recv #fixing point int cluster
                fc = comm.gather(clusters, root = 0)#gathering all clusters 
                
                if rank == 0:	#master process
                        nfc = [] #to store each cluster center
                        no = [] #count of elements in each cluster
                        for i in range(len(initial)):#for i ranges from 0 to k-1
                                nfc.append(['0'] * dim) #cluster centroid intialised as zero for each cluster
                                no.append('0') #intially each cluster is empty hence zero
                        for i in range(len(fc)): #for each gathered cluster
                                for j in range(len(fc[i])): #checking returned clusters
                                        if len(fc[i][j]) != 0:#if that returned cluster group cluster is not empty
                                                no[j] = int(no[j]) + 1#increase that cluster count by 1
                                                for k in range(len(fc[i][j])):#for each dim element
                                                        nfc[j][k] = float(nfc[j][k]) + float(fc[i][j][k])	#add it to cluster center			
							
									
                        for i in range(len(nfc)):# now we have cluster sums
                                for j in range(len(nfc[i])):
                                        nfc[i][j] = float(nfc[i][j]) / float(no[i])#compute centroid by dividing by mean
                        flag = 0		
                        for i in range(len(nfc)): #finidng no of points passing cutoff
                                if eucl_distance(nfc[i], initial[i]) > cutoff:
                                        flag += 1
                        if flag == 0: #if flag==0 then we have computed optimal centroid
                                compare_cutoff = False
                                print (nfc)
                                compare_cutoff = comm.bcast(compare_cutoff, root = 0)
                                print (fc)			
                                print ("Execution time %s seconds" % (time.time() - start_time))
                                print (loop)
                        else:
                                initial = nfc #exit
        mpi4py.MPI.Finalize()		
        exit(0)	
if __name__ == "__main__":
        start_time = time.time()
        main()
	
