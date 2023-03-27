import numpy as np
import itertools

file_path = '/Users/loasis/Desktop/chicaneScan.txt'
scan_number = 17
devices = [ 'U_ChicaneTDK','U_ChicaneTDK',]
variables = ['Current_Limit.Ch1','Current_Limit.Ch2']

energy=100 #Energy in MeV
R56_Start=0
R56_End=600
num_steps=20
R56s=np.linspace(R56_Start,R56_End,num_steps)
currents=np.sqrt(energy**2*R56s/560968.636)
n_shots_per_scan = 10

with open(file_path, 'w+') as f:
    f.write('[Scan' + str(scan_number) + ']\n')
    f.write('Device = "' + ','.join(devices) + '"\n')
    f.write('Variable = "' + ','.join(variables) + '"\n')
    f.write('Values:#shots = "' )
    for i in range(len(currents)):
        f.write('('+str(round(currents[i]*1000)/1000)+","+str(round(-currents[i]*1000)/1000)+'):' + str(n_shots_per_scan) + '|')