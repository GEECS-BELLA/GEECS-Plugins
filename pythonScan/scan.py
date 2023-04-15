from __future__ import absolute_import, print_function
import os
import numpy as np
import socket
import struct
from collections import OrderedDict
import time
import select
import mysql.connector
import csv
#from multiprocessing import Process
import multiprocessing


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    

def client_factory(ip, port, var):
    print('in the client factory for device: ',var)
    client=socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    client.connect((str(ip),int(port)))
    # client.setblocking(0)
    subscription_string = bytes('Wait>>'+str(var),'ascii')
    # get length of subscription message
    subscription_cmd_length = len(subscription_string)
    # Flatten the length of the subscription message length
    size_pack = struct.pack('>i', subscription_cmd_length)
    # Send the size of the message followed by the message
    client.sendall( size_pack + subscription_string)
    return client


# simple access fcn
def get1(pv_client):
    # info('function get1')
    index = pvdictClients[pv_client]
    if pvBusy[index] == 0:  # trying to check out a socket so that when a process calls to get the value,
        # you don't have multiple attempts to read/clear the buffer
        # print("socket was clear when requested")
        pvBusy[index] = 1

        if index == 1000000:  # skipping
            # print('objective function')
            f(x)
            if hasattr(y, '__iter__'):
                return y[0]
            else:
                return y
        else:
            client = pvClients[index]
            # print("got client: ",client)
            dt = 0
            iter = 0
            while dt < 0.001:
                iter = iter+1
                t0 = time.monotonic()
                ready = select.select([client], [], [], .002 )  # last arguement is timeout in seconds
                # print(ready)
                if ready[0]:
                    size = struct.unpack('>i', client.recv(4))[0]  # Extract the msg size from four bytes
                    # - mind the encoding
                    str_data = client.recv(size)
                    geecs = str_data.decode('ascii').split(",")
                    # print(geecs)
                    geecs = geecs[-2].split(" ")[0]
                    # print(geecs)
                    if len(geecs) == 0:
                        geecs = "nan"
                    if geecs == 'on':
                        geecs = 1
                    if geecs == 'off':
                        geecs = 0
                    print(gotValues[index])
                    # print(geecs)
                    if type(geecs) == str:
                        if any(c.isalpha() for c in geecs):
                            geecs = 0
                    gotValues[index] = geecs
                    print(gotValues[index])
                    newDataFlags[index] = 1
                    # print("chewing through TCP buffer. Device value: ",geecs)
                else:
                    # print("Buffer cleared")
                    if iter == 1:
                        geecs = gotValues[index]
                        newDataFlags[index] = 0
                t1 = time.monotonic()
                dt = t1-t0
            pvBusy[index] = 0  # release the socket
            # print("socket released")
    else:
        print("socket was busy when requested")
        geecs = gotValues[index]
        newDataFlags[index] = 0
        pvBusy[index] = 0
        print("new data: ", newDataFlags[index])
    # print("in get1 gotvalue ans index "+str(gotValues[index])+' '+str(index))
    return geecs


#look for the database info in the COnfigurations.INI file
userData=open("..\\..\\..\\..\\user data\\Configurations.INI")
userDataRead=csv.reader(userData,delimiter="\t")
dbitems=[]
for row in userDataRead:
    if len(row)>0:
        dbitems.append(row[0])


#locate the database portion of the configurations file. could be smarter and 
#pasrse the info based on name rather than index...
start=dbitems.index('[Database]')
dbName=dbitems[start+1].split('=')[-1]
dbIP= dbitems[start+3].split('=')[-1]
dbUser=dbitems[start+4].split('=')[-1]
dbPW=dbitems[start+5].split('=')[-1]


mydb = mysql.connector.connect(
host=dbIP,
user=dbUser,
password=dbPW)

#        mydb = mysql.connector.connect(
#        host="192.168.6.14",
#        user="loasis",
#        password="dat+l0sim")

#config=open("mint/bella/"+configPath)
config=open("HTU.txt")
read_config=csv.reader(config,delimiter="\t")
deviceVars=[]
for row in read_config:
    deviceVars.append(row)

configHeader=deviceVars[0]
deviceVars=deviceVars[1:]

selectors=["ipaddress","commport"]
selectorString=",".join(selectors)

selectors2=["device","enabled"]
selectorString2=",".join(selectors2)

configHeader=configHeader+selectors

# pvs=np.array([])
# pvIPs=np.array([])
# pvPorts=np.array([])
# pvVars=np.array([])
# gotValues=np.array([])
# newDataFlags=np.array([]);
# pvLowLims=np.array([])
# pvHighLims=np.array([])
# pvBusy=np.array([])

# mycursor = mydb.cursor()
# for row in deviceVars:
    # mycursor.execute("SELECT "+selectorString+" FROM "+dbName+".device where name="+'"' + str(row[1]) + '"'+";")
    # #mycursor.execute("SELECT "+selectorString+" FROM loasis.device where name="+'"' + str(row[1]) + '"'+";")
    # myresult = list(mycursor.fetchall()[0])
    # temp=row
    # for i in myresult:
        # temp.append(i)
    # pvs=np.append(pvs,temp[0])
    # pvIPs=np.append(pvIPs,temp[5])
    # pvPorts=np.append(pvPorts,temp[6])
    # pvVars=np.append(pvVars,temp[2])
    # pvLowLims=np.append(pvLowLims,temp[3])
    # pvHighLims=np.append(pvHighLims,temp[4])
    # gotValues=np.append(gotValues,0.0)
    # pvBusy=np.append(pvBusy,0)
    # newDataFlags=np.append(newDataFlags,1)    

# print(pvIPs)
# c=client_factory(pvIPs[0],pvPorts[0],pvVars[0])
# print(c)

# pvClients=np.array([])
# for i in range(len(pvs)):
    # pvClients=np.append(pvClients,client_factory(pvIPs[i],pvPorts[i],pvVars[i]))
    # print(pv)

# time.sleep(1)

# ndims = len(pvs)

#send a command to sql to grab experiment devices from database

mycursor = mydb.cursor()
table1=mycursor.execute("SELECT A.* FROM loasis.expt_device_variable A WHERE A.expt_device_id IN (SELECT B.id from loasis.expt_device B where B.expt= 'Undulator' )")
res1=np.array(list(mycursor.fetchall()))
devIDs=res1[:,1]
devSubVars=res1[:,2]
devSubBool=res1[:,3]


pvs=np.array([])
pvIPs=np.array([])
pvPorts=np.array([])
pvVars=np.array([])
gotValues=np.array([])
newDataFlags=np.array([]);
pvLowLims=np.array([])
pvHighLims=np.array([])
pvBusy=np.array([])
pvDeviceNames=np.array([])
pvDeviceEnabled=np.array([])
#make a list of subscribed devices

for i in range(len(devIDs)):
    if devSubBool[i]=='yes':
        query=mycursor.execute("SELECT "+selectorString2+" FROM "+dbName+".expt_device where id="+'"' + str(devIDs[i]) + '"'+";")
        queryRes= list(mycursor.fetchall())[0]
        mycursor.execute("SELECT "+selectorString+" FROM "+dbName+".device where name="+'"' + queryRes[0] + '"'+";")
        myresult = list(mycursor.fetchall()[0])
        #print(queryRes)
        #print(myresult)
        if queryRes[1]=='yes':
            pvs=np.append(pvs,devSubVars[i])
            pvIPs=np.append(pvIPs,myresult[0])
            pvPorts=np.append(pvPorts,myresult[1])
            pvDeviceEnabled=np.append(pvDeviceEnabled,queryRes[1])
            pvVars=np.append(pvVars,devSubVars[i])
            gotValues=np.append(gotValues,0.0)
            pvBusy=np.append(pvBusy,0)
            pvDeviceNames=np.append(pvDeviceNames,queryRes[0])
            newDataFlags=np.append(newDataFlags,1)  
            

t0=time.monotonic()
pvClients=np.array([])
for i in range(len(pvs)):
    try: 
        print('going to client factory for: ',pvDeviceNames[i])
        print('ip adress',pvIPs[i])
        cq=client_factory(pvIPs[i],pvPorts[i],pvVars[i])
    except:
        cq='NA'
    pvClients=np.append(pvClients,cq)



t1=time.monotonic()
print(t1-t0)

pvdict = dict() # for simple lookup
pvdictIPs = dict()
pvdictPorts = dict()
pvdictVars = dict()
pvdictClients = dict()
pvdictGotVals=dict()


for i in range(len(pvs)):
    pvdict[pvs[i]] = i
    pvdictIPs[pvIPs[i]] = i
    pvdictPorts[pvPorts[i]] = i
    pvdictVars[pvVars[i]] = i
    pvdictGotVals[gotValues[i]]= i
    pvdictClients[pvClients[i]]= i


time.sleep(1)
        
#query=mycursor.execute("SELECT device FROM loasis.expt_device where id="+str(429)
#queryRes= list(mycursor.fetchall())
t0=time.monotonic()


# if __name__ == '__main__':
    # info('main line')
    # p = Process(target=get1, args=(pvs[0]))
    # p.start()
    # p.join()


for i in range(len(pvs)):
    v=get1(pvClients[i])
    print(pvDeviceNames[i]+' '+pvs[i]+' '+str(v))
    print("in loop gotvalue ans i "+str(gotValues[i])+' '+str(i))
    
# pool=multiprocessing.Pool(4)
# test=pool.map(get1,pvClients)


t1=time.monotonic()   
print(gotValues)
print(t1-t0)

t0=time.monotonic()

print(pvs)


for i in pvClients:
    i.close()

# print(pvs[1])
# print(get1(pvs[1]))
# print('first call is done')
# print(pvBusy[1])
# t0=time.monotonic()
# print(get1(pvs[1]))
# t1=time.monotonic()
# print(t1-t0)