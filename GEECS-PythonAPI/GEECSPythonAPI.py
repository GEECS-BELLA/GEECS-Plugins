# -*- coding: utf-8 -*-
"""
Created on Fri Feb  22 21:33:23 2023

@author: SamBarber
"""

#%% init
import socket
import struct
import mysql.connector
import numpy as np
import time
import select

#%% GEECS device class

class GEECSDevice:
    """ .

    General usage, with `img` an image from the HTU Gasjet Phasics camera 
       pia = PhasicsImageAnalyzer()
       phase_map = pia.calculate_phase_map(pia.crop_image(img))

              
    Methods
    -------
    set(value)
    get()

    """
    
    def __init__(self,
                 device_name = None,
                 variables = None,
                ):
        """ 
        Parameters
        ----------
        device_name : name of a GEECS device

        variables: variables associated with device. Should be in format of [variable_name, [min_value,max_value], 
            Boolean, tcp_client]

        """

        self.device_name = device_name
        self.variables = variables
        self.busy = 0
        self.newDataFlag = 0
        self.actual_value=None
    
    def echo_dev_name(self):
        print(self.device_name)
        print(self.variables)
        
    def create_tcp_subscribing_client(self,var):
        #print('in the client factory for device: ',var)
        client=socket.socket(socket.AF_INET, socket.SOCK_STREAM);
        client.connect((str(self.ip),int(self.tcp_port)))
        #client.setblocking(0)
        subcriptionstring = bytes('Wait>>'+str(var),'ascii')
        #get length of subscription message
        SubcriptionCmdLength = len(subcriptionstring)
        #Flatten the length of the subscription message length
        sizepack = struct.pack('>i', SubcriptionCmdLength)
        #Send the size of the message followed by the message
        client.sendall( sizepack + subcriptionstring)
        self.tcp_client=client
        return client
    
    def database_lookup(self):
        
        mydb = mysql.connector.connect(
        host="192.168.6.14",
        user="loasis",
        password="dat+l0sim")

        selectors=["ipaddress","commport"]
        selectorString=",".join(selectors)

        mycursor = mydb.cursor()
        db_name='loasis'
        select_stmt="SELECT "+selectorString+" FROM "+db_name+".device where name="+'"'+self.device_name+ '"'+";"
        #print(select_stmt)
        mycursor.execute(select_stmt)
        myresult = list(mycursor.fetchall()[0])
        #print(myresult)

        self.ip = myresult[0]
        self.tcp_port = int(myresult[1])
        bufferSize = 1024
        #print(self.ip)
    
    def device_initialize(self):
        self.database_lookup()
        self.variables[3]=self.create_tcp_subscribing_client(self.variables[0])
        
    def set(self,value):
        
        #example message to set something
        MESSAGE = bytes("set"+self.variables[0]+">>" + str(value), 'ascii')
        
        #create socket for UDP command
        sock = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP

        # get the port number used for the UDP command
        sock.bind(('', 0))
        info = sock.getsockname()[1]
        
        #send message
        bufferSize = 1024
        sock.sendto(MESSAGE, (self.ip, self.tcp_port))

        msgFromServer = sock.recvfrom(bufferSize)

        msg = "Message from Server {} ".format(msgFromServer[0])

        print(msg)
        sock.close()


        s = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        #sock.settimeout(0.0001)
        # to get socket port?
        s.bind(('', info+1))
        info = s.getsockname()[1]
        print(info)

        #s.sendto(MESSAGE, (UDP_IP, UDP_PORT))

        msgFromServer = s.recvfrom(bufferSize)

        msgSlow = "Message from Server {} ".format(msgFromServer[0])

        print(msgSlow)
        sock.close()
        
    def get_tcp_nonblocking(self):    
        #info('function get1')

        #start by trying to check out a socket so that when a process calls  
        #to get the value, you don't have multiple attempts to read/clear the buffer.
        #If the socket isn't currently busy, swith it to "busy" until finished
        if self.busy==0: 
            #print("socket was clear when requested")
            self.busy=1
            if False: #skipping
                #print('objective function')
                f(x)
                if hasattr(y, '__iter__'):
                    return y[0]
                else:
                    return y
            else:
                client=self.tcp_client
                #print("got client: ",client)
                dt=0
                counter=0
                #note: the dt defined below should be shorter than the timeout in the select.select command
                #The select.select command asks the client if there is any information to transmist. If there
                #is, it returns true. If there is not any information after the timeout, it reports false.
                #Typical response time when a device has information to transmit is well below 1 ms. So, we rely
                # on the timeout to tell us that there is no information on the buffer, and we are waiting on 
                # another iteration of the device's acquire loop.
                while dt<0.0015:
                    counter=counter+1
                    t0=time.monotonic()
                    ready=select.select([client],[],[],.002 ) #last arguement is timeout in seconds
                    #print(ready)
                    if ready[0]:
                        size = struct.unpack('>i', client.recv(4))[0]  # Extract the msg size from four bytes - mind the encoding
                        str_data = client.recv(size)
                        geecs=str_data.decode('ascii').split(",")
                        #print(geecs)
                        geecs=geecs[-2].split(" ")[0]
                        #print(geecs)
                        if len(geecs)==0:
                            geecs="nan"
                        if geecs=='on':
                            geecs=1
                        if geecs=='off':
                            geecs=0
                        #print(geecs)
                        if type(geecs) ==  str:
                            if any(c.isalpha() for c in geecs):
                                geecs=0
                        self.actual_value=geecs
                        self.newDataFlag=1
                        #print("chewing through TCP buffer. Device value: ",geecs)
                    else:
                        #print("Buffer cleared")
                        if counter==1:
                            geecs=self.actual_value
                            self.newDataFlag=0
                    t1=time.monotonic()
                    dt=t1-t0
                self.busy=0 #release the socket
                #print("socket released")
        else:
            print("socket was busy when requested")
            geecs=self.actual_value
            self.newDataFlag=0
            slef.busy=0
            print("new data: ",newDataFlags[index])
        #print("in get1 gotvalue ans index "+str(gotValues[index])+' '+str(index))
        return geecs
    
    def get(self):
        #can change how get is defined but use this function elsewhere
        value=self.get_tcp_nonblocking()
        return value