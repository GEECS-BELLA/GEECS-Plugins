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
import os
import sys
import configparser

# define Python user-defined exceptions
class UDPCommunicationError(Exception):
    "Raised when the UDP communication with a device fails"
    pass

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
                ):
        """ 
        Parameters
        ----------
        device_name : name of a GEECS device

        variables: variables associated with device. Should be in format of [variable_name, [min_value,max_value], 
            Boolean, tcp_client]
            
        database_ip: ip address of the database

        """

        self.device_name = device_name
        self.busy = 0
        self.newDataFlag = 0
        self.actual_value=None
    
    def echo_dev_name(self):
        print(self.device_name)
        
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
        host = self.database_ip,
        user = self.database_user,
        password = self.database_password)

        selectors=["ipaddress","commport"]
        selectorString=",".join(selectors)

        mycursor = mydb.cursor()
        db_name='loasis'
        select_stmt="SELECT "+selectorString+" FROM "+db_name+".device where name="+'"'+self.device_name+ '"'+";"
        mycursor.execute(select_stmt)
        myresult = list(mycursor.fetchall()[0])

        self.ip = myresult[0]
        self.tcp_port = int(myresult[1])
        bufferSize = 1024
    
    def find_database(self):
        found=False
        while not found:
            os.chdir('..')
            found = os.path.exists('user data')
        os.chdir('user data')
        config = configparser.ConfigParser()
        config.read('Configurations.INI')
        self.database_ip=config['Database']['ipaddress']
        self.database_password=config['Database']['password']
        self.database_user=config['Database']['user']

    def device_initialize(self):
        self.find_database()
        self.database_lookup()
        
    def tcp_close_client(self):
        self.tcp_client.close()
        
    def command(self,command_string,var,**kwargs):
        
        command_accepted=False
        timedout=False
        valid_command = False
        if 'port_bind' in kwargs:
            port_bind=kwargs['port_bind']
        else:
            port_bind=True
            
        timeout=30
        t0=time.monotonic()
        
        ####bit of code below to assemble the UDP message, which is either a 'get' or 'set' command
        try:
            if command_string == 'get':
                MESSAGE = f"{command_string}{var}>>".encode('ascii')
                valid_command = True

            elif command_string == 'set':
                try:
                    if 'value' in kwargs:
                        value=kwargs['value']
                        MESSAGE = f"{command_string}{var}>>{value:.6f}".encode('ascii')
                        valid_command = True
                    else:
                        raise UDPCommunicationError

                except UDPCommunicationError:
                    print('no value passed for set command')

            else:
                raise UDPCommunicationError
        except UDPCommunicationError:
            print('invalid command')
            
        
        while not command_accepted and not timedout and valid_command:

            #create socket for UDP command
            sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
            sock.settimeout(15)
            # get the port number used for the UDP command
            sock.bind(('', 0))
            info = sock.getsockname()[1]

            #send message
            bufferSize = 1024
            sock.sendto(MESSAGE, (self.ip, self.tcp_port))

            msgFromServer = sock.recvfrom(bufferSize)

            t1=time.monotonic()
            if t1-t0>timeout:
                timedout=True
                print(f"{command_string} command timed out")
           
            resp=(msgFromServer[0].decode('ascii')).split(">>")[-1]
            if resp=='accepted':
                command_accepted=True
                print(f"{command_string} command accepted")
            else:
                #print("command rejected")
                pass
                
            time.sleep(0.05)
            
            sock.close()

        if command_accepted and port_bind:
            self.slow_UDP_socket = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
            self.slow_UDP_socket.settimeout(30)
            # to get socket port?
            self.slow_UDP_socket.bind(('', info+1))

            if kwargs['wait_for_response']:
                self.read_slow_UDP()

    
    def read_slow_UDP(self):
        bufferSize = 1024
        msgFromServer = self.slow_UDP_socket.recvfrom(bufferSize)
        msgSlow = "Message from Server {} ".format(msgFromServer[0])
        self.last_slow_udp=msgFromServer[0]
        self.slow_UDP_socket.close()
        return self.last_slow_udp.decode('ascii').split(">>")[-2]

    def get_only_udp(self,var_name,**kwargs):
        if 'port_bind' in kwargs:
            port_bind=kwargs['port_bind']
        else:
            port_bind=True
        self.command('get',var_name, port_bind=port_bind, wait_for_response=False)
#         print(self.last_slow_udp)
#         return self.last_slow_udp.decode('ascii').split(">>")[-2]

    def set_only_udp(self,var_name,value, **kwargs):
        if 'port_bind' in kwargs:
            port_bind=kwargs['port_bind']
        else:
            port_bind=True
        self.command('set',var_name,value=value, port_bind=port_bind, wait_for_response=False)

    def get_and_wait_udp(self,var_name):
        self.command('get',var_name, wait_for_response=True)
        print(self.last_slow_udp)
        return self.last_slow_udp.decode('ascii').split(">>")[-2]
    
    def set_and_wait_udp(self,var_name,value):
        self.command('set',var_name,value=value,wait_for_response=True)
        return self.last_slow_udp.decode('ascii').split(">>")[-2]
    

                
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
                while dt<0.0045:
                    counter=counter+1
                    t0=time.monotonic()
                    ready=select.select([client],[],[],.005 ) #last arguement is timeout in seconds
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
                    #print(dt)
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
    
    
#%% GEECS device class
class OptimizationControl(GEECSDevice):
    """ .

    General usage, with `img` an image from the HTU Gasjet Phasics camera 
       pia = PhasicsImageAnalyzer()
       phase_map = pia.calculate_phase_map(pia.crop_image(img))

              
    Methods
    -------
    set()
    get()

    """
    
    def __init__(self,
                  device_name = None,
                  variable = None,
                 bounds = None,
                 settable = None
                ):
        """ 
        Parameters
        ----------
        device_name : name of a GEECS device

        variables: variables associated with device. Should be in format of [variable_name, [min_value,max_value], 
            Boolean, tcp_client]
            
        database_ip: ip address of the database

        """
        self.device_name = device_name
        self.variable = variable
        self.bounds = bounds
        self.busy = 0
        self.newDataFlag = 0
        self.actual_value=None
        
    def xopt_set(self,value):
        if self.bounds == None:
            print('no bounds defined for this device variable. This is unsafe so set function is disabled')
        else:  
            if self.bounds[0]<=value<=self.bounds[1]:
                print("can set")
                self.set_and_wait_udp(self.variable,value)
            else:
                print("out of bounds")
                
    
