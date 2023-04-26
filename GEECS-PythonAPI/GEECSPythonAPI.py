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
import datetime

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
    
    def create_udp_client(self):
        self.udp_fast = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        self.udp_fast.settimeout(5)
        # get the port number used for the UDP command
        self.udp_fast.bind(('', 0))
        info = self.udp_fast.getsockname()[1]


        self.udp_slow = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        self.udp_slow.settimeout(6)

        self.udp_slow.bind(('', info+1))
        print(info)

    
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
        self.create_udp_client()
        
    def tcp_close_client(self):
        self.tcp_client.close()
        
    def command(self,command_string,var,**kwargs):
        
        command_accepted=False
        timedout=False
        valid_command = False

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
                        if type(value)==str:
                            MESSAGE = f"{command_string}{var}>>{value}".encode('ascii')
                            valid_command = True
                        else:
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

            #send message
            bufferSize = 1024
            max_tries=3
            for i in range(max_tries):
                try:
                    self.udp_fast.sendto(MESSAGE, (self.ip, self.tcp_port))
                    msgFromServer = self.udp_fast.recvfrom(bufferSize)
                    break
                except Exception:
                    print('udp command not received')
                    continue


            t1=time.monotonic()
            if t1-t0>timeout:
                timedout=True
                print(f"{command_string} command timed out")
           
            resp=(msgFromServer[0].decode('ascii')).split(">>")[-1]
            if resp=='accepted':
                command_accepted=True
                #print(f"{command_string} command accepted")
            else:
                #print("command rejected")
                pass
                
            time.sleep(0.25)

        if command_accepted:
            if kwargs['wait_for_response']:
                self.read_slow_udp()

    
    def read_slow_udp(self):
        bufferSize = 1024
        max_tries=3
        for i in range(max_tries):
            try:
                msgFromServer = self.udp_slow.recvfrom(bufferSize)
                break
            except Exception:
                print('udp command not received')
                continue

        msgSlow = "Message from Server {} ".format(msgFromServer[0])
        self.last_slow_udp=msgFromServer[0]
        return self.last_slow_udp.decode('ascii').split(">>")[-2]

    def get_only_udp(self,var_name,**kwargs):
        self.command('get',var_name, wait_for_response=False)

    def set_only_udp(self,var_name,value, **kwargs):
        self.command('set',var_name,value=value, wait_for_response=False)

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
                
    def xopt_get_udp(self):
        val=float(self.get_and_wait_udp(self.variable))
        return val
                
    
                  
    def xopt_get(self):
        self.create_tcp_subscribing_client(self.variable)
        time.sleep(.5)
        val=float(self.get_tcp_nonblocking())
        
        self.tcp_close_client()
        return val
                              
    
#%% GEECS device class
class ExperimentControl:
    """ .

    General usage,NA
              
    Methods
    -------
    set()
    get()

    """
    
    def __init__(self,
                 MC_IP = '192.168.7.203',
                 MC_remote_port=61561
                ):
        """ 
        Parameters
        ----------

        """
        
        self.MC_IP = MC_IP
        self.MC_remote_port = MC_remote_port

    #function to start a no scan, shot number determined by local MC. Currently hardcoded to use 7.203 computer
    def start_no_scan(self,description):
        mc_udp = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP

        mc_udp.settimeout(5)
        # get the port number used for the UDP command
        mc_udp.bind(('', 0))

        info = mc_udp.getsockname()[1]
        MESSAGE=('ScanStart>>'+description).encode('ascii')
        mc_udp.sendto(MESSAGE, (self.MC_IP, self.MC_remote_port))
        resp=mc_udp.recvfrom(1024)
        print(resp)


    def scan_search(self,scan_num):
        dirs=self.define_data_directories()
        return os.path.exists(dirs[0]+'Scan'+str(scan_num).rjust(3,'0'))

    def scan_finished(self,scan_num):
        dirs=self.define_data_directories()
        return os.path.exists(dirs[1]+'s'+str(scan_num)+'.txt')



    def check_scan_status(self,scan_num):
        t0=time.monotonic()
        while not self.scan_search(scan_num):
            t1=time.monotonic()
            print('waiting for Scan '+str(scan_num)+'to start')
            time.sleep(5)
            if t1-t0>60:
                print('scan failed to start')
                break
        print('Scan '+str(scan_num)+' appears to be have started')
        while not self.scan_finished(scan_num):
            t2=time.monotonic()
            print('waiting for '+str(scan_num)+' to finish')
            time.sleep(5)
            if t2-t1>300:
                print('scan failed to finish properly')
                break
        print('Scan '+str(scan_num)+' appears to have finished')

    def define_data_directories(self):    
        #initialize some date and time stuff, define paths, look for last scan
        current_time = datetime.datetime.now()

        month_dict={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

        month=month_dict[current_time.month]
        monthNum=f'{current_time.month:02}'
        yearLong=str(current_time.year)
        yearShort=yearLong[-2:]
        date=f'{current_time.day:02}'

        #topDir='/Volumes/vol1/data/Undulator/Y'+yearLong+'/'+monthNum+'-'+month+'/'+yearShort+'_'+monthNum+date+'/'
        top_dir='Z:\\data\\Undulator\\Y'+yearLong+'\\'+monthNum+'-'+month+'\\'+yearShort+'_'+monthNum+date+'\\'
        analysis_dir=top_dir+'analysis\\'
        scan_dir=top_dir+'scans\\'
        return [scan_dir,analysis_dir]

    def get_last_scan(self):
        dirs=self.define_data_directories()
        scan_dirs=os.listdir(dirs[0])
        if len(scan_dirs)>0:
            last_scan=int(scan_dirs[-1][-3:])
        else:
            last_scan=0
        return last_scan

    def run_scan(self,description):
        current_scan_number=self.get_last_scan()+1
        self.start_no_scan(description)
        self.check_scan_status(current_scan_number)
    

    