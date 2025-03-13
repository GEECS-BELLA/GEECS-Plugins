#!/usr/bin/python

import os, sys 
import ctypes
import numpy

import io_thirdparty_load_library as imop_library
import io_wavekit_enum as imop_enum

class Server(object):
    """Class Server
    
    - Constructor from parameters :
        - **config_file_path** - string : Absolute path to haso configuration file
        - **port** - ushort : Port value
    """
    
    def __init_(
        self,
        config_file_path,
        port
        ):
        """Server constructor from configuration file and port
        
        :param config_file_path: Absolute path to haso configuration file
        :type config_file_path: string
        :param port: Port value
        :type port: ushort
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Server_New(
                message,
                ctypes.pointer(self.server),
                ctypes.c_char_p(config_file_path.encode('utf-8')),
                ctypes.c_ushort(port)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : __init__',exception)

    def __init__(self,  **kwargs):
        """Server constructor
        """
        self.server = ctypes.c_void_p()
        self.dll   = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try :
            if(arg_size == 2):
                if('config_file_path' in kwargs
                    and 'port' in kwargs):
                    entered = 1
                    self.__init_(kwargs['config_file_path'], kwargs['port'])
        except Exception as exception:
            raise Exception(__name__+' : init',exception)
        if(entered == 0):
            raise Exception('IO_Error','---CAN NOT CREATE SERVER OBJECT---')
    
    def __del_obj__(self):
        """Server Destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Server_Delete(message, self.server)
        if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def start(self):
        """Start server
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Server_Start(
                message,
                self.server
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
        except Exception as exception:
            raise Exception(__name__+' : start',exception)

    def add_data(
        self,
        data_name,
        data_type,
        data_value
        ):
        """Configure server to support transfer of data named data_name with type data_type and value data_value
        
        :param data_name: Data name
        :type data_name: string
        :param data_type: Data type
        :type data_type: E_TYPES_T
        :param data_value: Data value
        :type data_value: bool, int, double, string, HasoSlopes or Image
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Server_AddData(
                message,
                self.server,
                ctypes.c_char_p(data_name.encode('utf-8')),
                ctypes.c_int(data_type)
                )
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)
            if(data_type == imop_enum.E_TYPES.BOOL):
                self.dll.Imop_Server_SetBool(
                    message,
                    self.server,
                    ctypes.c_char_p(data_name.encode('utf-8')),
                    ctypes.c_bool(data_value)
                    )
            elif(data_type == imop_enum.E_TYPES.INT):
                self.dll.Imop_Server_SetInt(
                    message,
                    self.server,
                    ctypes.c_char_p(data_name.encode('utf-8')),
                    ctypes.c_int(data_value)
                    )
            elif(data_type == imop_enum.E_TYPES.REAL):
                self.dll.Imop_Server_SetReal(
                    message,
                    self.server,
                    ctypes.c_char_p(data_name.encode('utf-8')),
                    ctypes.c_float(data_value)
                    )  
            elif(data_type == imop_enum.E_TYPES.STRING):
                self.dll.Imop_Server_SetString(
                    message,
                    self.server,
                    ctypes.c_char_p(data_name.encode('utf-8')),
                    ctypes.c_char_p(data_value.encode('utf-8'))
                    )   
            elif(data_type == imop_enum.E_TYPES.SLOPES):
                self.dll.Imop_Server_SetSlopes(
                    message,
                    self.server,
                    ctypes.c_char_p(data_name.encode('utf-8')),
                    data_value.hasoslopes
                    )  
            elif(data_type == imop_enum.E_TYPES.IMAGE):
                self.dll.Imop_Server_SetImage(
                    message,
                    self.server,
                    ctypes.c_char_p(data_name.encode('utf-8')),
                    data_value.image
                    )
            else:
                raise Exception('IO_Error', 'Wrong parameter type')
            if message.value != '' and message.value != b'' : raise Exception('IO_Error',message.value)               
        except Exception as exception:
            raise Exception(__name__+' : add_data',exception)
