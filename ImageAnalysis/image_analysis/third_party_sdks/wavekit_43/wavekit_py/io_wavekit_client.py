#!/usr/bin/python

import os, sys
import ctypes
import numpy

import io_thirdparty_load_library as imop_library

import io_wavekit_structure as imop_struct
import io_wavekit_hasoslopes as imop_hslp
import io_wavekit_enum as imop_enum
import io_wavekit_image as imop_image


class Client(object):
    """Class Client
    
    - Constructor from parameters :
        - **ip** - string : Server ip adress
        - **port** - ushort : Server port
        - **timeout** - uint : Connection Time out
    """

    def __init_(self, ip, port, timeout):
        """Client constructor
        
        :param ip: Server ip adress
        :type ip: string
        :param port: Server port
        :type port: ushort
        :param timeout: Timeout
        :type timeout: uint
        """
        try:
            message = ctypes.create_string_buffer(256)
            self.dll.Imop_Client_New(
                message,
                ctypes.pointer(self.client),
                ctypes.c_char_p(ip.encode("utf-8")),
                ctypes.c_ushort(port),
                ctypes.c_uint(timeout),
            )
            if message.value != "" and message.value != b"":
                raise Exception("IO_Error", message.value)
        except Exception as exception:
            raise Exception(__name__ + " : init", exception)

    def __init__(self, **kwargs):
        """Client constructor
        """
        self.client = ctypes.c_void_p()
        self.dll = imop_library.load_dll()
        entered = 0
        arg_size = len(kwargs)
        try:
            if arg_size == 3:
                if "ip" in kwargs and "port" in kwargs and "timeout" in kwargs:
                    entered = 1
                    self.__init_(kwargs["ip"], kwargs["port"], kwargs["timeout"])
        except Exception as exception:
            raise Exception(__name__ + " : init", exception)
        if entered == 0:
            raise Exception("IO_Error", "---CAN NOT CREATE CLIENT OBJECT---")

    def __del_obj__(self):
        """CameraSet destructor
        """
        message = ctypes.create_string_buffer(256)
        self.dll.Imop_Client_Delete(message, self.client)
        if message.value != "" and message.value != b"":
            raise Exception("IO_Error", message.value)

    def __del__(self):
        self.__del_obj__()
        imop_library.free_dll(self.dll._handle)

    def get_haso_serial_number(self):
        """Get Haso serial number.
        
        :return: Haso serial number 
        :rtype: string
        """
        try:
            message = ctypes.create_string_buffer(256)
            serial_out = ctypes.create_string_buffer(256)
            self.dll.Imop_Client_GetHasoSerialNumber(message, self.client, serial_out)
            if message.value != "" and message.value != b"":
                raise Exception("IO_Error", message.value)
            return serial_out.value.decode("utf-8")
        except Exception as exception:
            raise Exception(__name__ + " : get_haso_serial_number", exception)

    def get_data_number(self):
        """Get Number of data server can send
        
        :return: Number of data
        :rtype: int
        """
        try:
            message = ctypes.create_string_buffer(256)
            size_out = ctypes.c_int()
            self.dll.Imop_Client_GetDataNumber(
                message, self.client, ctypes.byref(size_out)
            )
            if message.value != "" and message.value != b"":
                raise Exception("IO_Error", message.value)
            return size_out.value
        except Exception as exception:
            raise Exception(__name__ + " : get_data_number", exception)

    def get_data_name_and_type(self, index):
        """Get Data name and type at corresponding index
        
        :param index: Data index
        :type index: int
        
        :return: Data name, Data type
        :rtype: tuple(string, int)
        
        .. seealso:: E_TYPES
        """
        try:
            message = ctypes.create_string_buffer(256)
            name_out = ctypes.create_string_buffer(256)
            type_out = ctypes.c_int()
            self.dll.Imop_Client_GetDataNameAndType(
                message,
                self.client,
                ctypes.c_int(index),
                name_out,
                ctypes.byref(type_out),
            )
            if message.value != "" and message.value != b"":
                raise Exception("IO_Error", message.value)
            return (name_out.value.decode("utf-8"), type_out.value)
        except Exception as exception:
            raise Exception(__name__ + " : get_data_name_and_type", exception)

    def get_data_value(self, index):
        """Get Data at corresponding index
        
        :param index: Data index
        :type index: int
        
        :return: Data value, Data Id
        :rtype: tuplet(int or double or bool or string or HasoSlopes or Image, int)
        """
        try:
            message = ctypes.create_string_buffer(256)
            data_name, data_type = self.get_data_name_and_type(index)
            if data_type == imop_enum.E_TYPES.BOOL:
                data_value = ctypes.c_bool()
                data_id = ctypes.c_int()
                self.dll.Imop_Client_GetBool(
                    message,
                    self.client,
                    ctypes.c_char_p(data_name.encode("utf-8")),
                    ctypes.byref(data_value),
                    ctypes.byref(data_id),
                )
                if message.value != "" and message.value != b"":
                    raise Exception("IO_Error", message.value)
                return (data_value.value, data_id.value)
            if data_type == imop_enum.E_TYPES.INT:
                data_value = ctypes.c_int()
                data_id = ctypes.c_int()
                self.dll.Imop_Client_GetInt(
                    message,
                    self.client,
                    ctypes.c_char_p(data_name.encode("utf-8")),
                    ctypes.byref(data_value),
                    ctypes.byref(data_id),
                )
                if message.value != "" and message.value != b"":
                    raise Exception("IO_Error", message.value)
                return (data_value.value, data_id.value)
            if data_type == imop_enum.E_TYPES.REAL:
                data_value = ctypes.c_float()
                data_id = ctypes.c_int()
                self.dll.Imop_Client_GetReal(
                    message,
                    self.client,
                    ctypes.c_char_p(data_name.encode("utf-8")),
                    ctypes.byref(data_value),
                    ctypes.byref(data_id),
                )
                if message.value != "" and message.value != b"":
                    raise Exception("IO_Error", message.value)
                return (data_value.value, data_id.value)
            if data_type == imop_enum.E_TYPES.STRING:
                data_value = ctypes.create_string_buffer(256)
                data_id = ctypes.c_int()
                self.dll.Imop_Client_GetString(
                    message, 
                    self.client, 
                    ctypes.c_char_p(data_name.encode("utf-8")), 
                    data_value, 
                    ctypes.byref(data_id)
                )
                if message.value != "" and message.value != b"":
                    raise Exception("IO_Error", message.value)
                return (data_value.value.decode("utf-8"), data_id.value)
            if data_type == imop_enum.E_TYPES.SLOPES:
                sn = self.get_haso_serial_number()
                data_value = imop_hslp.HasoSlopes(
                    dimensions=imop_struct.dimensions(
                        imop_struct.uint2D(1, 1), imop_struct.float2D(1.0, 1.0)
                    ),
                    serial_number=sn,
                )
                data_id = ctypes.c_int()
                self.dll.Imop_Client_GetSlopes(
                    message,
                    self.client,
                    ctypes.c_char_p(data_name.encode("utf-8")),
                    data_value.hasoslopes,
                    ctypes.byref(data_id),
                )
                if message.value != "" and message.value != b"":
                    raise Exception("IO_Error", message.value)
                return (data_value, data_id.value)
            if data_type == imop_enum.E_TYPES.IMAGE:
                data_value = imop_image.Image(
                    size=imop_struct.uint2D(1, 1), bit_depth=16
                )
                data_id = ctypes.c_int()
                self.dll.Imop_Client_GetImage(
                    message,
                    self.client,
                    ctypes.c_char_p(data_name.encode("utf-8")),
                    data_value.image,
                    ctypes.byref(data_id),
                )
                if message.value != "" and message.value != b"":
                    raise Exception("IO_Error", message.value)
                return (data_value, data_id.value)
            raise Exception("IO_Error", "Unknown parameter type")
        except Exception as exception:
            raise Exception(__name__ + " : get_data_value", exception)

