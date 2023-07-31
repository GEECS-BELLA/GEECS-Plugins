
import numpy as np
from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
import os.path

const_superpath = 'C:/Users/CEDoss/Desktop/Data/'

def GetTDMSWriter(tdmsFilename):
    return TdmsWriter(tdmsFilename)

def WriteAnalyzeDictionary(tdms_writer, camera_name, analyze_dict):
    """
    root_object = RootObject(properties={
        "prop1": "foo",
        "prop2": 3,
    })
    group_object = GroupObject("group_1", properties={
        "prop1": 1.2345,
        "prop2": False,
    })
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    channel_object = ChannelObject("group_1", "channel_1", data, properties={})
    """
    #with TdmsWriter(tdmsFilename) as tdms_writer:
        # Write first segment

    group_name = camera_name
    for item in analyze_dict:
        channel_name = item
        data = np.array([analyze_dict[item]])

        root_object = RootObject(properties={})
        group_object = GroupObject(group_name, properties={"Camera-Name": group_name})
        channel_object = ChannelObject(group_name, channel_name, data, properties={})

        tdms_writer.write_segment([
            root_object,
            group_object,
            channel_object])


def ReadFullTDMSScan(tdmsFilepath):
    return TdmsFile.read(tdmsFilepath)

def ReturnChannelArray(tdms_data, camera_name, array_name):
    group = tdms_data[camera_name]
    channel = group[array_name]
    return channel.data

def CompileFilename(day, month, year, scan):
    dateStr = '{:02d}'.format(month) + '-' + '{:02d}'.format(day) + '-' + '{:02d}'.format(year%100)
    scanStr = 'Scan'+'{:03d}'.format(scan)
    return const_superpath + dateStr + '_' + scanStr + '.tdms'

def CheckExist(tdmsFilepath):
    return os.path.isfile(tdmsFilepath)
