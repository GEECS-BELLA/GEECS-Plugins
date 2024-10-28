#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample use for main spyspark functions, and sample Axon code for the
FLEXLAB team.

Note there is also a simple console mode for spyspark.py, mostly for testing.

Created on Mon Nov 20 00:33:22 2017

@author: rvitti
"""
import spyspark
import re

"""Test zinc, csv, json formats with axon_request and request functions"""
query = 'readAll(site).findAll(s => s->dis != "X1")'
spyspark.axon_request(query, "text/csv")
spyspark.axon_request(query, "text/zinc")
spyspark.axon_request(query, "application/json")
spyspark.request("http://skyspark.lbl.gov/api/flexlab/eval",\
                 """{\n"meta":{"ver":"3.0"},\n\
                       "cols":[{"name":"expr"}],\n\
                       "rows":[{"expr":"readAll(site)"}]}"""
                ,"application/json","application/json")

"""Create a new site through Axon code"""
# Write Axon expression (can only be one line for REST Eval operation)
# Axon's commit function takes a grid of change descriptions, provided by diff
# Axon's diff function takes three arguments:
#  1) record (Dict) to be modified, or null if creating a new record
#  2) Dict of tags to add to or modify in an existing record, or create new
#  3) Flag {add} to create a new record, or null to modify an existing one
# A site record must have the site tag, a display string, and a timezone
create_site_str = 'commit(diff(null, \
                               {dis: "FLEXLAB Test Site", \
                                geoCountry: "US", \
                                site, \
                                tz: "GMT+8"}, \
                               {add}))'

# Send Axon expression string through REST Eval
site_created = spyspark.axon_request(create_site_str, "text/csv")

# Parse id of newly created site
site_id = re.match(r"(^@p:\w+:r:\w{8}-\w{8}).+", \
                   site_created.split('\n')[1]) \
            .group(1)
            
# Show results
print(f"New site created, with id: {site_id}\n")



"""Create new sites with commit and Zinc file"""
# Write Zinc file with grid of new site(s)
# Grid meta has commit tag with "add" string
# We choose to give these sites an additional "tmp" tag
create_sites_zinc = \
"""\
ver:"3.0" commit:"add"
dis,site,tmp,tz
"Other Test Site 1",M,M,"Los_Angeles"
"Other Test Site 2",M,M,"GMT+8"
"""

# Send Zinc contents through REST Commit 
more_sites_created = spyspark.commit(create_sites_zinc)

# Show results
print(f"More sites created, result: {more_sites_created}")



"""Remove sites with tmp tag through Axon code"""
# Write Axon expression. We use the readAll function to call up all records
# with both the "site" and "tmp" tags. Then we loop through the results and
# use commit + diff to add the "trash" tag to these records.
remove_tmp_sites = 'readAll(site and tmp)\
                    .each(site => commit(diff(site, {trash})))'

# Send Axon expression string through REST Eval
# Return value for this Axon expression will be an empty grid
spyspark.axon_request(remove_tmp_sites, "text/csv")



"""Add two equips to the FLEXLAB Test Site with commit and Zinc file"""
# For equips, instead of using the dis tag, we'll use disMacro, which allows
# us to make use of tags in the display string (e.g. in Historian app)
# We add tmp tag to only one of two equips
create_equips_zinc = \
f"""\
ver:"3.0" commit:"add"
navName,disMacro,equip,siteRef,tmp
"Acquisition Controller","\$siteRef \$navName",M,{site_id},
"Backup Controller","\$siteRef \$navName",M,{site_id},M
"""

# Send Zinc contents through REST Commit 
equips_created = spyspark.commit(create_equips_zinc)

# Show results
print(f"Equips created, result: {equips_created}")



"""Retrieve id of desired equip"""
# If not already parsed and stored from previous operations, we have to look
# up the id of an equip before adding points to it. Here we'll query based
# on site id and navName
equip_id_query = f'read(equip and siteRef=={site_id} \
                        and navName=="Acquisition Controller")->id'

# Send Axon expression string through REST Eval
equip_result = spyspark.axon_request(equip_id_query, "text/csv")

# Parse id of newly created site
equip_id = re.match(r"(^@p:\w+:r:\w{8}-\w{8}).+", \
                    equip_result.split('\n')[1]) \
             .group(1)
            
# Show results
print(f"Target equip has id: {equip_id}\n")



"""Create a new point on target equip through Axon code"""
# Point object needs to specify both a reference to an equip and a reference
# to a site. Hierarchy is not implied. To store history to this point, we
# need to give it the "his" tag, as well as the kind, unit (if kind is Number),
# and tz
create_point_str = f'commit(diff(\
  null, \
  {{navName: "Zone Temperature A", disMacro: "\$equipRef \$navName",\
   zone, air, temp, sensor, point,\
   siteRef: {site_id}, equipRef: {equip_id},\
   his, kind: "Number", unit: "°F", tz: "GMT+8"}}, \
  {{add}}))'

# Send Axon expression string through REST Eval
spyspark.axon_request(create_point_str, "text/csv")



"""Create more points through Zinc and commit operation"""
# Two more zone temperature sensors, with the same tags
# And one with millisecond precision
# Here python's encode seems needed for proper encoding of ° symbol
create_points_zinc = \
f"""\
ver:"3.0" commit:"add"
navName,disMacro,zone,air,temp,sensor,point,\
  siteRef,equipRef,his,kind,unit,tz,hisTsPrecision
"Zone Temperature B","\$equipRef \$navName",M,M,M,M,M,\
  {site_id},{equip_id},M,"Number","°F","GMT+8",
"Zone Temperature C","\$equipRef \$navName",M,M,M,M,M,\
  {site_id},{equip_id},M,"Number","°F","GMT+8",
"Zone Temperature D","\$equipRef \$navName",M,M,M,M,M,\
  {site_id},{equip_id},M,"Number","°F","GMT+8",1ms
"""\
.encode('utf-8')

# Send Zinc contents through REST Commit 
spyspark.commit(create_points_zinc)



"""Retrieve id of point with 1ms precision and write history to it"""
# Axon query on site id and hisTsPrecision
ms_point_id_query = f'read(point and siteRef=={site_id} \
                           and hisTsPrecision==1ms)->id'

# Send Axon expression string through REST Eval
ms_pt_result = spyspark.axon_request(ms_point_id_query, "text/csv")

# Parse id of point
ms_pt_id = re.match(r"(^@p:\w+:r:\w{8}-\w{8}).+", \
                    ms_pt_result.split('\n')[1]) \
             .group(1)
            
# Show results
print(f"Point with millisecond precision has id: {ms_pt_id}\n")

# Create zinc file with history data to write to the point
# Explicit utf-8 encoding is needed with my current setup
his_zinc_ms = \
f"""\
ver:"3.0" id:{ms_pt_id}
ts,val
2019-08-20T10:40:00.025-08:00 GMT+8,72.1°F
2019-08-20T10:40:00.065-08:00 GMT+8,72.0°F
2019-08-20T10:45:00.030-08:00 GMT+8,71.9°F
2019-08-20T10:50:00.030-08:00 GMT+8,72.1°F
2019-08-20T10:55:00.521-08:00 GMT+8,72.0°F
2019-08-20T11:00:00.030-08:00 GMT+8,72.5°F
2019-08-20T11:05:00.350-08:00 GMT+8,72.5°F
2019-08-20T11:10:00.200-08:00 GMT+8,72.4°F
2019-08-20T11:15:00.031-08:00 GMT+8,72.6°F
"""\
.encode('utf-8')

# Send history data in Zinc format through hisWrite REST operation
print(spyspark.his_write(his_zinc_ms))

print(spyspark.axon_request(f'readById({ms_pt_id}).hisRead(2019-08)',\
                            "text/csv"))



"""Retrieve ids of other points"""
# Axon query on site id and hisTsPrecision, with readAll
points_id_query = f'readAll(point and siteRef=={site_id} \
                            and not hisTsPrecision)'

# Send Axon expression string through REST Eval
pts_result = spyspark.axon_request(points_id_query, "text/csv")

# Parse ids of points
pt1_id = re.match(r"(^@p:\w+:r:\w{8}-\w{8}).+", \
                  pts_result.split('\n')[1]) \
           .group(1)
pt2_id = re.match(r"(^@p:\w+:r:\w{8}-\w{8}).+", \
                  pts_result.split('\n')[2]) \
           .group(1)
pt3_id = re.match(r"(^@p:\w+:r:\w{8}-\w{8}).+", \
                  pts_result.split('\n')[3]) \
           .group(1)



"""Write some history to all points, one at a time"""
his_zinc_pt1 = \
f"""\
ver:"3.0" id:{pt1_id}
ts,val
2019-08-20T10:40:00-08:00 GMT+8,1.0°F
2019-08-20T10:42:00-08:00 GMT+8,2.0°F
2019-08-20T10:45:00-08:00 GMT+8,3.0°F
2019-08-20T10:51:00-08:00 GMT+8,4.0°F
2019-08-20T10:55:00-08:00 GMT+8,5.0°F
2019-08-20T11:00:00-08:00 GMT+8,6.0°F
2019-08-20T11:05:00-08:00 GMT+8,7.0°F
2019-08-20T11:10:00-08:00 GMT+8,8.0°F
2019-08-20T11:15:00-08:00 GMT+8,9.0°F
2019-08-20T11:20:00-08:00 GMT+8,10.0°F
"""\
.encode('utf-8')

his_zinc_pt2 = \
f"""\
ver:"3.0" id:{pt2_id}
ts,val
2019-08-20T10:40:00-08:00 GMT+8,10.0°F
2019-08-20T10:50:00-08:00 GMT+8,12.0°F
2019-08-20T10:55:00-08:00 GMT+8,15.0°F
2019-08-20T11:00:00-08:00 GMT+8,15.0°F
2019-08-20T11:10:00-08:00 GMT+8,15.0°F
2019-08-20T11:20:00-08:00 GMT+8,15.0°F
"""\
.encode('utf-8')

his_zinc_pt3 = \
f"""\
ver:"3.0" id:{pt3_id}
ts,val
2019-08-20T10:40:00-08:00 GMT+8,0.0°F
2019-08-20T10:45:00-08:00 GMT+8,0.0°F
2019-08-20T10:50:00-08:00 GMT+8,0.0°F
2019-08-20T10:55:00-08:00 GMT+8,10.0°F
2019-08-20T11:00:00-08:00 GMT+8,10.0°F
2019-08-20T11:05:00-08:00 GMT+8,10.0°F
2019-08-20T11:10:00-08:00 GMT+8,0.0°F
2019-08-20T11:15:00-08:00 GMT+8,0.0°F
2019-08-20T11:20:00-08:00 GMT+8,0.0°F
"""\
.encode('utf-8')

# Send history data in Zinc format through hisWrite REST operation
spyspark.his_write(his_zinc_pt1)
spyspark.his_write(his_zinc_pt2)
spyspark.his_write(his_zinc_pt3)



"""Manipulation of history data"""
# The following steps display results one by one, but store nothing in the DB
# Grouping the data from sensors A, B and C shows that the data is not aligned
data_query = f'readAll(point and siteRef=={site_id} \
                       and zone and air and temp and \
                       not hisTsPrecision)\
               .hisRead(2019-08-20)'
print(spyspark.axon_request(data_query, "text/csv"))

# To perform calculations on the data, we roll it up first to align timestamps
data_query = f'readAll(point and siteRef=={site_id} \
                       and zone and air and temp and \
                       not hisTsPrecision)\
               .hisRead(2019-08-20)\
               .hisRollup(avg, 10min)'
print(spyspark.axon_request(data_query, "text/csv"))

# We can filter data out to remove rows that have any missing values
data_query = f'readAll(point and siteRef=={site_id} \
                       and zone and air and temp and \
                       not hisTsPrecision)\
               .hisRead(2019-08-20)\
               .hisRollup(avg, 10min)\
               .findAll(row => row.remove("ts").all(v => v != null))'
print(spyspark.axon_request(data_query, "text/csv"))

# Now we can fold all the columns into one summary value
data_query = f'readAll(point and siteRef=={site_id} \
                       and zone and air and temp and \
                       not hisTsPrecision)\
               .hisRead(2019-08-20)\
               .hisRollup(avg, 10min)\
               .findAll(row => row.remove("ts").all(v => v != null))\
               .hisFoldCols(avg)'
print(spyspark.axon_request(data_query, "text/csv"))



"""Remove historical data points"""
# One at a time
remove_one_ts = f'readById({pt3_id})\
                  .hisRemove(dateTime(date(2019,08,20),\
                                      time(11,20,00),\
                                      "GMT+8"))'
spyspark.axon_request(remove_one_ts, "text/csv")

# In a specific range of time
remove_range_ts = f'readById({pt1_id})\
                    .hisRemove(dateTime(date(2019,08,20),\
                                        time(10,45,00),\
                                        "GMT+8")..\
                               dateTime(date(2019,08,20),\
                                        time(11,00,00),\
                                        "GMT+8")\
                              )'
spyspark.axon_request(remove_range_ts, "text/csv")

# In a range that covers full months only, for faster and cleaner removal
remove_months = f'readById({pt2_id})\
                  .hisClear(2019-01-01..2019-08-31)'
spyspark.axon_request(remove_months, "text/csv")



"""Create two sample calibration records using Zinc and commit"""
create_calib_zinc = \
f"""\
ver:"3.0" commit:"add"
ptRef,calibration,disMacro,offset,start
{pt1_id},M,"Cal \$ptRef after \$start",0.23°F,2019-08-10T10:00:00-08:00 GMT+8
{pt2_id},M,"Cal \$ptRef after \$start",0.55°F,2019-08-10T10:00:00-08:00 GMT+8
{pt3_id},M,"Cal \$ptRef after \$start",0.02°F,2019-08-10T10:00:00-08:00 GMT+8
{pt3_id},M,"Cal \$ptRef after \$start",-0.12°F,2019-08-20T10:00:00-08:00 GMT+8
"""\
.encode('utf-8')

# Send Zinc contents through REST Commit 
spyspark.commit(create_calib_zinc)
