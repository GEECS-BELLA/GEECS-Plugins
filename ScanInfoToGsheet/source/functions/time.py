from dateutil.relativedelta import relativedelta
from datetime import datetime
from pytz import timezone

def PT_timestr(lvtimestamp,strformat):
    """
    Conert the labview timestamp to pacific time.
    lvtimestamp: labview timestamp (float). should be 10 digit (36...)
    strformat: format of the string to be returned. ex) "%m/%d/%Y, %H:%M:%S"
    """
    
    lv_dt = datetime.fromtimestamp(lvtimestamp) #labview time
    utc_dt = lv_dt - relativedelta(years=66, days=1) #UTC time
    #convert to Pacific time
    ca_tz = timezone('America/Los_Angeles')
    ca_date = utc_dt.astimezone(ca_tz)

    return ca_date.strftime(strformat)