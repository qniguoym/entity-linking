import time
from datetime import datetime,timedelta
time_now = time.time()
time.sleep(2)
b = time.time()
print((b-time_now)/(60*60))
date = timedelta(b-time_now)
print(date)