## code hiervan gejat
##https://stackoverflow.com/questions/26740227/create-random-time-stamp-list-in-python
## state 0 = ok 1= maybe error 2= error

import numpy as np
import random
from os import listdir
from os.path import isfile, join
import json

from random import randrange
import datetime
offset= 6000


def random_date(start,l,offset):
   state = 0

   current = start
   result = []
   while l >= 0:
      if state >1:
         state = 0
      else:
         state += 1

      if state ==1:
         delta = datetime.timedelta(minutes=randrange(0,offset))
         current = current + delta
      else:
         delta = datetime.timedelta(minutes=randrange(0, 120)) ##make offset shorter on orange maybe error thingy becaues fixed
         current = current + delta

      l-=1
      result.append([current.strftime("%d/%m/%y %H:%M"),state])



   return result



startDate = datetime.datetime(2018, 3, 1,00,00)

output = []
for m in range(0,51):
   data = random_date(startDate,1000,offset)
   output.append(data)

with open(join('./output/output.json'), 'w') as outfile:
   json.dump(output, outfile)