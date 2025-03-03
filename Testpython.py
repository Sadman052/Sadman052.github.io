# calculate-width-of-a-river-channel-from-a-missing-survey

#program to calculate width of a river channel from a missing survey
#data where, B and C points are in the same bank side, A point is 
#in the other side and < BAC = 30 , AC = 40 meter. Use python math module. (hints: AB = AC x cos < BAC )

import math

def calculation(AC, BAC):
    AB = AC * math.cos(BAC)*math.pi/180
    return AB

output = calculation(AC =  40, BAC = 30)
output = round(output,2)
print("The width of the river in meter = ", output)
