import numpy as np
import random
from os import listdir
from os.path import isfile, join

random.seed(10000)
number_of_datapoints = 10 #set this for output



anomalities = "../OutlierDetector/outliers/"
normal_digits = "../OutlierDetector/goodDigits/"

anom_files = [f for f in listdir(anomalities) if isfile(join(anomalities, f))]
print(anom_files)

normal_files = [f for f in listdir(normal_digits) if isfile(join(anomalities, f))]
print(normal_files)

anom = {}  # dict for storing anoms keys are "1" etc
normal = {}

for a in anom_files:
    anom[a.rstrip(".npy")] = np.load(join(anomalities + a))
for a in normal_files:
    normal[a.rstrip(".npy")] = np.load(join(normal_digits + a))


def generate_random_scenario(machine, sequence):
    result = []
    for m in sequence:
        if m == 1:
            result.append(generate_random_anomality(machine))
        if m == 0:
            result.append(normal[machine][0])
    return result


def generate_random_anomality(machine):
    anoms = anom[machine]
    return anoms[random.randint(0, len(anoms) - 1)]


def generate_random_normality(machine):
    anoms = anom[machine]
    return normal[random.randint(0, len(anoms) - 1)]


for i in range(0,10):
    for y in range(1, 6):
        sequence = np.random.choice([0, 1], size=(number_of_datapoints,), p=[9. / 10, 1. / 10])
        print(str(i*5+y))
        s = generate_random_scenario(str(i), sequence)
        #print(s)
        np.savez(join("./sequences/"+str(i*5+y)),s)


#TODO make cherrypicked sequence for demo
#sequence = np.random.choice([0, 1], size=(number_of_datapoints,), p=[9. / 10, 1. / 10])
#s = generate_random_scenario(str(i), sequence)
#np.save(join("./sequences/"+str(i*5+y)),s)