from flask import Flask, render_template
import json
import os
import sys
import glob
import pickle
import re

from OutlierDetector import outlierDetector

app = Flask(__name__)
app.static_folder = 'static'


def ids():
    ids_per_cluster = matrix = [[0]*13 for i in range(13)]

    for (dirpath, dirnames, filenames) in os.walk(
            '/Users/margot.rutgers/mirror/mirror/SimilarityChecker/cluster_machines'):
        for filename in filenames:
            with app.open_resource(os.path.join(dirpath, filename)) as f:
                linenum, clusternum = get_numbers_from_filename(filename)
                ids = pickle.load(f)
                ids_per_cluster[int(linenum)][int(clusternum)] = ids
    return ids_per_cluster


def get_numbers_from_filename(filename):
    regex = re.compile(r'\d+')
    nums = regex.findall(filename)
    for num in nums:
        num = int(num)
    return nums


@app.route('/')
def hello_world():
    print("Loading machine passports")
    with app.open_resource('static/MachinePassports.json') as f:
        machinepassports = json.load(f)

    outliers = outlierDetector.detect_outliers()

    print("Loading timestamps")
    with app.open_resource('static/output.json') as f:
        timestamps = json.load(f)

    # An machine type is a letter, but sometimes indicated by a number...
    toLetter = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J'}

    return render_template('index.html', machinepassports=machinepassports,
                           outliers=outliers, toLetter=toLetter, timestamps=timestamps,
                           ids_per_cluster=ids())

if __name__ == '__main__':
    app.run(debug=True, port=5001)
