from flask import Flask, render_template
import json
import sys
import glob

from OutlierDetector import outlierDetector

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def hello_world():
    print("Loading machine passports")
    with app.open_resource('static/MachinePassports.json') as f:
        machinepassports = json.load(f)

    outliers = outlierDetector.detect_outliers()

    with app.open_resource('static/timestamps.json') as f:
        timestamps = json.load(f)

    # An machine type is a letter, but sometimes indicated by a number...
    toLetter = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I'}

    return render_template('index.html', machinepassports=machinepassports,
                            outliers=outliers, toLetter=toLetter, timestamps=timestamps)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
