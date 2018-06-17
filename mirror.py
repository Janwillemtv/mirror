from flask import Flask, render_template
import json
import sys

from OutlierDetector import outlierDetector

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def hello_world():
    print("Loading machine passports")
    with app.open_resource('static/MachinePassports.json') as f:
        machinepassports = json.load(f)

    print("Detecting current outliers")
    outliers = outlierDetector.detect_outliers()

    return render_template('index.html', machinepassports=machinepassports, outliers=outliers)


if __name__ == '__main__':
    app.run(debug=True)
