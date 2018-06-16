from flask import Flask, render_template
import json

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def hello_world():
    with app.open_resource('static/MachinePassports.json') as f:
        machinepassports = json.load(f)

    return render_template('index.html', machinepassports=machinepassports)

if __name__ == '__main__':
    app.run(debug=True)
