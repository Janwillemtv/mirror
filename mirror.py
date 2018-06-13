from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome to  m i r r o r'


if __name__ == '__main__':
    app.run()
