from flask import render_template
from app import app


@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Miguel'}
    return render_template('wizard-book-room.html')
