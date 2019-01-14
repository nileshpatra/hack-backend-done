from flask import Flask,render_template,request,redirect,url_for 
import import_ipynb
import os

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('home.html')

@app.route('/model')
def output():
	return file.read();

@app.route('/handle_data',methods = ['POST'])
def handle_data():
	if request.method == 'POST':
		projectpath = request.form['projectFilePath']
		#os.mkdir('hello')
		print('PROOOJJJEEECTTTT PATH' , projectpath)
		file = open('pred.html','w+')
		file.write(projectpath)
		file = open('pred.html' , 'r')
		print(file.read())
		# import
		import model_run as clf
		return render_template('P4Output.html')
	else:
		return render_template('home.html')

if __name__ == '__main__':
	app.run(debug=True)
