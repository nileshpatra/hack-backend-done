import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense , Flatten , Convolution2D , MaxPooling2D , Conv2D , Dropout 
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator



from keras.preprocessing.image import ImageDataGenerator


import pickle
filename = 'ker_model.sav'
loaded_model = pickle.load(open(filename ,'rb'))

fl = open('pred.html' , 'r')
path = fl.read()
fl.close()
print(path)

train_datagen = ImageDataGenerator(
rescale = 1./255,
shear_range = 0.2,
zoom_range=0.2 ,
horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


xtest = train_datagen.flow_from_directory(path
                                          , target_size = (64,64),
                                          batch_size = 32 ,
                                          class_mode = 'binary'
                                        )
_ , (image , label) = next(enumerate(xtest))

result = loaded_model.predict(image)
print(result)
var = ''
remedy = ''
if result.argmax() == 0:
    var = 'healthy'
    remedy = ''
elif result.argmax() == 1:
    var = 'rot'
    remedy = 'The most common fungicide spray for apple fruit rots is captan. Captan is a contact fungicide that stays on the surface of the apple and stops energy production in the fungus'
elif result.argmax() == 2:
    var = 'rust'
    remedy = 'Choose resistant cultivars when available.Rake up and dispose of fallen leaves and other debris from under trees.'
else:
    var = 'scab'
    remedy = 'Apply a foliar spray of zinc sulfate and urea in the fall. This spray forces the tree to drop its leaves quickly and causes rapid decomposition of the leaf litter.'

print(var)
print(remedy)
if(var=='healthy'):

	s = """
	<!DOCTYPE html>
	<html>
	<div id="header">
	    <div id="header-content">
	        <div class="topnav">
	            <!-- replace the hyperlinks that are to be linked with the '#' in href --> <a font-family: "Times New Roman", Times, serif href="#" class="pull-left">Dr.Agro
	            </a> 
	            <a padding:200px class="pull-right" href="#">Common Diseases</a>
	            <a class="pull-right" href="#">Contact Us</a>
	        </div>
	        
	    </div>
	</div>
	<head>
		<title>Dr.Agro</title>

		<style type="text/css">
		@import url(https://fonts.googleapis.com/css?family=Open+Sans:400italic,400,300,600);
		#header {
	    background-color: transparent;
	    -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=0)";
	    filter: alpha(opacity=100);
	    -moz-opacity: 0.8;
	    -khtml-opacity: 0.8;
	    opacity: 0.8;
	    color: #000;
	    position: fixed;
	    top: 0;
	    left: 0;
	    width: 100%;
	    height: 60px;
	    padding: 0;
	    margin: 0;
	    z-index: 500;
	}

	#header #header-content {
	    margin: 5px;
	}

	.topnav a {
	  float: left;
	  color: green;
	  text-align: center;
	  padding: 0% 0% 0% 11%;
	  text-decoration: none;
	  font-weight: bold;
	  font-size: 20px;
	}

	.c{
		 position: absolute;
	  top: 32%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.containr {
	  position: relative;
	  text-align: center;
	  color: white;
	}

	.centered {
	  position: absolute;
	  top: 20%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 60px;
	  color: green;
	}

	.centere {
	  position: absolute;
	  top: 25%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.topnavhead{
	    size: 100dp;
	}


	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea, #contact button[type="submit"] { font:400 12px/16px "Open Sans", Helvetica, Arial, sans-serif; }

	#contact {
		background:#F9F9F9;
		padding:25px;
		margin:50px 0;
	}

	#contact h3 {
		color: #F96;
		display: block;
		font-size: 30px;
		font-weight: 400;
	}

	#contact h4 {
		margin:5px 0 15px;
		display:block;
		font-size:13px;
	}

	fieldset {
		border: medium none !important;
		margin: 0 0 10px;
		min-width: 100%;
		padding: 0;
		width: 100%;
	}

	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea {
		width:100%;
		border:1px solid #CCC;
		background:#FFF;
		margin:0 0 5px;
		padding:10px;
	}

	#contact input[type="text"]:hover, #contact input[type="email"]:hover, #contact input[type="tel"]:hover, #contact input[type="url"]:hover, #contact textarea:hover {
		-webkit-transition:border-color 0.3s ease-in-out;
		-moz-transition:border-color 0.3s ease-in-out;
		transition:border-color 0.3s ease-in-out;
		border:1px solid #AAA;
	}

	#contact textarea {
		height:100px;
		max-width:100%;
	  resize:none;
	}

	#contact button[type="submit"] {
		cursor:pointer;
		width:100%;
		border:none;
		background:#0CF;
		color:#FFF;
		margin:0 0 5px;
		padding:10px;
		font-size:15px;
	}

	#contact button[type="submit"]:hover {
		background:#09C;
		-webkit-transition:background 0.3s ease-in-out;
		-moz-transition:background 0.3s ease-in-out;
		transition:background-color 0.3s ease-in-out;
	}

	#contact button[type="submit"]:active { box-shadow:inset 0 1px 3px rgba(0, 0, 0, 0.5); }

	#contact input:focus, #contact textarea:focus {
		outline:0;
		border:1px solid #999;
	}
	::-webkit-input-placeholder {
	 color:#888;
	}
	:-moz-placeholder {
	 color:#888;
	}
	::-moz-placeholder {
	 color:#888;
	}
	:-ms-input-placeholder {
	 color:#888;
	}
	containers {
		max-width:400px;
		width:100%;
		margin:0 auto;
		position:relative;
	}



			
		</style>
	</head>
	<body>
		<div class="containr">
			<img src="https://images.pexels.com/photos/6535/field-agriculture-farm-cereals.jpg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" width="100%">
			<!-- <div class="centered">Welcome to Dr.Agro!</div> -->
			<div class="centered">all good and set:)</div>
			<div class="centere"></div>
			<br><div class = "c"> State of crop : """ + var +"""</div><!-- <form action="{{ url_for('handle_data') }}" method="post">
	    Project file path: <input type="text" name="projectFilePath"><br>
	    <input type="submit" value="Submit">
	</form> -->
		<div>
			No issues , all good :)))
		</div>



		

	</form>	
		</div>
		
	<div id="divId"></div>
	<div class="containers">  
	  <form id="contact" action="" method="post">
	    <h3>Contact Us</h3>
	    <h4>Contact us today, and get reply with in 24 hours!</h4>
	    <fieldset>
	      <input placeholder="Your name" type="text" tabindex="1">
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Email Address" type="email" tabindex="2" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Phone Number" type="tel" tabindex="3" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Web Site starts with http://" type="url" tabindex="4" required>
	    </fieldset>
	    <fieldset>
	      <textarea placeholder="Type your Message Here...." tabindex="5" required></textarea>
	    </fieldset>
	    <fieldset>
	      <button name="submit" type="submit" id="contact-submit" data-submit="...Sending">Submit</button>
	    </fieldset>
	  </form>
	 
	  
	</div>
	<button onclick="scrollWin()">Scroll to top</button><br><br>
	<script>
	function scrollWin() {
	  window.scrollTo(500, 0);
	}
	</script>
	<div> """ + var + """</div>
	</body>
	</html>"""
	with open('P4Output.html', 'w') as f:
	    f.write(s)

	import os
	import shutil

	os.rename("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html",
	 "/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('ossss YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')

	#shutil.move("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html", 
		#"/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('shutil  YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')
	# file = open('pred.txt' , 'r')
	# print(file.read())

elif(var=='rot'):

	s = """
	<!DOCTYPE html>
	<html>
	<div id="header">
	    <div id="header-content">
	        <div class="topnav">
	            <!-- replace the hyperlinks that are to be linked with the '#' in href --> <a font-family: "Times New Roman", Times, serif href="#" class="pull-left">Dr.Agro
	            </a> 
	            <a padding:200px class="pull-right" href="#">Common Diseases</a>
	            <a class="pull-right" href="#">Contact Us</a>
	        </div>
	        
	    </div>
	</div>
	<head>
		<title>Dr.Agro</title>

		<style type="text/css">
		@import url(https://fonts.googleapis.com/css?family=Open+Sans:400italic,400,300,600);
		#header {
	    background-color: transparent;
	    -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=0)";
	    filter: alpha(opacity=100);
	    -moz-opacity: 0.8;
	    -khtml-opacity: 0.8;
	    opacity: 0.8;
	    color: #000;
	    position: fixed;
	    top: 0;
	    left: 0;
	    width: 100%;
	    height: 60px;
	    padding: 0;
	    margin: 0;
	    z-index: 500;
	}

	#header #header-content {
	    margin: 5px;
	}

	.topnav a {
	  float: left;
	  color: green;
	  text-align: center;
	  padding: 0% 0% 0% 11%;
	  text-decoration: none;
	  font-weight: bold;
	  font-size: 20px;
	}

	.c{
		 position: absolute;
	  top: 32%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.containr {
	  position: relative;
	  text-align: center;
	  color: white;
	}

	.centered {
	  position: absolute;
	  top: 30%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 60px;
	  color: green;
	}

	.centere {
	  position: absolute;
	  top: 40%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.topnavhead{
	    size: 100dp;
	}


	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea, #contact button[type="submit"] { font:400 12px/16px "Open Sans", Helvetica, Arial, sans-serif; }

	#contact {
		background:#F9F9F9;
		padding:25px;
		margin:50px 0;
	}

	#contact h3 {
		color: #F96;
		display: block;
		font-size: 30px;
		font-weight: 400;
	}

	#contact h4 {
		margin:5px 0 15px;
		display:block;
		font-size:13px;
	}

	fieldset {
		border: medium none !important;
		margin: 0 0 10px;
		min-width: 100%;
		padding: 0;
		width: 100%;
	}

	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea {
		width:100%;
		border:1px solid #CCC;
		background:#FFF;
		margin:0 0 5px;
		padding:10px;
	}

	#contact input[type="text"]:hover, #contact input[type="email"]:hover, #contact input[type="tel"]:hover, #contact input[type="url"]:hover, #contact textarea:hover {
		-webkit-transition:border-color 0.3s ease-in-out;
		-moz-transition:border-color 0.3s ease-in-out;
		transition:border-color 0.3s ease-in-out;
		border:1px solid #AAA;
	}

	#contact textarea {
		height:100px;
		max-width:100%;
	  resize:none;
	}

	#contact button[type="submit"] {
		cursor:pointer;
		width:100%;
		border:none;
		background:#0CF;
		color:#FFF;
		margin:0 0 5px;
		padding:10px;
		font-size:15px;
	}

	#contact button[type="submit"]:hover {
		background:#09C;
		-webkit-transition:background 0.3s ease-in-out;
		-moz-transition:background 0.3s ease-in-out;
		transition:background-color 0.3s ease-in-out;
	}

	#contact button[type="submit"]:active { box-shadow:inset 0 1px 3px rgba(0, 0, 0, 0.5); }

	#contact input:focus, #contact textarea:focus {
		outline:0;
		border:1px solid #999;
	}
	::-webkit-input-placeholder {
	 color:#888;
	}
	:-moz-placeholder {
	 color:#888;
	}
	::-moz-placeholder {
	 color:#888;
	}
	:-ms-input-placeholder {
	 color:#888;
	}
	containers {
		max-width:400px;
		width:100%;
		margin:0 auto;
		position:relative;
	}



			
		</style>
	</head>
	<body>
		<div class="containr">
			<img src="https://images.pexels.com/photos/6535/field-agriculture-farm-cereals.jpg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" width="100%">
			<!-- <div class="centered">Welcome to Dr.Agro!</div> -->
			<div class="centered">Welcome to Dr.Agro!</div>
			<div class="centere">The most common fungicide spray for apple fruit rots is captan. Captan is a contact fungicide that stays on the surface of the apple and stops energy production in the fungus
			</div></div>
			<br><div class = "c"> State of crop : """ + var +"""</div><!-- <form action="{{ url_for('handle_data') }}" method="post">
	    Project file path: <input type="text" name="projectFilePath"><br>
	    <input type="submit" value="Submit">
	</form> -->
		<div>
			The most common fungicide spray for apple fruit rots is captan. Captan is a contact fungicide that stays on the surface of the apple and stops energy production in the fungus
		</div>



		

	</form>	
		</div>
		
	<div id="divId"></div>
	<div class="containers">  
	  <form id="contact" action="" method="post">
	    <h3>Contact Us</h3>
	    <h4>Contact us today, and get reply with in 24 hours!</h4>
	    <fieldset>
	      <input placeholder="Your name" type="text" tabindex="1">
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Email Address" type="email" tabindex="2" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Phone Number" type="tel" tabindex="3" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Web Site starts with http://" type="url" tabindex="4" required>
	    </fieldset>
	    <fieldset>
	      <textarea placeholder="Type your Message Here...." tabindex="5" required></textarea>
	    </fieldset>
	    <fieldset>
	      <button name="submit" type="submit" id="contact-submit" data-submit="...Sending">Submit</button>
	    </fieldset>
	  </form>
	 
	  
	</div>
	<button onclick="scrollWin()">Scroll to top</button><br><br>
	<script>
	function scrollWin() {
	  window.scrollTo(500, 0);
	}
	</script>
	<div> """ + var + """</div>
	</body>
	</html>"""
	with open('P4Output.html', 'w') as f:
	    f.write(s)

	import os
	import shutil

	os.rename("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html",
	 "/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('ossss YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')

	#shutil.move("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html", 
		#"/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('shutil  YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')
	# file = open('pred.txt' , 'r')
	# print(file.read())

elif(var=='rust'):

	s = """
	<!DOCTYPE html>
	<html>
	<div id="header">
	    <div id="header-content">
	        <div class="topnav">
	            <!-- replace the hyperlinks that are to be linked with the '#' in href --> <a font-family: "Times New Roman", Times, serif href="#" class="pull-left">Dr.Agro
	            </a> 
	            <a padding:200px class="pull-right" href="#">Common Diseases</a>
	            <a class="pull-right" href="#">Contact Us</a>
	        </div>
	        
	    </div>
	</div>
	<head>
		<title>Dr.Agro</title>

		<style type="text/css">
		@import url(https://fonts.googleapis.com/css?family=Open+Sans:400italic,400,300,600);
		#header {
	    background-color: transparent;
	    -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=0)";
	    filter: alpha(opacity=100);
	    -moz-opacity: 0.8;
	    -khtml-opacity: 0.8;
	    opacity: 0.8;
	    color: #000;
	    position: fixed;
	    top: 0;
	    left: 0;
	    width: 100%;
	    height: 60px;
	    padding: 0;
	    margin: 0;
	    z-index: 500;
	}

	#header #header-content {
	    margin: 5px;
	}

	.topnav a {
	  float: left;
	  color: green;
	  text-align: center;
	  padding: 0% 0% 0% 11%;
	  text-decoration: none;
	  font-weight: bold;
	  font-size: 20px;
	}

	.c{
		 position: absolute;
	  top: 32%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.containr {
	  position: relative;
	  text-align: center;
	  color: white;
	}

	.centered {
	  position: absolute;
	  top: 20%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 60px;
	  color: green;
	}

	.centere {
	  position: absolute;
	  top: 40%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.topnavhead{
	    size: 100dp;
	}


	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea, #contact button[type="submit"] { font:400 12px/16px "Open Sans", Helvetica, Arial, sans-serif; }

	#contact {
		background:#F9F9F9;
		padding:25px;
		margin:50px 0;
	}

	#contact h3 {
		color: #F96;
		display: block;
		font-size: 30px;
		font-weight: 400;
	}

	#contact h4 {
		margin:5px 0 15px;
		display:block;
		font-size:13px;
	}

	fieldset {
		border: medium none !important;
		margin: 0 0 10px;
		min-width: 100%;
		padding: 0;
		width: 100%;
	}

	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea {
		width:100%;
		border:1px solid #CCC;
		background:#FFF;
		margin:0 0 5px;
		padding:10px;
	}

	#contact input[type="text"]:hover, #contact input[type="email"]:hover, #contact input[type="tel"]:hover, #contact input[type="url"]:hover, #contact textarea:hover {
		-webkit-transition:border-color 0.3s ease-in-out;
		-moz-transition:border-color 0.3s ease-in-out;
		transition:border-color 0.3s ease-in-out;
		border:1px solid #AAA;
	}

	#contact textarea {
		height:100px;
		max-width:100%;
	  resize:none;
	}

	#contact button[type="submit"] {
		cursor:pointer;
		width:100%;
		border:none;
		background:#0CF;
		color:#FFF;
		margin:0 0 5px;
		padding:10px;
		font-size:15px;
	}

	#contact button[type="submit"]:hover {
		background:#09C;
		-webkit-transition:background 0.3s ease-in-out;
		-moz-transition:background 0.3s ease-in-out;
		transition:background-color 0.3s ease-in-out;
	}

	#contact button[type="submit"]:active { box-shadow:inset 0 1px 3px rgba(0, 0, 0, 0.5); }

	#contact input:focus, #contact textarea:focus {
		outline:0;
		border:1px solid #999;
	}
	::-webkit-input-placeholder {
	 color:#888;
	}
	:-moz-placeholder {
	 color:#888;
	}
	::-moz-placeholder {
	 color:#888;
	}
	:-ms-input-placeholder {
	 color:#888;
	}
	containers {
		max-width:400px;
		width:100%;
		margin:0 auto;
		position:relative;
	}



			
		</style>
	</head>
	<body>
		<div class="containr">
			<img src="https://images.pexels.com/photos/6535/field-agriculture-farm-cereals.jpg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" width="100%">
			<!-- <div class="centered">Welcome to Dr.Agro!</div> -->
			<div class="centered">Welcome to Dr.Agro!</div>
			<div class="centere">Choose resistant cultivars when available.Rake up and dispose of fallen leaves and other debris from under trees.
			</div>
			<br><div class = "c"> State of crop : """ + var +"""</div><!-- <form action="{{ url_for('handle_data') }}" method="post">
	    Project file path: <input type="text" name="projectFilePath"><br>
	    <input type="submit" value="Submit">
	</form> -->	

	</form>	
		</div>
	<div class = "c">
			</div>
		
	<div id="divId"></div>
	<div class="containers">  
	  <form id="contact" action="" method="post">
	    <h3>Contact Us</h3>
	    <h4>Contact us today, and get reply with in 24 hours!</h4>
	    <fieldset>
	      <input placeholder="Your name" type="text" tabindex="1">
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Email Address" type="email" tabindex="2" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Phone Number" type="tel" tabindex="3" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Web Site starts with http://" type="url" tabindex="4" required>
	    </fieldset>
	    <fieldset>
	      <textarea placeholder="Type your Message Here...." tabindex="5" required></textarea>
	    </fieldset>
	    <fieldset>
	      <button name="submit" type="submit" id="contact-submit" data-submit="...Sending">Submit</button>
	    </fieldset>
	  </form>
	 
	  
	</div>
	<button onclick="scrollWin()">Scroll to top</button><br><br>
	<script>
	function scrollWin() {
	  window.scrollTo(500, 0);
	}
	</script>
	<div> """ + var + """</div>
	</body>
	</html>"""
	with open('P4Output.html', 'w') as f:
	    f.write(s)

	import os
	import shutil

	os.rename("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html",
	 "/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('ossss YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')

	#shutil.move("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html", 
		#"/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('shutil  YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')
	# file = open('pred.txt' , 'r')
	# print(file.read())

else:

	s = """
	<!DOCTYPE html>
	<html>
	<div id="header">
	    <div id="header-content">
	        <div class="topnav">
	            <!-- replace the hyperlinks that are to be linked with the '#' in href --> <a font-family: "Times New Roman", Times, serif href="#" class="pull-left">Dr.Agro
	            </a> 
	            <a padding:200px class="pull-right" href="#">Common Diseases</a>
	            <a class="pull-right" href="#">Contact Us</a>
	        </div>
	        
	    </div>
	</div>
	<head>
		<title>Dr.Agro</title>

		<style type="text/css">
		@import url(https://fonts.googleapis.com/css?family=Open+Sans:400italic,400,300,600);
		#header {
	    background-color: transparent;
	    -ms-filter: "progid:DXImageTransform.Microsoft.Alpha(Opacity=0)";
	    filter: alpha(opacity=100);
	    -moz-opacity: 0.8;
	    -khtml-opacity: 0.8;
	    opacity: 0.8;
	    color: #000;
	    position: fixed;
	    top: 0;
	    left: 0;
	    width: 100%;
	    height: 60px;
	    padding: 0;
	    margin: 0;
	    z-index: 500;
	}

	#header #header-content {
	    margin: 5px;
	}

	.topnav a {
	  float: left;
	  color: green;
	  text-align: center;
	  padding: 0% 0% 0% 11%;
	  text-decoration: none;
	  font-weight: bold;
	  font-size: 20px;
	}

	.c{
		 position: absolute;
	  top: 45%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.containr {
	  position: relative;
	  text-align: center;
	  color: white;
	}

	.centered {
	  position: absolute;
	  top: 20%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 60px;
	  color: green;
	}

	.centere {
	  position: absolute;
	  top: 32%;
	  left: 50%;
	  transform: translate(-50%, -50%);
	  font-size: 30px;
	  color: green;
	}

	.topnavhead{
	    size: 100dp;
	}


	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea, #contact button[type="submit"] { font:400 12px/16px "Open Sans", Helvetica, Arial, sans-serif; }

	#contact {
		background:#F9F9F9;
		padding:25px;
		margin:50px 0;
	}

	#contact h3 {
		color: #F96;
		display: block;
		font-size: 30px;
		font-weight: 400;
	}

	#contact h4 {
		margin:5px 0 15px;
		display:block;
		font-size:13px;
	}

	fieldset {
		border: medium none !important;
		margin: 0 0 10px;
		min-width: 100%;
		padding: 0;
		width: 100%;
	}

	#contact input[type="text"], #contact input[type="email"], #contact input[type="tel"], #contact input[type="url"], #contact textarea {
		width:100%;
		border:1px solid #CCC;
		background:#FFF;
		margin:0 0 5px;
		padding:10px;
	}

	#contact input[type="text"]:hover, #contact input[type="email"]:hover, #contact input[type="tel"]:hover, #contact input[type="url"]:hover, #contact textarea:hover {
		-webkit-transition:border-color 0.3s ease-in-out;
		-moz-transition:border-color 0.3s ease-in-out;
		transition:border-color 0.3s ease-in-out;
		border:1px solid #AAA;
	}

	#contact textarea {
		height:100px;
		max-width:100%;
	  resize:none;
	}

	#contact button[type="submit"] {
		cursor:pointer;
		width:100%;
		border:none;
		background:#0CF;
		color:#FFF;
		margin:0 0 5px;
		padding:10px;
		font-size:15px;
	}

	#contact button[type="submit"]:hover {
		background:#09C;
		-webkit-transition:background 0.3s ease-in-out;
		-moz-transition:background 0.3s ease-in-out;
		transition:background-color 0.3s ease-in-out;
	}

	#contact button[type="submit"]:active { box-shadow:inset 0 1px 3px rgba(0, 0, 0, 0.5); }

	#contact input:focus, #contact textarea:focus {
		outline:0;
		border:1px solid #999;
	}
	::-webkit-input-placeholder {
	 color:#888;
	}
	:-moz-placeholder {
	 color:#888;
	}
	::-moz-placeholder {
	 color:#888;
	}
	:-ms-input-placeholder {
	 color:#888;
	}
	containers {
		max-width:400px;
		width:100%;
		margin:0 auto;
		position:relative;
	}



			
		</style>
	</head>
	<body>
		<div class="containr">
			<img src="https://images.pexels.com/photos/6535/field-agriculture-farm-cereals.jpg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" width="100%">
			<!-- <div class="centered">Welcome to Dr.Agro!</div> -->
			<div class="centered">Welcome to Dr.Agro!</div>
			<div class="centere">Apply a foliar spray of zinc sulfate and urea in the fall. This spray forces the tree to drop its leaves quickly and causes rapid decomposition of the leaf litter.</div>
			<br><div class = "c"> State of crop : """ + var +"""</div><!-- <form action="{{ url_for('handle_data') }}" method="post">
	    Project file path: <input type="text" name="projectFilePath"><br>
	    <input type="submit" value="Submit">
	</form> -->
		<div>

		</div>	

	</form>	
		</div>
		
	<div id="divId"></div>
	<div class="containers">  
	  <form id="contact" action="" method="post">
	    <h3>Contact Us</h3>
	    <h4>Contact us today, and get reply with in 24 hours!</h4>
	    <fieldset>
	      <input placeholder="Your name" type="text" tabindex="1">
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Email Address" type="email" tabindex="2" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Phone Number" type="tel" tabindex="3" required>
	    </fieldset>
	    <fieldset>
	      <input placeholder="Your Web Site starts with http://" type="url" tabindex="4" required>
	    </fieldset>
	    <fieldset>
	      <textarea placeholder="Type your Message Here...." tabindex="5" required></textarea>
	    </fieldset>
	    <fieldset>
	      <button name="submit" type="submit" id="contact-submit" data-submit="...Sending">Submit</button>
	    </fieldset>
	  </form>
	 
	  
	</div>
	<button onclick="scrollWin()">Scroll to top</button><br><br>
	<script>
	function scrollWin() {
	  window.scrollTo(500, 0);
	}
	</script>
	<div> """ + var + """</div>
	</body>
	</html>"""
	with open('P4Output.html', 'w') as f:
	    f.write(s)

	import os
	import shutil

	os.rename("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html",
	 "/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('ossss YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')

	#shutil.move("/home/nilesh/Desktop/MY FILES/hackathon_final_files/P4Output.html", 
		#"/home/nilesh/Desktop/MY FILES/hackathon_final_files/templates/P4Output.html")

	print('shutil  YOOOOOOOOOOOOOOOOOOOOLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOOO')
	# file = open('pred.txt' , 'r')
	# print(file.read())

