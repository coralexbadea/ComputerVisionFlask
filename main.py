from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from detect1 import Detect
app = Flask(__name__)

detect = Detect(np.array([[0,0,0,0]]))
#camera = cv2.VideoCapture("http://192.168.1.2:8080/video") 
#camera = cv2.VideoCapture(0) 
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

#def gen_frames():  # generate frame by frame from camera
	#while True:
		# Capture frame-by-frame
		#success, frame = camera.read()  # read the camera frame
		
		#image=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
		#gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		#face_rects=face_cascade.detectMultiScale(gray,1.3,5)
		#for (x,y,w,h) in face_rects:
			#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
			#break
			
		#run()
		
		# if not success:
		# 	break
		# else:
		# 	ret, buffer = cv2.imencode('.jpg', frame)
		# 	frame = buffer.tobytes()
		# 	yield (b'--frame\r\n'
		# 		   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
	#Video streaming route. Put this in the src attribute of an img tag
	
	return Response(detect.run(), mimetype='multipart/x-mixed-replace; boundary=frame')
	

@app.route('/date')
def get_date():
	# text = request.args.get('jsdata')


	# r = requests.get('http://suggestqueries.google.com/complete/search?output=toolbar&hl=ru&q={}&gl=in'.format(text))

	# soup = BeautifulSoup(r.content, 'lxml')

	# suggestions = soup.find_all('suggestion')

	# for suggestion in suggestions:
	#     suggestions_list.append(suggestion.attrs['data'])

	#print(suggestions_list)
	average = detect.average
	date = f'{average[0]}{average[1]} - {average[2]}{average[3]}'
	

	return render_template('date.html', date=date)



@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
