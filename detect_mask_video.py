# modules which are to import
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy 
import argparse as ap
import imutils as im
import time
import cv2
import os

def detectandpredictmask(frame, face, mask):
	#the dimensions of the frame and then construct a blob
	
	(height, width) = frame.shape[:2]
	blobs = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	face.setInput(blobs)
	detection= face.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	facess = []
	locss = []
	predss = []

	# loop over the detections
	for i in range(0, detection.shape[1]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detection[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.9:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detection[0, 0, i, 3:7] * numpy.array([width, height, width, height])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(width - 1, endX), min(height - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				facess.append(face)
				locss.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(facess) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		facess = numpy.array(facess, dtype="float32")
		predss = mask.predict(facess, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locss, predss)



# load our serialized face detector model from disk
print("[information]loading_face_detector_model")
prototxt = r"face_detector\deployeprototxt"
weights = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
face = cv2.dnn.readNet(prototxt, weights)

# load the face mask detector model from disk
print("[information]loading_face_mask_detector_model")
maskNet = load_model("mask_detector.model")

# initialize the video stream and allow the camera sensor to warm up
print("[information] starting_the_video_stream")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = im.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locss, predss) = detectandpredictmask(frame, face, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (bx, pre) in zip(locss, predss):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = bx
		(mask, withoutMask) = pre

		# determine the class label and color we'll use to draw
		# the bounding box and text
		labels = "Mask" if mask > withoutMask else "wear the Mask"
		colors = (0, 255, 0) if labels == "Mask" else (0, 0, 255)
			
		

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, labels, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), colors, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	flag = cv2.waitKey(1)&0xFF

	# if the `j` key was pressed, break from the loop
	if flag == ord("j"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
