#Smile detector. Takes a picture if all the faces in the frame are smiling

#Libraries
import cv2
import numpy as np
import time

#Emotion detector
class EmotionDetector:
	#Constructor
	def __init__(self, camera_number, number_of_pictures_to_take = 1):
		self.__haar_smile_path = 'haarcascade_smile.xml'
		self.__haar_face_path = 'haarcascade_frontalface_default.xml' 
		self.__smile_cascade = cv2.CascadeClassifier(self.__haar_smile_path)
		self.__face_cascade = cv2.CascadeClassifier(self.__haar_face_path)
		self.__capture = cv2.VideoCapture(camera_number)
		self.__face_rectangle_color = (0,0,255)
		self.__smile_rectangle_color = (255,0,0)
		self.__window = 'Smile Detector'
		self.__everybody_smiling = False
		self.__number_of_faces = 0
		self.__number_of_pictures_to_take = number_of_pictures_to_take
		
	#Get a new frame
	def __getNewFrame(self):
		cv2.waitKey(1)
		_, newFrame = self.__capture.read()
		return newFrame
	
	#Convert frame from BGR to gray scale
	def __convertToGray(self, frame):
		return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	#Detect faces using Cascade classifier
	def __getFaces(self, gray_frame):
		return self.__face_cascade.detectMultiScale(gray_frame, 1.1, 3)
	
	#Sets the value of everybody_smiles to True
	def __set_everybody_smiles(self, everybody_smiles):
		self.__everybody_smiles = everybody_smiles
	
	#Get everybody smiles
	def __get_everybody_smiles(self):
		return self.__everybody_smiles
	
	#Get number of pictures to take
	def __get_number_of_pictures_to_take(self):
		return self.__number_of_pictures_to_take
	
	#Decrease the number of pictures to be taken
	def __decrease_number_of_pictures_to_take_by_one(self):
		self.__number_of_pictures_to_take -= 1
	
	#Take a picture of the frame
	def __take_picture(self, frame):
		#Take frame as an png file
		if self.__get_number_of_pictures_to_take() > 0:
			cv2.imwrite('frame_{}.png'.format(str(time.time()).replace('.','_')),frame)
			self.__decrease_number_of_pictures_to_take_by_one()
	
	#Detect smiles using Cascade classifier
	def __getSmiles(self, roi_face):
		smiles =  self.__smile_cascade.detectMultiScale(roi_face, 1.7, 22)
		#If every face is smiling, change everybody_smiles to True
		if len(smiles) == self.__getNumberOfFaces():
			self.__set_everybody_smiles(True)
		else:
				self.__set_everybody_smiles(False)
		return smiles
		
	#Draw rectangles on face features
	def __draw_rectangles(self, frame, faces):
		self.__draw_face_features_rectangles(frame, faces)
	
	#Set number of faces
	def __setNumberOfFaces(self, number_of_faces):
		self.__number_of_faces = number_of_faces
	
	#Get number of faces
	def __getNumberOfFaces(self):
		return self.__number_of_faces

	#Draw the face rectangles and call another function to draw the rectangles on the eyes and the smile
	def __draw_face_features_rectangles(self, frame, faces):
		#Update number of faces
		self.__setNumberOfFaces(len(faces))
		
		for (x,y,w,h) in faces:
			#Draw rectangles around face
			cv2.rectangle(frame, (x,y), (x+w, y+h), self.__face_rectangle_color, 2)
			#Get region of interest
			roi_face = frame[y:y+h, x:x+w]
			roi_color_face = frame[y:y+h, x:x+w]
			#Draw rectangles around smile
			self.__draw_smile_rectangles(roi_face, roi_color_face)
	
	#Draw the smile rectangles
	def __draw_smile_rectangles(self, roi_face, roi_color_face):
		smiles = self.__getSmiles(roi_face)
		for (x, y, w, h) in smiles:
			cv2.rectangle(roi_color_face, (x, y), (x+w, y+h), self.__smile_rectangle_color, 2)
	
	#Display the frame on a window
	def __show_frame(self, frame):
		cv2.imshow(self.__window, frame)
	
	#Check is the camera is open
	def __captureIsOpened(self):
		return self.__capture.isOpened()
	
	#Release the camera
	def __closeCapture(self):
		self.__capture.release()
	
	#Call a function to release the camera
	def closeCapture(self):
		self.__closeCapture()
	
	#Delay camera to make sure it is ready
	def __delay_camera(self):
		for i in range(2):
			_ = self.__getNewFrame()
			_ = self.__getNewFrame()
			time.sleep(1)
		
	#Detect face features on the frame, draw rectangles and display the frame
	def detect(self):
		
			#Start camera and wait a few seconds to avoid empty frames
			self.__delay_camera()
			#Get a new frame
			frame = self.__getNewFrame()
			#Save a copy if it has to store the frame as an image file
			#clean_frame = np.copy(frame)
			clean_frame = frame
			#Convert frame to gray scale
			gray_frame = self.__convertToGray(frame)
			#Detect faces
			faces = self.__getFaces(gray_frame)
			#Draw rectangles on the frame
			self.__draw_rectangles(frame, faces)
			#If everybody is smiling, take a picture
			if self.__get_everybody_smiles() == True:
				self.__take_picture(clean_frame)
				
			#Display frame
			self.__show_frame(frame)
			
			

#Main
def main():
	#Variables
	keep_looping = True
	camera_number = 0
	number_of_pictures = 1 #Number of pictures to be taken
	#Create an emotion detector object
	emotion_detector = EmotionDetector(camera_number, number_of_pictures)

	#Show instructions to stop the loop
	print("Press 'Q' to exit the program")
	while(keep_looping):
		#Detect face features
		emotion_detector.detect()
		
		#If 'Q' is pressed, stop detecting faces
		if cv2.waitKey(1) & 0xFF == ord('q'):
			emotion_detector.closeCapture()
			keep_looping = False
			
#End of program

main()
