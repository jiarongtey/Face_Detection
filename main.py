import cv2

# Load pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Load image -> Rezize -> Covert to grayscale
img = cv2.imread("mark.jpg")
resized_img = cv2.resize(img,(560,560))
grayscaled_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# Detect face
face_coordinate = trained_face_data.detectMultiScale(grayscaled_img)

# Square the faces
for (x,y,w,h) in face_coordinate:
    cv2.rectangle(resized_img,(x,y), (x+w, y+h), (0,255,0), 2)

# Show 
cv2.imshow("Simple Face Detection" , resized_img)
cv2.moveWindow("Simple Face Detection", 40,30)

cv2.waitKey()