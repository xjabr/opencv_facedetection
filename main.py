import cv2
import sys
import time
import math

WIDTH = 640
HEIGHT = 480
FPS = 60
prev = 0

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, WIDTH)
video_capture.set(4, HEIGHT)

def draw_line_distance(frame, center_point) -> None:
    origin = (int(WIDTH/2), int(HEIGHT/2))
    cv2.line(frame, origin, center_point, (255, 255, 255), 2)

    # calc distance
    d = math.sqrt(((origin[0] - center_point[0]) ** 2) + ((origin[1] - center_point[1]) ** 2))

    print('Distance from origin is: %.2f' % d)
    cv2.putText(frame, "D: %.2f" % d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

def processing_img(frame, fps):
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # write line
    cv2.line(frame, (int(WIDTH/2), 0), (int(WIDTH/2), int(HEIGHT)), (0, 255, 0), 2)
    cv2.line(frame, (0, int(HEIGHT/2)), (WIDTH, int(HEIGHT/2)), (0, 255, 0), 2)

    # Draw a rectangle around the faces
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        color = (0, 255, 0)
        center_x = int(x+w/2)
        center_y = int(y+h/2)

        # conditions for movements
        if center_x < WIDTH/2 and center_y > HEIGHT/2:
            color = (0, 0, 255)
            print("Vai in alto a destra")
        if center_x < WIDTH/2 and center_y < HEIGHT/2:
            color = (255, 0, 0)
            print("Vai in basso a destra")
        if center_x > WIDTH/2 and center_y > HEIGHT/2:
            color = (0, 255, 255)
            print("Vai in alto a sinistra")
        if center_x > WIDTH/2 and center_y < HEIGHT/2:
            color = (255, 255, 0)
            print("Vai in basso a sinistra")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # draw origin point and center face point
        cv2.circle(frame, (int(x+w/2), int(y+h/2)), 3, color, 2)
        cv2.circle(frame, (int(WIDTH/2), int(HEIGHT/2)), 3, color, 2)
            
        # draw distance
        draw_line_distance(frame, (int(x+w/2), int(y+h/2)))

    cv2.imshow("Video", frame)    
    cv2.resizeWindow("Video", WIDTH, HEIGHT)
    
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    time_elapsed = time.time() - prev

    if time_elapsed > 1./FPS:
        prev = time.time()
        processing_img(frame, 1./FPS)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()