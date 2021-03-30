import cv2
import sys
import time
import math

WIDTH = 640
HEIGHT = 480
FPS = 120

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
video_capture.set(3, WIDTH)
video_capture.set(4, HEIGHT)

def draw_line_distance(_frame, center_point) -> None:
    origin = (int(WIDTH/2), int(HEIGHT/2))
    cv2.line(_frame, origin, center_point, (255, 255, 255), 2)

    # calc distance
    d = math.sqrt(((origin[0] - center_point[0]) ** 2) + ((origin[1] - center_point[1]) ** 2))

    # print('Distance from origin is: %.2f' % d)
    cv2.putText(_frame, "D: %.2f" % d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


def calc_distance_from_cam(_frame, w, h, data):
    d_cal = ((data[0] * data[1]) / data[2]) / 10
    d_new = ((w * data[1]) / data[2]) / 10
    d = d_cal - d_new + 50 # first distance from calibration - new distance + 50 (50 is the first distance for calibration).
    print('Distance from cam: %.2f' % d)


def processing_img(_frame, _gray, calibration_data):
    faces = faceCascade.detectMultiScale(
        _gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # write line
    cv2.line(_frame, (int(WIDTH/2), 0), (int(WIDTH/2), int(HEIGHT)), (0, 255, 0), 2)
    cv2.line(_frame, (0, int(HEIGHT/2)), (WIDTH, int(HEIGHT/2)), (0, 255, 0), 2)

    # Draw a rectangle around the faces
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        color = (0, 255, 0)
        center_x = int(x+w/2)
        center_y = int(y+h/2)

        # conditions for movements
        if center_x < WIDTH/2 and center_y > HEIGHT/2:
            color = (0, 0, 255)
            # print("Vai in alto a destra")
        if center_x < WIDTH/2 and center_y < HEIGHT/2:
            color = (255, 0, 0)
            # print("Vai in basso a destra")
        if center_x > WIDTH/2 and center_y > HEIGHT/2:
            color = (0, 255, 255)
            # print("Vai in alto a sinistra")
        if center_x > WIDTH/2 and center_y < HEIGHT/2:
            color = (255, 255, 0)
            # print("Vai in basso a sinistra")
        
        cv2.rectangle(_frame, (x, y), (x+w, y+h), color, 2)

        # draw origin point and center face point
        cv2.circle(_frame, (int(x+w/2), int(y+h/2)), 3, color, 2)
        cv2.circle(_frame, (int(WIDTH/2), int(HEIGHT/2)), 3, color, 2)
            
        # draw distance
        draw_line_distance(_frame, (int(x+w/2), int(y+h/2)))
        # calc distance of the object from the cam
        calc_distance_from_cam(_frame, w, h, calibration_data)

    cv2.imshow("Video", _frame)    
    cv2.resizeWindow("Video", WIDTH, HEIGHT)
    
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)



def calibration_system():
    # calibration
    one_face_founded = False
    data_cal = (0, 0, 0)

    print("Stand 50 cm from the camera and press enter to calibrate.")
    input()

    while not one_face_founded:
        ret, img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) > 0:
            one_face_founded = True
            data_cal = (faces[0][2], 50, 18)
    
    return data_cal



def main(calibration_data):
    prev = 0

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        time_elapsed = time.time() - prev

        if time_elapsed > 1./FPS:
            prev = time.time()
            processing_img(frame, gray, calibration_data)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    data = calibration_system()
    main(data)