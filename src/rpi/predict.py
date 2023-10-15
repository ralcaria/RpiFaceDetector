import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
from dataclasses import dataclass

@dataclass
class Pins: #define the rpi gpio pins --> R:11,G:12,B:13
    redPin: int = 11
    greenPin: int = 12
    bluePin: int = 13 

    def _outputLow(self,pin):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    def _outputHigh(self,pin):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)

    def redOn(self,Pins):
        self._outputLow(Pins.redPin)
        self._outputHigh(Pins.greenPin)
        self._outputHigh(Pins.bluePin)

    def greenOn(self,Pins):
        self._outputHigh(Pins.redPin)
        self._outputLow(Pins.greenPin)
        self._outputHigh(Pins.bluePin)

    def blueOn(self,Pins):
        self._outputHigh(Pins.redPin)
        self._outputHigh(Pins.greenPin)
        self._outputLow(Pins.bluePin)

def main():
    """Load the model --> Setup the camera --> Make inferences on each frame
    """
    #Setup the model
    TARGETS = {0: 'NotReigan', 1: 'Reigan'}
    model = tflite.Interpreter(model_path='face_detect_model.tflite')
    model.allocate_tensors() #Need to allocate tensors before making inferences

    #Setup video
    print("Starting camera, begin predictions...")
    CAMERA_WIDTH = 480
    CAMERA_HEIGHT = 320
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    pins = Pins() #RPI LED pins

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
          break

        grayimg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255 #model was trained on gray images, pixels scaled by 255
        
        #Get the input tensor (feeding frame to model)
        input_details = model.get_input_details()
        input_shape = tuple(input_details[0]['shape'])
        input_tensor = np.reshape(grayimg, input_shape).astype(np.float32)
        model.set_tensor(input_details[0]['index'],input_tensor)

        #Make the prediction
        model.invoke()
        output_details = model.get_output_details()
        raw_pred = model.get_tensor(output_details[0]['index'])
        pred = np.where(raw_pred >= 0.5, 1, 0).ravel()[0]
        if TARGETS[pred] == 'NotReigan':
          pins.redOn(pins)
        else:
          pins.greenOn(pins)

        #Calculate FPS and display on screen
        fps = int(1/(time.time() - start_time))
        cv2.putText(frame, f'fps: {str(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Output', frame)

        #Break out of the while loop
        if cv2.waitKey(1) == 27: #ESC key
            print("Exiting program and turning off LED...")
            break
    
    #Cleanup
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()