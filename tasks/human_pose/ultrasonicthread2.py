#for buzzer
import Jetson.GPIO as GPIO
import serial
import threading
import time, sys

ultrasonic_gpio = [ [17,18,0], [27,23,0], [22,24,0] , [5,25,0] ]

GPIO.setmode(GPIO.BCM)

#this function as a thread will loop add the gpio array and then continpusly update the ultrasonic_gpio_array[i][2]
class UltrasonicSensorThread(threading.Thread):
    
    global ultrasonic_gpio

    def __init__(self, *args, **kwargs): 
        super(UltrasonicSensorThread, self).__init__(*args, **kwargs) 
        self._stopper = threading.Event() 
  
     #  (avoid confusion)
    def stopit(self):       
        self._stopper.set() # ! must not use _stop
        for i in range (len(ultrasonic_gpio)):
            GPIO.output(ultrasonic_gpio[i][0], False)
        GPIO.cleanup()

    
    def stopped(self): 
        
        return self._stopper.isSet() 
  
    def run(self): 

        GPIO.setmode(GPIO.BCM)
        
        new_reading = False
        counter = 0

        for i in range (len(ultrasonic_gpio)):

            print (ultrasonic_gpio[i][0])
            print (ultrasonic_gpio[i][1])
            
            GPIO.setup( ultrasonic_gpio[i][0], GPIO.OUT)
            GPIO.setup( ultrasonic_gpio[i][1], GPIO.IN)
            GPIO.output( ultrasonic_gpio[i][0], False)

            print ('Waiting a few seconds for the ',i,' sensor to settle')
            time.sleep(0.5)

        while True:

            for i in range (len(ultrasonic_gpio)):

                if self.stopped(): 
                    return                

                GPIO.output(ultrasonic_gpio[i][0], True)
                time.sleep(0.00001)
                GPIO.output(ultrasonic_gpio[i][0], False)                

                counter = 0
                while GPIO.input( ultrasonic_gpio[i][1] )==0:
                    counter += 1
                    if counter == 5000:
                        new_reading = True
                        break                    
                    pulse_start = time.time()

                if new_reading:
                    print("NEW Reading1")
                    continue
                
                counter = 0
                new_reading = False
                while GPIO.input( ultrasonic_gpio[i][1] )==1:
                    counter += 1
                    if counter == 5000:
                        new_reading = True
                        break            
                    pulse_end = time.time()

                if new_reading:
                    print("NEW Reading2")
                    continue


                pulse_duration = pulse_end - pulse_start
                distance = pulse_duration * 17165
                distance = round(distance, 1)
                ultrasonic_gpio[i][2] = distance
                print ('Distance ',i, ':',distance,'cm')

                
                #time.sleep(0.002)


GPIO.setmode(GPIO.BCM)

ultrasonic_thread = UltrasonicSensorThread()

try:

    GPIO.setmode(GPIO.BCM)
    if not ultrasonic_thread.isAlive():                    
        ultrasonic_thread.start()


except KeyboardInterrupt:
    print("Keyboard interrupt exception caught")

    try:
        ultrasonic_thread.stopit()
        ultrasonic_thread.join()
    except:
        pass

finally:   
    pass
    #GPIO.cleanup() 