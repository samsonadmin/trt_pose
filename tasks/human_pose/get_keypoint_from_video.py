import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
#from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
#import emoji 
import serial
import threading

#for luma led display
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT
from luma.core import legacy

#for buzzer
import Jetson.GPIO as GPIO

#for smoothing
#for functions for faster calculations
import numpy as np
#cuPy replacement for numpy
#import cupy as cp

#serial_port = serial.Serial(
#    port="/dev/ttyTHS1",
#    baudrate=115200,
#    bytesize=serial.EIGHTBITS,
#    parity=serial.PARITY_NONE,
#    stopbits=serial.STOPBITS_ONE,
#)
#

def led_matrix(display_text):
    if led_matrix_ok:
        with canvas(led_device) as draw:        
            if display_text == "w":
                legacy.text(draw, (0, 0), "\0", fill="white", font=up_arrow_bitmap_font)
            elif display_text == "a":
                #legacy.text(draw, (0, 0), "\0", fill="white", font=left_turn_arrow_bitmap_font)
                #purposely switched left / right because user is facing forward
                legacy.text(draw, (0, 0), "\0", fill="white", font=right_turn_arrow_bitmap_font)
            elif display_text == "d":
                #legacy.text(draw, (0, 0), "\0", fill="white", font=right_turn_arrow_bitmap_font)
                #purposely switched left / right because user is facing forward
                legacy.text(draw, (0, 0), "\0", fill="white", font=left_turn_arrow_bitmap_font)
            elif display_text == "s":
                legacy.text(draw, (0, 0), "\0", fill="white", font=stop_arrow_bitmap_font)
            elif display_text == "q":
                legacy.text(draw, (0, 0), "\0", fill="white", font=left_arrow_bitmap_font)
            elif display_text == "e":
                legacy.text(draw, (0, 0), "\0", fill="white", font=right_arrow_bitmap_font)
            else:
                text(draw, (1, 1), display_text, fill="white", font=proportional(LCD_FONT))
            #
#    for _ in range(2):
#        time.sleep(0.05)
#        led_device.hide()
#
#        time.sleep(0.05)
#        led_device.show()



try:
    serial_port = serial.Serial(
        port="/dev/ttyUSB0",
        #port="/dev/ttyTHS1",
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )
    serial_ok = True
    print("Serial port OK")

except:
    serial_ok = False
    print("Serial Port issue")


try:
    # create matrix device
    serial = spi(port=0, device=0, gpio=noop())
    led_device = max7219(serial, width=8, height=8, rotate=1, block_orientation=0)
    print("Created device")
    led_matrix_ok = True

except:
    led_matrix_ok = False



led_matrix("#")
print('Loading models...')

class LedCountDownThread(threading.Thread):
    def __init__(self, *args, **kwargs): 
        super(LedCountDownThread, self).__init__(*args, **kwargs) 
        self._stopper = threading.Event() 

     #  (avoid confusion)
    def stopit(self):       
        self._stopper.set() # ! must not use _stop
        led_matrix("")      
  
    def stopped(self): 
        return self._stopper.isSet() 
  
    def run(self):          
        for i in range (18,1,-1):
            if self.stopped(): 
                return            

            led_matrix(str(i))
            time.sleep(0.25)
            led_matrix('')
            time.sleep(0.25)
            led_matrix(str(i))
            time.sleep(0.25)
            led_matrix('')
            time.sleep(0.25)
    
led_countdown = LedCountDownThread()
led_countdown.start()

# Pin Definitions
output_pin = 12  # BOARD pin 12, BCM pin 18

last_serial_command_sent = ""
next_serial_command_to_send = ""
fall_detected_outer_rectangle_is_on = False
halt_non_stop_buzzer_thread = False

def buzzer_thread(beeptimes, sleeptime):

    GPIO.setmode(GPIO.BCM)
    #GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
    curr_value = GPIO.HIGH

    try:
        for x in range(beeptimes):            
            # Toggle the output every second
            #print("Outputting {} to pin {}".format(curr_value, output_pin))
            GPIO.output(output_pin, curr_value)
            curr_value ^= GPIO.HIGH
            time.sleep(sleeptime)
    finally:
        GPIO.cleanup()
        buzzer = threading.Thread(target=buzzer_thread, args=(2,0.04, ))


class NonStopBuzzerThread(threading.Thread):
    #https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/

    def __init__(self, *args, **kwargs): 
        super(NonStopBuzzerThread, self).__init__(*args, **kwargs) 
        self._stopper = threading.Event() 
  
     #  (avoid confusion)
    def stopit(self):       
        self._stopper.set() # ! must not use _stop
        GPIO.output(output_pin, GPIO.LOW)       
  
    def stopped(self): 
        return self._stopper.isSet() 
  
    def run(self): 
        GPIO.setmode(GPIO.BCM)
        #GPIO.setmode(GPIO.BOARD)
        # set pin as an output pin with optional initial state of HIGH
        GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)
        curr_value = GPIO.HIGH        

        while True:
            if self.stopped(): 
                return

            #print("Outputting {} to pin {}".format(curr_value, output_pin))
            GPIO.output(output_pin, curr_value)
            curr_value ^= GPIO.HIGH                     
            time.sleep(0.5) 


non_stop_buzzer = NonStopBuzzerThread()

buzzer = threading.Thread(target=buzzer_thread, args=(2,0.04, ))

#https://www.riyas.org/2013/12/online-led-matrix-font-generator-with.html
up_arrow_bitmap_font = [ [0x18,0x3c,0x7e,0xdb,0x99,0x99,0x18,0x18] ]
left_arrow_bitmap_font = [ [0x38,0x0c,0x06,0xff,0xff,0x06,0x0c,0x38] ]
left_turn_arrow_bitmap_font = [ [0x30,0x60,0xfc,0xfe,0x63,0x33,0x06,0x0c] ]
right_turn_arrow_bitmap_font = [ [0x0c,0x06,0x3f,0x7f,0xc6,0x6c,0x30,0x18] ]
right_arrow_bitmap_font = [ [0x18,0x0c,0x06,0xff,0xff,0x06,0x0c,0x18] ]
down_arrow_bitmap_font = [ [0x18,0x18,0x99,0x99,0xdb,0x7e,0x3c,0x18] ]
stop_arrow_bitmap_font = [ [0x3c,0x42,0x40,0x3c,0x02,0x42,0x42,0x3c] ]


parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
parser.add_argument("-v", "--video", required=True,	help="path to input video file")

args = parser.parse_args()

WINDOW_NAME = 'human_pose'

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = '/home/jetsonnano/trained-weights/trt_human_pose/resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = '/home/jetsonnano/trained-weights/trt_human_pose/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:    
    print('------ model = densenet--------')
    MODEL_WEIGHTS = '/home/jetsonnano/trained-weights/trt_human_pose/densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = '/home/jetsonnano/trained-weights/trt_human_pose/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    print("Optimized Model NOT found, loading from Model")
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    print("Optimized Model saved")


model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()


print("It took %5.1f"%(50.0 / (t1 - t0)),"s to load Model" )


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

#print("")

display_width = 1920
display_height = 1080


body_labels = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder', 
               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
              15:'lAnkle', 16:'rAnkle', 17:'neck'}


largest_valid_human_keypoints = []

'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''
def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)

            #edited following code might make it slower
            #print(body_labels[j], ' index:%d : success [%5.5f, %5.5f]'%(j, peak[1], peak[2]) )

        else:    
            peak = (j, None, None)
            kpoint.append(peak)

            #edited following code might make it slower
            #print(body_labels[j], ' index:%d : None %d'%(j, k) )
    return kpoint


#class GetKeypoints(object):
#    def __init__(self, topology):
#        self.topology = topology
#        self.body_labels = {0:'nose', 1: 'lEye', 2: 'rEye', 3:'lEar', 4:'rEar', 5:'lShoulder', 6:'rShoulder',
#               7:'lElbow', 8:'rElbow', 9:'lWrist', 10:'rWrist', 11:'lHip', 12:'rHip', 13:'lKnee', 14:'rKnee',
#              15:'lAnkle', 16:'rAnkle', 17:'neck'}
#        self.body_parts = sorted(self.body_labels.values())
#
#    def __call__(self, image, object_counts, objects, normalized_peaks):
#        topology = self.topology
#        height = image.shape[0]
#        width = image.shape[1]
#
#        K = topology.shape[0]
#        count = int(object_counts[0])
#        if count > 1:
#            count = 1
#        K = topology.shape[0]
#        
#        body_dict = {}
#        feature_vec = []
#        for i in range(count):
#            obj = objects[0][i]
#            C = obj.shape[0]
#            for j in range(C):
#                k = int(obj[j])
#                if k >= 0:
#                    peak = normalized_peaks[0][j][k]
#                    x = round(float(peak[1]) * width)
#                    y = round(float(peak[0]) * height)
#                    body_dict[self.body_labels[j]] = [x,y]
#        for part in self.body_parts:
#            feature_vec.append(body_dict.get(part, [0,0]))
#        feature_vec = [item for sublist in feature_vec for item in sublist]
#        return feature_vec
#

#class ListHumans(object):
#    def __init__(self, body_labels=body_labels):
#        self.body_labels = body_labels
#
#    def __call__(self, objects, normalized_peaks):
#
#        pose_list = []
#        for obj in objects[0]:
#            pose_dict = {}
#            C = obj.shape[0]
#            for j in range(C):
#                k = int(obj[j])
#                if k >= 0:
#                    peak = normalized_peaks[0][j][k]
#                    x = round(float(peak[1]) * WIDTH)
#                    y = round(float(peak[0]) * HEIGHT)
#                    #cv2.circle(image, (x, y), 3, color, 2)
#                    pose_dict[self.body_labels[j]] = (x,y)
#            pose_list.append(pose_dict)
#
#        return pose_list
#
#humans = ListHumans()
#



def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def execute(img, src, t):

    global last_serial_command_sent
    global next_serial_command_to_send
    global fall_detected_outer_rectangle_is_on
    global buzzer
    global non_stop_buzzer

    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()

    ##Action
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)

    # make dictionary from obj id to cmap

    #edited following code might make it slower
    #pose_list = humans(objects, peaks)    


     #explict counting of detect raised hands humnas
    #hand_raised_human = 0
    valid_raised_hand_human = False
    last_valid_human_nose_neck_distance_squared = 0
    text_to_display = []
    target_keypoint = []
    temp_target_keypoint = []
        
    for i in range(counts[0]):

        valid_raised_hand_human = False

        temp_text_to_display = []

        keypoints_x = []
        keypoints_y = []        

        #This is all the human key points
        keypoints = get_keypoint(objects, i, peaks)
        #print("LEN:",len(keypoints), "CNT:", counts[0])


        #i is the different poople detected

        #detecting if hand is raised, only follow if you are facing the wheelchair, and the camera can see your face and upper body
        
        #if lShoulder 5 or rShoulder 6 higher than nose 0 and can see both nose 0, lEye 1, rEye 2, lHip 11, rHip 12
        if keypoints[5][1] and keypoints[6][1] and keypoints[0][1] and keypoints[1][1] and keypoints[2][1] and keypoints[17][1] and( keypoints[9][1] or keypoints[10][1] ):


            #if camera can see both Wrist
            if keypoints[9][1] and keypoints[10][1]:
                

                if keypoints[10][1] < keypoints[9][1]:
                    
                    #if right hand higher than right shoulder 
                    if keypoints[10][1] < keypoints[6][1]:

                        ###text_to_display.append("Right Wrist is Higher")

                        x = round(keypoints[10][2] * WIDTH * X_compress)
                        y = round(keypoints[10][1] * HEIGHT * Y_compress)      

                        cv2.circle(src, (x, y), 20, (0, 176, 176), 4, cv2.LINE_AA)  
                        #hand_raised_human+=1
                        valid_raised_hand_human = True

                        #set wrist as target
                        #temp_target_keypoint = [keypoints[10][2], keypoints[10][1]]
                        #change to nose as target
                        temp_target_keypoint = [keypoints[0][2], keypoints[0][1]]

                      

                else:

                    #if left hand higher than left shoulder 
                    if keypoints[9][1] < keypoints[5][1]:

                        ###text_to_display.append("Left Wrist is Higher")

                        x = round(keypoints[9][2] * WIDTH * X_compress)
                        y = round(keypoints[9][1] * HEIGHT * Y_compress)  

                        cv2.circle(src, (x, y), 10, (0, 176, 176), 3, cv2.LINE_AA)  
                        #hand_raised_human+=1
                        valid_raised_hand_human = True
                        #set wrist as target
                        #temp_target_keypoint = [keypoints[9][2], keypoints[9][1]]
                        #change to nose as target
                        temp_target_keypoint = [keypoints[0][2], keypoints[0][1]]                        

            #see the left wrist
            elif keypoints[9][1]:
                #print( "Left Hand", keypoints[9][1], keypoints[5][1])

                #if left hand higher than left shoulder 
                if keypoints[9][1] < keypoints[5][1]:                
                    ###text_to_display.append("Left Wrist")
                    x = round(keypoints[9][2] * WIDTH * X_compress)
                    y = round(keypoints[9][1] * HEIGHT * Y_compress)

                    cv2.circle(src, (x, y), 10, (0, 176, 176), 3, cv2.LINE_AA)  
                    #hand_raised_human+=1
                    valid_raised_hand_human = True

                    #set wrist as target
                    #temp_target_keypoint = [keypoints[9][2], keypoints[9][1]]
                    #change to nose as target
                    temp_target_keypoint = [keypoints[0][2], keypoints[0][1]]                    

            #see the right wrist                              
            elif keypoints[10][1]:
                #print( "Right Hand", keypoints[10][1], keypoints[6][1])

                #if right hand higher than right shoulder 
                if keypoints[10][1] < keypoints[6][1]:                
                    ###text_to_display.append("Right Wrist")
                    x = round(keypoints[10][2] * WIDTH * X_compress)
                    y = round(keypoints[10][1] * HEIGHT * Y_compress)
                              
                    cv2.circle(src, (x, y), 10, (0, 176, 176), 3, cv2.LINE_AA)  
                    #hand_raised_human+=1
                    valid_raised_hand_human = True

                    #set wrist as target
                    #temp_target_keypoint = [keypoints[10][2], keypoints[10][1]] 
                    #change to nose as target
                    temp_target_keypoint = [keypoints[0][2], keypoints[0][1]]                    


            else:
                pass


            if valid_raised_hand_human:
                #using euclidean distance between the nose and neck, the larger distance, the bigger the human is
                this_nose_neck_distance_squared = pow( (keypoints[0][2]-keypoints[17][2]),2 ) + pow( (keypoints[0][1]-keypoints[17][1]),2 )

                if this_nose_neck_distance_squared > last_valid_human_nose_neck_distance_squared:
                    last_valid_human_nose_neck_distance_squared = this_nose_neck_distance_squared
                    largest_valid_human_keypoints = keypoints
                    target_keypoint = temp_target_keypoint


        #detection for fall down

        #if (keypoints[5][1] or keypoints[6][1]) and (keypoints[11][1] or keypoints[12][1]):
        #    print('fall down')




        if (keypoints[5][1] or keypoints[6][1]) and (keypoints[7][1] or keypoints[8][1]) and (keypoints[11][1] or keypoints[12][1]) and (keypoints[13][1] or keypoints[14][1]):

            for j in range(len(keypoints)):
                if keypoints[j][1]:
                    keypoints_x.append( keypoints[j][2] )
                    keypoints_y.append( keypoints[j][1] )

            if len(keypoints_x) > 7:
                standard_dev_x = np.std(keypoints_x)
                standard_dev_y = np.std(keypoints_y)
                ###text_to_display.append( "{} points with x: {:.4f} y: {:.4f}".format(len(keypoints_x), standard_dev_x,standard_dev_y) )

                ##Possible tuning of variable, the higher the variable, the less likely to fall
                ##The wider the lens, it is it require higher number,
                ##normal lensn 90degree can use 1.2
                fall_detection_percent = standard_dev_x / standard_dev_y / 1.6
                if fall_detection_percent > 1:
                    fall_detection_percent = 1

                if standard_dev_y / standard_dev_x > 2:
                    fall_detection_percent = 1
                else:
                    text_to_display.append("Possible fall down {:2.2f}%".format(fall_detection_percent*100))
                    

        '''
        rhip > lknee
        '''


        ##Below is for display purposes


        #color = (112,107,222)
        color = (0, 255, 136)
        #Loop all the keypoints in all humans and draw the dot
        for j in range(len(keypoints)):

            if keypoints[j][1]:

                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                cv2.circle(src, (x, y), 3, color, 2, cv2.LINE_AA)
                cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 45, 45), 2, cv2.LINE_AA)



        #draw nose to neck
        if keypoints[0][1] and keypoints[17][1]:
            x0 = round(keypoints[0][2] * WIDTH * X_compress)
            y0 = round(keypoints[0][1] * WIDTH * Y_compress)
            x1 = round(keypoints[17][2] * WIDTH * X_compress)
            y1 = round(keypoints[17][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw neck to rshoulder
        if keypoints[17][1] and keypoints[6][1]:
            x0 = round(keypoints[17][2] * WIDTH * X_compress)
            y0 = round(keypoints[17][1] * WIDTH * Y_compress)
            x1 = round(keypoints[6][2] * WIDTH * X_compress)
            y1 = round(keypoints[6][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw neck to lshoulder
        if keypoints[17][1] and keypoints[5][1]:
            x0 = round(keypoints[17][2] * WIDTH * X_compress)
            y0 = round(keypoints[17][1] * WIDTH * Y_compress)
            x1 = round(keypoints[5][2] * WIDTH * X_compress)
            y1 = round(keypoints[5][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw rshoulder to rElbow
        if keypoints[6][1] and keypoints[8][1]:
            x0 = round(keypoints[6][2] * WIDTH * X_compress)
            y0 = round(keypoints[6][1] * WIDTH * Y_compress)
            x1 = round(keypoints[8][2] * WIDTH * X_compress)
            y1 = round(keypoints[8][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)
                    
        #draw rElbow to rWrist
        if keypoints[8][1] and keypoints[10][1]:
            x0 = round(keypoints[10][2] * WIDTH * X_compress)
            y0 = round(keypoints[10][1] * WIDTH * Y_compress)
            x1 = round(keypoints[8][2] * WIDTH * X_compress)
            y1 = round(keypoints[8][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw lshoulder to lElbow
        if keypoints[5][1] and keypoints[7][1]:
            x0 = round(keypoints[5][2] * WIDTH * X_compress)
            y0 = round(keypoints[5][1] * WIDTH * Y_compress)
            x1 = round(keypoints[7][2] * WIDTH * X_compress)
            y1 = round(keypoints[7][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw lElbow to lWrist
        if keypoints[7][1] and keypoints[9][1]:
            x0 = round(keypoints[7][2] * WIDTH * X_compress)
            y0 = round(keypoints[7][1] * WIDTH * Y_compress)
            x1 = round(keypoints[9][2] * WIDTH * X_compress)
            y1 = round(keypoints[9][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw rShoulder to rHip
        if keypoints[6][1] and keypoints[12][1]:
            x0 = round(keypoints[6][2] * WIDTH * X_compress)
            y0 = round(keypoints[6][1] * WIDTH * Y_compress)
            x1 = round(keypoints[12][2] * WIDTH * X_compress)
            y1 = round(keypoints[12][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw lShoulder to lHip
        if keypoints[5][1] and keypoints[11][1]:
            x0 = round(keypoints[5][2] * WIDTH * X_compress)
            y0 = round(keypoints[5][1] * WIDTH * Y_compress)
            x1 = round(keypoints[11][2] * WIDTH * X_compress)
            y1 = round(keypoints[11][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        
        #draw rHip to rKnee
        if keypoints[12][1] and keypoints[14][1]:
            x0 = round(keypoints[12][2] * WIDTH * X_compress)
            y0 = round(keypoints[12][1] * WIDTH * Y_compress)
            x1 = round(keypoints[14][2] * WIDTH * X_compress)
            y1 = round(keypoints[14][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw lHip to lKnee
        if keypoints[11][1] and keypoints[13][1]:
            x0 = round(keypoints[11][2] * WIDTH * X_compress)
            y0 = round(keypoints[11][1] * WIDTH * Y_compress)
            x1 = round(keypoints[13][2] * WIDTH * X_compress)
            y1 = round(keypoints[13][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw rHip to lHip
        if keypoints[12][1] and keypoints[11][1]:
            x0 = round(keypoints[12][2] * WIDTH * X_compress)
            y0 = round(keypoints[12][1] * WIDTH * Y_compress)
            x1 = round(keypoints[11][2] * WIDTH * X_compress)
            y1 = round(keypoints[12][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)        


#        #draw neck to center hip
#        if keypoints[17][1] and keypoints[11][1] and keypoints[12][1]:
#            x0 = round(keypoints[17][2] * WIDTH * X_compress)
#            y0 = round(keypoints[17][1] * WIDTH * Y_compress)
#            x1 = round( (keypoints[11][2]+keypoints[12][2])/2 * WIDTH * X_compress)
#            y1 = round( (keypoints[11][1]+keypoints[12][1])/2 * WIDTH * Y_compress)
#            cv2.line(src, (x0, y0), (x1, y1), color, 2)
#
#        #draw center hip to right knee
#        if keypoints[14][1] and keypoints[11][1] and keypoints[12][1]:
#            x0 = round(keypoints[14][2] * WIDTH * X_compress)
#            y0 = round(keypoints[14][1] * WIDTH * Y_compress)
#            x1 = round( (keypoints[11][2]+keypoints[12][2])/2 * WIDTH * X_compress)
#            y1 = round( (keypoints[11][1]+keypoints[12][1])/2 * WIDTH * Y_compress)
#            cv2.line(src, (x0, y0), (x1, y1), color, 2)        
#
#
#        #draw center hip to left knee
#        if keypoints[13][1] and keypoints[11][1] and keypoints[12][1]:
#            x0 = round(keypoints[13][2] * WIDTH * X_compress)
#            y0 = round(keypoints[13][1] * WIDTH * Y_compress)
#            x1 = round( (keypoints[11][2]+keypoints[12][2])/2 * WIDTH * X_compress)
#            y1 = round( (keypoints[11][1]+keypoints[12][1])/2 * WIDTH * Y_compress)
#            cv2.line(src, (x0, y0), (x1, y1), color, 2)      

        #draw rKnee to rAnkle
        if keypoints[14][1] and keypoints[16][1]:
            x0 = round(keypoints[14][2] * WIDTH * X_compress)
            y0 = round(keypoints[14][1] * WIDTH * Y_compress)
            x1 = round(keypoints[16][2] * WIDTH * X_compress)
            y1 = round(keypoints[16][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)

        #draw lKnee to lAnkle
        if keypoints[13][1] and keypoints[15][1]:
            x0 = round(keypoints[13][2] * WIDTH * X_compress)
            y0 = round(keypoints[13][1] * WIDTH * Y_compress)
            x1 = round(keypoints[15][2] * WIDTH * X_compress)
            y1 = round(keypoints[15][1] * WIDTH * Y_compress)
            cv2.line(src, (x0, y0), (x1, y1), color, 2)            

    #output the selected human and action that should be taken
    if last_valid_human_nose_neck_distance_squared > 0:
        #largest_valid_human_keypoints
        keypoints = largest_valid_human_keypoints       


        #Variables to Tune
        if target_keypoint[0] < 0.4:            
            text_to_display.append("ACTION: Turn Left 'a'") 
            next_serial_command_to_send = "a"
        elif target_keypoint[0] > 0.5:
            text_to_display.append("ACTION: Turn Right 'd'")
            next_serial_command_to_send = "d"               
        else:
            
            #if nose to neck is too small, it mean it is close to person, stop
            ##Tune this number to stop going forward
            if (  pow( (keypoints[1][2]-keypoints[2][2]),2) + pow( (keypoints[1][1]-keypoints[2][1]),2) < 0.00182 ):

                text_to_display.append("eyes: %5.5f"%( pow( (keypoints[1][2]-keypoints[2][2]),2) + pow( (keypoints[1][1]-keypoints[2][1]),2) ))
                text_to_display.append("ACTION: Go Straight Forward 'w'")
                next_serial_command_to_send = "w"
            else:
                #pass
                text_to_display.append("ACTION: Stop, coming to close")
                next_serial_command_to_send = "s"

                if non_stop_buzzer.isAlive():
                    try:
                        non_stop_buzzer.stopit()
                        non_stop_buzzer.join()
                        non_stop_buzzer = NonStopBuzzerThread() ##create a new thread
                    except:
                        pass                
  

        text_to_display.append( "target-keypoint: %5.5f, %5.5f"%( target_keypoint[0], target_keypoint[1] ) )
              

        #draw rectangle, with nose as center point
        x1 = round(  (keypoints[0][2] - (keypoints[17][1]-keypoints[0][1])/2*0.8 )  * WIDTH * X_compress)
        y1 = round(  (keypoints[0][1] - (keypoints[17][1]-keypoints[0][1])*0.8 )  * HEIGHT * Y_compress)
        x2 = round(  (keypoints[0][2] + (keypoints[17][1]-keypoints[0][1])/2*0.8 )  * WIDTH * X_compress)
        y2 = round(  (keypoints[0][1] + (keypoints[17][1]-keypoints[0][1])*0.8 )  * HEIGHT * Y_compress)
        cv2.rectangle(src, (x1,y1), (x2,y2), (153,255,51), 2)

        ##Instead of the circle on the nose
        #x = round(target_keypoint[0] * WIDTH * X_compress)
        #y = round(target_keypoint[1] * HEIGHT * Y_compress)
        
        #cv2.circle(src, (x, y), 12, (153,255,51), 2, cv2.LINE_AA)

    else:
        #if suddently no people raise hand, send "s" to stop
        next_serial_command_to_send = "s"
        #also stop buzzer
        #print ("Will STOP halt_non_stop_buzzer_thread")
        if non_stop_buzzer.isAlive():
            try:
                non_stop_buzzer.stopit()
                non_stop_buzzer.join()
                non_stop_buzzer = NonStopBuzzerThread() ##create a new thread
            except:
                pass


    #deciding which serial command to send
    if last_serial_command_sent != next_serial_command_to_send:

        if (last_serial_command_sent == "a" or last_serial_command_sent == "d" ) and ( next_serial_command_to_send == "w" or next_serial_command_to_send == "s" ):
            #send "s" to stop if it was turning
            if serial_ok:
                serial_port.write ("s\r\n".encode() )
            text_to_display.append("s")
            led_matrix("s")


        if serial_ok:
            serial_port.write( (next_serial_command_to_send + "\r\n").encode() )
        led_matrix(next_serial_command_to_send)
        text_to_display.append(next_serial_command_to_send)

        if next_serial_command_to_send == "a" or next_serial_command_to_send == "d" or  next_serial_command_to_send == "w":
            if not non_stop_buzzer.isAlive():                
                non_stop_buzzer.start()

        last_serial_command_sent = next_serial_command_to_send

    else:
        pass
        #send a dummy "z" char
        #if serial_ok:
        #    serial_port.write ("z\r\n".encode() )        

    #render the text with open CV
    for j in range(len(text_to_display)):
        cv2.putText(src , text_to_display[j], (12, 52 + 70 * j),  cv2.FONT_HERSHEY_DUPLEX, 1.7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(src , text_to_display[j], (10, 50 + 70 * j),  cv2.FONT_HERSHEY_DUPLEX, 1.7, (234,181,234), 2, cv2.LINE_AA)


        if text_to_display[j].find('Possible fall') > -1 :
            if fall_detected_outer_rectangle_is_on:
                cv2.rectangle( src, (0,0), (1920,1080), (0, 0, 255 ), 20 )
                fall_detected_outer_rectangle_is_on = False
            else:                
                fall_detected_outer_rectangle_is_on = True
                
            text_to_display[j] = bcolors.WARNING + text_to_display[j] + bcolors.ENDC
            
            if not non_stop_buzzer.isAlive():                    
                non_stop_buzzer.start()

    if len(text_to_display) > 0: 
        print( ', '.join(text_to_display) )
        text_to_display = []

    #print("FPS:%3.2f "%(fps))
    #draw_objects(img, counts, objects, peaks)

    cv2.putText(src , "FPS: %3.2f" % (fps), (11, 21),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(src , "FPS: %3.2f" % (fps), (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (219, 255, 51), 1)
	#skip writing
    #out_video.write(src)
    cv2.imshow(WINDOW_NAME, src)

    #edited following code might make it slower
    #return img, pose_list

#send a dummy "z" char
#if serial_ok:
#    serial_port.write ("z\r\n".encode() )

cap = cv2.VideoCapture(args.video, cv2.CAP_GSTREAMER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)

ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

#skip writing
#out_video = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (display_width, display_height))

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) 
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow(WINDOW_NAME, 640, 360)

count = 0

X_compress = display_width / WIDTH * 1.0
Y_compress = display_height / HEIGHT * 1.0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
#draw_objects = DrawObjects(topology)

try:

    try:
        led_countdown.stopit()
        led_countdown.join()
    except:
        pass

    while cap.isOpened():

        t = time.time()
        ret_val, dst = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break

        img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        execute(img, dst, t)
        count += 1
        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"): # ESC 
            break

except KeyboardInterrupt:
    print("Keyboard interrupt exception caught")

    try:
        non_stop_buzzer.stopit()
        non_stop_buzzer.join()
    except:
        pass

    if serial_ok:
        serial_port.close()

    GPIO.setmode(GPIO.BCM)
    #GPIO.setmode(GPIO.BOARD)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.LOW)        
    GPIO.output(output_pin, GPIO.LOW)


finally:
    cv2.destroyAllWindows()
    #out_video.release()
    cap.release()

    
