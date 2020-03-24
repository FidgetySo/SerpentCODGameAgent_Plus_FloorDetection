from serpent.game_agent import GameAgent

from serpent.enums import InputControlTypes

from serpent.frame_grabber import FrameGrabber
from serpent.input_controller import KeyboardKey
from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent
import pytesseract

import numpy as np

from torch.autograd import Variable
import torch

from serpent.config import config

from serpent.logger import Loggers

import serpent.cv
import signal
import time
import random

from mss import mss

from PIL import Image
import os

import cv2
import skimage

import pyautogui
import keyboard
import pynput
import ctypes
from pynput.mouse import Button, Controller
mouse = Controller()
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
Wd, Hd = 1920, 1080
ACTIVATION_RANGE = 300
YOLO_DIRECTORY = "models"
CONFIDENCE = 0.36
THRESHOLD = 0.22
ACTIVATION_RANGE = 350
labelsPath = os.path.sep.join([YOLO_DIRECTORY, "coco-dataset.labels"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.weights"])
configPath = os.path.sep.join([YOLO_DIRECTORY, "yolov3-tiny.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
origbox = (int(Wd/2 - ACTIVATION_RANGE/2),
    int(Hd/2 - ACTIVATION_RANGE/2),
    int(Wd/2 + ACTIVATION_RANGE/2),
    int(Hd/2 + ACTIVATION_RANGE/2))
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def set_pos(x, y):
    current_x, current_y = pyautogui.position()
    new_x = current_x + x
    new_y = current_y + y
    x = 1 + int(new_x * 65536./Wd)
    y = 1 + int(new_x * 65536./Hd)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
def set_pos_aimbot(x, y):
    ScreenCenterX = 1920 / 2
    ScreenCenterY = 1080 / 2
    AimSpeed = 1.0
    if x > ScreenCenterX:
        target_x = -(ScreenCenterX - x)
        target_x = target_x / AimSpeed
    elif x < ScreenCenterX:
        target_x = x - ScreenCenterX;
        target_x = target_x / AimSpeed;
    if y > ScreenCenterY:
        target_y = -(ScreenCenterY - Y)
        target_y = target_y / AimSpeed
    elif y < ScreenCenterY:
        target_y = x - ScreenCenterY;
        target_y = target_y / AimSpeed;
    x = 1 + int(target_x * 65536./Wd)
    y = 1 + int(target_y * 65536./Hd)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
def find_floor():
    #Define im as frame
    y1= 0
    x1= 0
    y2= 1080
    x2= 1920
    with mss() as sct:
        monitor_var = sct.monitors[1]
        monitor = np.array(sct.grab(monitor_var))
    gray = cv2.cvtColor(monitor, cv2.COLOR_BGR2GRAY) 
    blur = cv2.medianBlur(gray, 25)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,27,6)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilate = cv2.dilate(close, kernel, iterations=2)
    roi = dilate[y1:y2, x1:x2]
    roi2 = monitor[y1:y2, x1:x2]
    cnts = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    areaArray = []
    for i, c in enumerate(contours):
        area1 = cv2.contourArea(c)
        areaArray.append(area1)
    sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)
    if len(sorteddata) == 0:
        pass
    else:
        largest = sorteddata[0][1]
        x1, y1, w1, h1 = cv2.boundingRect(largest)
        print("Area: X1: " + str(x1) + " Y1:" + str(y1) + " W:" + str(w1) + " H:" + str(h1))
        cv2.rectangle(roi2, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)
        x_c, y_c, w_c, h_c = cv2.boundingRect(largest)
        center_X_area = x_c +(w_c / 2)
        center_Y_area = y_c +(h_c / 2)
        current_x_area, current_y_area = pyautogui.position()
        diff_cc_x_area = current_x_area - center_X_area
        diff_cc_y_area = current_y_area - center_Y_area
        if (diff_cc_x_area >= 375 and diff_cc_x_area >= 150) or (diff_cc_x_area <= -375 and diff_cc_x_area <= -150):
            set_pos_aimbot(center_X_area, center_Y_area)
    minimum_area = 1000
    max_area = 9223372036854775
    for c in cnts:
        area = cv2.contourArea(c)
        if area > minimum_area and area < max_area:
            print(area)
            # Find centroid
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(roi2, (cX, cY), 20, (36, 255, 12), 2) 
            x,y,w,h = cv2.boundingRect(c)
            cv2.putText(roi2, 'Radius: {}'.format(w/2), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            cv2.drawContours(roi2,c,-1,(255,255,0),-1)
    cv2.imshow("Floor", roi2)
    cv2.waitKey(1)
class SerpentCODGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        self.frame_handler_pause_callbacks["PLAY"] = self.handle_play_pause

    def setup_play(self):
        self.max_epi_iter = 30000
        self.max_MC_iter = 100
        self.environment = self.game.environments["GAME"](
            game_api=self.game.api,
            input_controller=self.input_controller,
            episodes_per_startregions_track=100000000000
        )

        self.game_inputs = [
            {
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT", "COMBAT", "CURSOR", "FIRE"])
            }
        ]
        print(len(self.game_inputs[0]["inputs"]))
        self.agent = RainbowDQNAgent(
            "COD",
            game_inputs=self.game_inputs,
            callbacks=dict(
                after_observe=self.after_agent_observe,
                before_update=self.before_agent_update,
                after_update=self.after_agent_update
                ),
            rainbow_kwargs=dict(
                replay_memory_capacity=250000,
                observe_steps=100,
                batch_size=8,
                save_steps=100,
                model="datasets/rainbow_dqn_COD.pth"
                ),
            logger=Loggers.COMET_ML,
            logger_kwargs=dict(
                api_key="api_key",
                project_name="serpent-ai-cod",
                reward_func=self.reward
            )
            )
        self.analytics_client.track(event_key="COD", data={"name": "COD"})
        self.environment.new_episode(maximum_steps=350)  # 5 minutes
        self.overs = 0
        self.input_non_lethal = False
    def handle_play(self, game_frame, game_frame_pipeline):
        print("handling play")
        self.paused_at = None
        with mss() as sct:
            monitor_var = sct.monitors[1]
            self.monitor = sct.grab(monitor_var)
            valid_game_state = self.environment.update_startregions_state(self.monitor)
        if not valid_game_state:
            return None

        reward, over_boolean = self.reward(1.0)
        terminal = over_boolean
        
        self.agent.observe(reward=reward, terminal=terminal)

        if not terminal:
            game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
            agent_actions = self.agent.generate_actions(game_frame_buffer)
            print("Current Action: ")
            print(agent_actions)
            str_agent_actions = str(agent_actions)
            self.human()
            if "MOVE MOUSE X1" in str_agent_actions:
                set_pos(200, 0)
            if "MOVE MOUSE Y1" in str_agent_actions:
                set_pos(0, 200)
            if "MOVE MOUSE XY1" in str_agent_actions:
                set_pos(100, 100)
            if "MOVE MOUSE X2" in str_agent_actions:
                set_pos(-200, 0)
            if "MOVE MOUSE Y2" in str_agent_actions:
                set_pos(0, -200)
            if "MOVE MOUSE XY2" in str_agent_actions:
                set_pos(-100, -100)
            if "MOVE MOUSE XY3" in str_agent_actions:
                set_pos(-100, 100)
            if "MOVE MOUSE XY4" in str_agent_actions:
                set_pos(100, -100)
            if "LETHAL" in str_agent_actions:
                self.input_non_lethal = True
            if "SHOOT1" in str_agent_actions:
                mouse.click(Button.left, 1)
            if "SHOOT2" in str_agent_actions:
                mouse.press(Button.right)
                mouse.click(Button.left, 1)
                mouse.release(Button.right)
            if "SHOOT3" in str_agent_actions:
                mouse.click(Button.left, 2)
            if "SHOOT4" in str_agent_actions:
                mouse.press(Button.right)
                mouse.click(Button.left, 2)
                mouse.release(Button.right)
            self.environment.perform_input(agent_actions)
            find_floor()
        else:
            self.environment.clear_input()
            self.agent.reset()

            time.sleep(30)
            #To Do
            #Choose Loadout (Meduim Range)
            self.environment.end_episode()
            self.environment.new_episode(maximum_steps=350)
            print("New Episode")
    def handle_play_pause(self):
        self.input_controller.handle_keys([])
    def num_there(self, s):
        return any(i.isdigit() for i in s)

    def get_health(self, image):
        img = Image.frombytes('RGB', image.size, image.rgb)
        red_O = 0
        for red in img.getdata():
            if red == (117,54,34):
                red_O += 1
        return red_O
    def get_xp(self, image_xp):
        img = Image.frombytes('RGB', image_xp.size, image_xp.rgb)
        pixels = 0
        for pixel in img.getdata():
            if pixel == (255,194,21):
                pixels += 1
        return pixels
    def is_startregions_over(self, image):
        image = Image.frombytes("RGB", image.size, image.bgra, "raw", "BGRX")
        ocr_result = pytesseract.image_to_string(image, lang='eng')
        print("Text: ")
        print(ocr_result)
        if "KILLCAM" in ocr_result:
            return True
        else:
            return False
    def human(self):
        with mss() as sct:
            W, H = None, None
            frame = np.array(sct.grab(origbox))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


            if W is None or H is None:
                (H, W) = frame.shape[: 2]

            frame = cv2.UMat(frame)
            blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
                swapRB=False, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = 0
                    confidence = scores[classID]
                    if confidence > CONFIDENCE:
                        box = detection[0: 4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
            if len(idxs) > 0:
                bestMatch = confidences[np.argmax(confidences)]

                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw target dot on the frame
                    cv2.circle(frame, (int(x + w / 2), int(y + h / 5)), 5, (0, 0, 255), -1)

                    # draw a bounding box rectangle and label on the frame
                    # color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y),
                                    (x + w, y + h), (0, 0, 255), 2)

                    text = "TARGET {}%".format(int(confidences[i] * 100))
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if bestMatch == confidences[i]:
                        mouseX = origbox[0] + (x + w/1.5)
                        mouseY = origbox[1] + (y + h/5)
                        set_pos_aimbot(mouseX, mouseY)
                        mouse.click(Button.left, 2)
                        time.sleep(.05)
                        mouse.click(Button.left, 2)
                        time.sleep(.05)
    def reward(self, object_reward_func):
        with mss() as sct:
            image = sct.grab(sct.monitors[1])
            value = self.get_health(image)
            print("Health: ")
            print(value * -1)
            monitor = {"top": 452, "left": 1000, "width": 144, "height": 51, "mon": 1}
            image_xp = sct.grab(monitor)
            xp = self.get_xp(image_xp)
            monitor_custom_game = {"top": 47, "left": 50, "width": 230, "height": 66, "mon": 1}
            image_over = sct.grab(monitor_custom_game)
            over_check = self.is_startregions_over(image_over)
            reward = 0.0
            over = False
            if over_check:
                reward = -1.0
                self.overs += 1
                if self.overs >= 6:
                    print("Game Over")
                    over = True
                    self.overs = 0
                else:
                    over = False
            else:
                reward = 1.0
                if value > 1:
                    if value == 1:
                        reward += -0.25
                    elif (value >= 2 and value <= 4):
                        reward += -0.60
                    elif (value >= 5 and value < 20):
                        reward += -0.80
                    elif value == 0:
                        reward += 0.0
                if xp >= 7:
                    reward += 3.0
                    
            print("Reward: ")
            print(reward)
            return reward, over
    def after_agent_observe(self):
        self.environment.episode_step()

    def before_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(1)

    def after_agent_update(self):
        self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
        time.sleep(3)
