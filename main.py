#!/usr/bin/env python3
# -*- coding: utf-8 -*-


getFaceAfter = 5  #剛剛抓到的FACE可能晃動或使用者還未準備好，因此改為使用連續抓取後的第幾張FACE
waitSeconds = 30  #若持續多久沒有動靜就進入預設的等待畫面
LCD_size_w = 240
LCD_size_h = 320
LCD_Rotate = 180
speakNameInEnglish = 1  #若辨識出英文人名，要用英文唸出而不用中文嗎？

#Libraries required
from picamera import PiCamera
from time import sleep
import os
import cv2
import numpy as np
import json
import time

#Import Google TTS library
from gtts import gTTS

#for IBM Watson API
from os.path import join, dirname
from os import environ
from watson_developer_cloud import VisualRecognitionV3
visual_recognition = VisualRecognitionV3('2016-05-20', api_key='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

#PICamera setting
camera = PiCamera()
camera.resolution = (320, 240)
camera.rotation = 90
camera.video_stabilization = False
camera.hflip = True
camera.vflip = False
camera.brightness = 60
#camera.ISO = 800

#Import LCD 9341 libraries
import Adafruit_ILI9341 as TFT
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
import Adafruit_GPIO.SPI as SPI
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
DC = 18
RST = 23
SPI_PORT = 0
SPI_DEVICE = 0

#TFT LCD configuration and initial the LCD
font = ImageFont.truetype('fonts/WCL-07.ttf', 30)
disp = TFT.ILI9341(DC, rst=RST, spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE, max_speed_hz=64000000))
disp.begin()  # Initialize display.

lastFaceDectedTime = time.clock()
defaultDisplay = 0
faceGetAccumulated = 0  #與getFaceAfter搭配，目前是第幾次抓到。

screenRatio = camera.resolution[0] / 320.00

#---------------------------------------------------------------------

def displayDefaultImg():
    global defaultDisplay
    image = Image.open("face.png")
    image = image.rotate(LCD_Rotate).resize((LCD_size_w, LCD_size_h))
    defaultDisplay = 1
    disp.display(image)

def draw_rotated_text(image, text, position, angle, font, fill=(255,255,255)):
    # Get rendered font width and height.
    draw = ImageDraw.Draw(image)
    width, height = draw.textsize(text, font=font)
    # Create a new image with transparent background to store the text.
    textimage = Image.new('RGBA', (width, height), (0,0,0,0))
    # Render the text.
    textdraw = ImageDraw.Draw(textimage)
    textdraw.text((0,0), text, font=font)
    # Rotate the text image.
    rotated = textimage.rotate(angle, expand=1)
    # Paste the text into the image, using it as a mask for transparency.
    image.paste(rotated, position, rotated)

def checkFace(imgfilePath, ynWatson = 0, useTTS = 0):
    global lastFaceDectedTime, defaultDisplay, getFaceAfter, faceGetAccumulated

    #if detectAge==1:
    displayImg = Image.open(imgfilePath)
    displayImg = displayImg.rotate(LCD_Rotate).resize((LCD_size_w, LCD_size_h))
    print ('(defaultDisplay=%s , %s)' % (defaultDisplay, time.clock()-lastFaceDectedTime))

    if time.clock()-lastFaceDectedTime < waitSeconds: 
        defaultDisplay = 0
        disp.display(displayImg)
    else:
        if defaultDisplay==0: displayDefaultImg()

    img = cv2.imread(imgfilePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
    #eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_small.xml')
    eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    #eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60)
    )

    iFace = 0
    xFace = 0
    yFace = 0
    if(len(faces)>0):
        lastFaceDectedTime = time.clock()
        for (x,y,w,h) in faces:
            print('Image size w=%s, h=%s' % (img.shape[1], img.shape[0]))
            print('FACE: X=%s ,Y=%s, W=%s, H=%s  ---> X=%s, Y=%s' % (x,y,w,h,x+(w/2),y+(h/2)))
            if iFace == 0: 
                xFace = x+(w/2)
                yFace = y+(h/2)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,246),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            iFace += 1

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                print('EYE: X=%s ,Y=%s, W=%s, H=%s' % (ex,ey,ew,eh))
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(251,34,12),2)

            mouth = mouth_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(60, 60))
            for (mx,my,mw,mh) in mouth:
                print('Mouth: X=%s ,Y=%s, W=%s, H=%s' % (mx,my,mw,mh))
                cv2.rectangle(roi_color,(mx+int(mw/5),my+int(mh/2+(mh/5))),(mx+(mw-int(mw/5)),my+int(mh/2+mh/4)),(4,162,252),2)
                #cv2.rectangle(roi_color,(mx,my),(mx+mw,my+int(mh/2)),(4,162,252),2)

            nose = nose_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(60, 60))
            for (nx,ny,nw,nh) in nose:
                print('Nose: X=%s ,Y=%s, W=%s, H=%s' % (nx,ny,nw,nh))
                cv2.rectangle(roi_color,(nx+int((nw/2)-(nw/5)),ny-int(nh/3)),(nx+int((nw/2)+(nw/4)),ny+int(nh*2/3)),(21,221,30),2)
            #    cv2.rectangle(roi_color,(nx,ny),(nx+int(nw/2),ny+nh),(21,221,30),2)


            cv2.imwrite("cv2.jpg", img)
            defaultDisplay = 0
            displayImg = Image.open("cv2.jpg")
            displayImg = displayImg.rotate(LCD_Rotate).resize((LCD_size_w, LCD_size_h))
            disp.display(displayImg)

            faceGetAccumulated += 1
            print('faceGetAccumulated=%s' % (faceGetAccumulated))

            if (faceGetAccumulated>=getFaceAfter and ynWatson==1 and xFace<(320*screenRatio) and xFace>(90*screenRatio) and yFace<(180*screenRatio) and yFace>(80*screenRatio)):  #Use Watson to analysis if the first face is at center (160<x<170)
                faceGetAccumulated = 0
                os.system('omxplayer --no-osd voice/face1.mp3')
                with open(join(dirname(__file__), imgfilePath), 'rb') as image_file:
                    results = visual_recognition.detect_faces(images_file=image_file)
                    print(json.dumps(results))
                    ageMin = 0
                    ageMax = 0
                    ageScore = 0
                    genderSix = ""
                    genderScore = 0
                    faceWidth = 0
                    faceHeight = 0
                    faceLeft = 0
                    faceTop = 0
                    nameFace = ""
                    scoreNameFace = 0
                    jsonencode =  json.loads(json.dumps(results))
 
                    if len(jsonencode['images'][0]['faces'])>0:
                        if 'min' in jsonencode['images'][0]['faces'][0]['age']: ageMin = jsonencode['images'][0]['faces'][0]['age']['min']
                        if 'max' in jsonencode['images'][0]['faces'][0]['age']: ageMax = jsonencode['images'][0]['faces'][0]['age']['max']
                        if 'score' in jsonencode['images'][0]['faces'][0]['age']: ageScore = jsonencode['images'][0]['faces'][0]['age']['score']
                        if 'gender' in jsonencode['images'][0]['faces'][0]['gender']: genderSix = jsonencode['images'][0]['faces'][0]['gender']['gender']
                        if 'score' in jsonencode['images'][0]['faces'][0]['gender']: genderScore = jsonencode['images'][0]['faces'][0]['gender']['score']
                        if 'width' in jsonencode['images'][0]['faces'][0]['face_location']: faceWidth = jsonencode['images'][0]['faces'][0]['face_location']['width']
                        if 'height' in jsonencode['images'][0]['faces'][0]['face_location']: faceHeight = jsonencode['images'][0]['faces'][0]['face_location']['height']
                        if 'left' in jsonencode['images'][0]['faces'][0]['face_location']: faceLeft = jsonencode['images'][0]['faces'][0]['face_location']['left']
                        if 'top' in jsonencode['images'][0]['faces'][0]['face_location']: faceTop = jsonencode['images'][0]['faces'][0]['face_location']['top']
                        if 'identity' in jsonencode['images'][0]['faces'][0]: 
                            nameFace = jsonencode['images'][0]['faces'][0]['identity']['name']
                            scoreNameFace = jsonencode['images'][0]['faces'][0]['identity']['score']
                            print('Name --> %s (%s)' % (nameFace, scoreNameFace))
                    
                        print(json.dumps(results))
                        if(genderSix=="FEMALE"):
                           six = "女性"
                           mp3file = "ageguess_female.mp3"
                        elif (genderSix=="MALE"):
                           six = "男性"
                           mp3file = "ageguess_male.mp3"
                        else:
                           six = ""

                        wordsSpeak1 = "我猜您是" + six
                        wordsSpeak2 = "年紀應該不超過"
                        wordsSpeak3 = "  " + str(ageMax) + "歲"
                        wordsSpeak4 = "至少也有"
                        wordsSpeak5 = "  " + str(ageMin) + "歲了" 
                        wordsSpeak6 = "等等，我認得你！"
                        if speakNameInEnglish != 1:
                            wordsSpeak7 = "您是不是" + nameFace + "?"
                        else:
                            wordsSpeak7 = "...您是不是..."

                        displayWords2 = "AGE: " + str(ageMin) + "~" + str(ageMax)
                        displayWords1 = "You are " + genderSix + "!"
                        print(wordsSpeak1)
                        print(wordsSpeak2)
                        print(wordsSpeak3)
                        print(wordsSpeak4)
                        print(wordsSpeak5)
                        print(wordsSpeak6)
                        print(wordsSpeak7)
    
                        disp.clear((0, 0, 0))
                        if nameFace != "":
                            if speakNameInEnglish==1: draw_rotated_text(disp.buffer, nameFace, (10, 10), 180, font, fill=(2,2,252))
                            draw_rotated_text(disp.buffer, wordsSpeak7, (10, 50), 180, font, fill=(2,2,252))
                            draw_rotated_text(disp.buffer, wordsSpeak6, (10, 90), 180, font, fill=(2,2,252))
                        if ageMin>0: 
                            draw_rotated_text(disp.buffer, wordsSpeak5, (10, 130), 180, font, fill=(255,255,255))
                            draw_rotated_text(disp.buffer, wordsSpeak4, (10, 170), 180, font, fill=(255,255,255))
                        if ageMax>0: 
                            draw_rotated_text(disp.buffer, wordsSpeak3, (10, 210), 180, font, fill=(255,255,255))
                            draw_rotated_text(disp.buffer, wordsSpeak2, (10, 240), 180, font, fill=(255,255,255))
                        if six != "": 
                            draw_rotated_text(disp.buffer, wordsSpeak1, (10, 280), 180, font, fill=(255,255,255))
                    
                        defaultDisplay = 0
                        disp.display()

                        txtTTS = ""
                        if six != "": 
                            if useTTS==1:
                                txtTTS = wordsSpeak1 + ", "
                            else:
                                os.system('omxplayer --no-osd voice/' + mp3file)
                        if ageMax>0: 
                            if useTTS==1:
                                txtTTS += wordsSpeak2 + wordsSpeak3 + ", "
                            else:
                                os.system('omxplayer --no-osd voice/age_not_over.mp3')
                                os.system('omxplayer --no-osd voice/number/' + str(ageMax) + '.mp3')
                        if ageMin>0: 
                            if useTTS==1:
                                txtTTS += wordsSpeak4 + wordsSpeak5 + ", "
                            else:
                                os.system('omxplayer --no-osd voice/atleast_age.mp3')
                                os.system('omxplayer --no-osd voice/number/' + str(ageMin) + '.mp3')
                                os.system('omxplayer --no-osd voice/end_age.mp3')
                        if nameFace != "":
                            if useTTS==1:
                                txtTTS += "。  " + wordsSpeak6 + ", " + wordsSpeak7
                        
                        if useTTS==1:
                            tts = gTTS( txtTTS, lang="zh-TW")
                            tts.save("tts.mp3")
                            os.system('omxplayer --no-osd tts.mp3')
                            #Use English to speak name
                            if speakNameInEnglish==1:
                                if nameFace!="":
                                    tts = gTTS( nameFace, lang="en")
                                    tts.save("name.mp3")
                                    os.system('omxplayer --no-osd name.mp3')
                        else:
                            if nameFace != "":
                                tts = gTTS( wordsSpeak6 + " " + wordsSpeak7, lang="zh-TW")
                                tts.save("tts.mp3")
                                os.system('omxplayer --no-osd tts.mp3')
                            #Use English to speak name
                                if speakNameInEnglish==1:
                                    tts = gTTS( nameFace, lang="es")
                                    tts.save("name.mp3")
                                    os.system('omxplayer --no-osd name.mp3')

                        #time.sleep(2)

    else:
        if time.clock() - lastFaceDectedTime >  waitSeconds:
            if defaultDisplay==0: displayDefaultImg()

displayDefaultImg()

while True:
    camera.capture('camera.jpg')

    try:
        checkFace("camera.jpg",1,1)
    except:
        draw_rotated_text(disp.buffer, "糟糕, 程式發生錯誤了!", (10, 10), 180, font, fill=(2,2,252))
