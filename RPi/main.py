import numpy as np
import cv2
import dlib
import csv
from matplotlib import pyplot as plt

#Library that is used to choose file
import tkinter
from tkinter import filedialog
#사진 파일 선택
root = tkinter.Tk()
root.withdraw()#외부 윈도우창 최소화




detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./dilb_model/shape_predictor_68_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('./dilb_model/dlib_face_recognition_resnet_model_v1.dat')
prediction = 0.3 #얼굴 정확도 체크 함수: float ( 거리 제곱꼴)
mode = int(input("얼굴 등록 모드: 0 , 얼굴 인식 모드: 1 --> "))

if not(mode): #얼굴 등록 모드
    #파일 입출력
    facedatabase = open('./facedb/facedb.csv','w',encoding='utf8')
    w = csv.writer(facedatabase)

    photo_bgr = cv2.imread(filedialog.askopenfilename(parent=root, initialdir="./img/",title='처리할 사진을 선택해 주세요.',filetypes = (("jpg 파일","*.jpg"),("jpeg 파일","*.jpeg"),("모든 파일","*.*"))))
    photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)

    faces = detector(photo)
    photosfile = [photo for n in range(0,len(faces))]
    photos = [_ for _ in faces]
    for i in range(len(photos)):
        print(photos[i])
    names = list()

    faces = np.asarray(faces)
    print(photos)
    print(type(photos))

    #face(한 사람)당 인식 작업
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        landmarks = landmark_predictor(photo, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            print(photosfile[0])
            print(photosfile[1])
            print(photosfile[2])
            print(photosfile[0] == photosfile[1])
            print(photosfile[2] == photosfile[1])
            print(photosfile[2] == photosfile[0])
            cv2.circle(photosfile[faces.where(face)], (x, y), 4, (255, 0, 0), -1)

        face_descriptor = face_recognizer.compute_face_descriptor(photosfile[photos.index(face)],landmarks)
        #print(face_descriptor)
        image = cv2.cvtColor(photosfile[photos.index(face)], cv2.COLOR_RGB2BGR)
        image = photosfile[photos.index(face)]
        #사람 점 찍어서 표시
        plt.figure(f'Person {photos.index(face)}')
        plt.imshow(image, cmap=plt.cm.gray) 
        plt.xticks([]), plt.yticks([])
        plt.show()

        #인식한 사람 이름 적기
        name = input("이 사람의 이름은?")
        if name != "":
            names.append(name) #새로 뜬 사람이 누구인지 확인
            w.writerow([str(photos.index(face)),str(name)])
            w.writerow(face_descriptor)

        #파일 임시 저장
        cv2.imwrite(f'./cache/image{photos.index(face)}.jpg', image)
    facedatabase.close()
#얼굴 인식 모드
if mode:
    #다양한 상황에서의 얼굴 인식
    facedb = list() #이미 등록된 얼굴 데이터베이스 배열
    facenamedb = list() #얼굴 사람 이름 데이터베이스
    facerate = list() #얼굴 점수 유사도
    photo_bgr = cv2.imread(filedialog.askopenfilename(parent=root, initialdir="./img/",
    title='처리할 사진을 선택해 주세요.',filetypes = (("jpg 파일","*.jpg"),("jpeg 파일","*.jpeg"),
    ("모든 파일","*.*"))))
    photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    face = detector(photo)
    photos = [_ for _ in face]
    face = face[0] #ractangles -> ractangle
    if len(photos) == 0:
        print("얼굴이 안 보입니다.")
    if len(photos) > 1:
        print("2개 이상의 얼굴이 감지되었습니다.")
    else:
        print("얼굴 1개가 정상적으로 감지되었습니다.")
    with open(r'./facedb/facedb.csv','r'#사진읽기) as facedatabase:
        reader = csv.reader(facedatabase)
        for txt in reader:
            if len(txt) == 2: # (번호, 사람이름) 이렇게 써진 곳일 떄
                facenamedb.append(txt[1]) #사람 이름을 이름 데이터베이스에 추가
            if len(txt) == 128: #얼굴 정보가 128차원일 때.
                facedb.append(txt)
                #print(txt)
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    landmarks = landmark_predictor(photo, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(photo, (x, y), 4, (255, 0, 0), -1)

    face_descriptor = face_recognizer.compute_face_descriptor(photo,landmarks) #얼굴 특징을 128 차원으로 만든다.
    sum_rate = 0
    #print(face_descriptor[0])
    for i in range(len(facedb)):
        for j in range(len(txt)):
            sum_rate += pow((face_descriptor[j] - float(facedb[i][j])),2) #128차원 피타고라스
        facerate.append(sum_rate)
        if sum_rate < prediction:
            print(facenamedb[i],"인식!")
        else:
            print(facenamedb[i],"미인식!")
        print("sum_rate = ",sum_rate) #작을수록 정확
    #다양한 상황의 얼굴 사진으로 인식
    facedatabase.close()
