import cv2
import tkinter as tk
from tkinter import filedialog


def open_image_dialog():

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'

        model = cv2.dnn_DetectionModel(frozen_model, config_file)

        classLabels = []
        filename = 'labels.txt'
        with open(filename, 'rt') as spt:
            classLabels = spt.read().rstrip('\n').split('\n')

        # greater this value better the reults tune it for best output
        model.setInputSize(440, 450)
        model.setInputScale(1.0/127.5)
        model.setInputMean((127.5, 127.5, 127.5))
        model.setInputSwapRB(True)

        img = cv2.imread(file_path)

        classIndex, confidence, bbox = model.detect(
            img, confThreshold=0.5)  # tune confThreshold for best results

        font = cv2.FONT_HERSHEY_COMPLEX

        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, boxes, (30, 25, 25), 2)
            cv2.putText(img, classLabels[classInd-1], (boxes[0] + 10, boxes[1] + 40),
                        font, fontScale=1, color=(0, 255, 0), thickness=2)

        cv2.imshow('result', img)
        cv2.waitKey(0)

        cv2.imwrite('result.png', img)


def open_video_dialog():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if file_path:
        config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    classLabels = []
    filename = 'labels.txt'
    with open(filename, 'rt') as spt:
        classLabels = spt.read().rstrip('\n').split('\n')

    # greater this value better the reults but slower. Tune it for best results
    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    cap = cv2.VideoCapture("test_video.mp4")
    ret, frame = cap.read()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 25 is the frame rate of output video you can change it as required
    video = cv2.VideoWriter('Video_Output.avi', fourcc, 24,
                            (frame.shape[1], frame.shape[0]))

    font = cv2.FONT_HERSHEY_PLAIN

    while (True):

        ret, frame = cap.read()

        classIndex, confidence, bbox = model.detect(
            frame, confThreshold=0.60)  # tune the confidence  as required
        if (len(classIndex) != 0):
            for classInd, boxes in zip(classIndex.flatten(), bbox):
                if (classInd <= 80):
                    cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv2.putText(frame, classLabels[classInd-1], (boxes[0] + 10,
                                boxes[1] + 40), font, fontScale=1, color=(0, 255, 0), thickness=2)

        video.write(frame)
        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    video.release()
    cv2.destroyAllWindows()


root = tk.Tk()

#getting screen width and height of display
width= root.winfo_screenwidth()
height= root.winfo_screenheight()
#setting tkinter window size
root.geometry("%dx%d" % (width, height))
root.title("Obect Detector")

wellcome_text = tk.Label(text="Wellcome to Object Detector Program",background="lightblue", padx=15,pady=30,font=("comicsansms",29,"bold"))
wellcome_text.pack()

button_style = {"background": "lightblue", "foreground": "black", "padx": 15, "pady": 10, "font":"comicsansms","border":20}
image_button = tk.Button(root, text="Open Image", command=open_image_dialog, **button_style)
image_button.pack(pady=20)

video_button = tk.Button(root, text="Open Video", command=open_video_dialog, **button_style)
video_button.pack(pady=20)

root.mainloop()