
import cv2
from brainframe.api import BrainFrameAPI
from time import strftime, localtime 
import os
from argparse import ArgumentParser

from PIL import ImageFont, ImageDraw, Image
from typing import Tuple
import numpy as np
import platform

def read_frame(stream_uri, frame_index):
    cap = cv2.VideoCapture(stream_uri)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    rst, frame = cap.read()
    if not rst:
        print(f"Failed to read frame: {frame_index}")
    cap.release()
    return frame

 
def detect_image(api, frame, capsule_names=None):
    if capsule_names is None:
        capsule_names = ["detector_person_openvino", "detector_face_openvino"]
    detections = api.process_image(frame, capsule_names, {})
    return detections
    
    
def render_rectangle(img,coords,bgrcolor,text=""):
    # img, (lefttop), (rightbottom), bgrcolor(255,0,0), thicknes, linetype
    # maybe draw name in rectangle topleft
    x0,y0 = coords[0]
    x1,y1 = coords[1]
    fontheight = 30

    cv2.rectangle(img, tuple(coords[0]), tuple(coords[2]), bgrcolor, 1, 4)
    #cv2 only surport ascii char, show other unicode need to use other way
    cv2.putText(img,text,(x0,y0+fontheight),cv2.FONT_HERSHEY_SIMPLEX,1,bgrcolor)
    
    
def corp_rectangle(frame,coords):
    # coords [[1095, 600], [1185, 600], [1185, 924], [1095, 924]]
    # [[x, y], [x, y], [x, y], [x,y]]
    # img[50:200, 700:1500]  # cut【y1,y2：x1,x2】
    return frame[ coords[0][1]:coords[2][1], coords[0][0]:coords[2][0] ]


def render_by_names(img,detections,names):
    bgr_dic = {'person': (0, 255, 0),'face':( 200, 50, 255),
            'vehicle':(0, 255, 255),'car':(0, 255, 255),'truck':(0, 255, 255),
            'motorcycle':(0, 255, 255), 'bus':(0, 255, 255)
            }

    for detection in detections:
        if detection.class_name in names:
            render_rectangle(img,detection.coords,bgr_dic[detection.class_name],detection.class_name)
    
#########pillow image render##########################
def render_rectangle_pillow(img, xy: Tuple[float, float, float, float], width=1,bgrcolor=(200,50,255)):
    # 'face':( 200, 50, 255),
    # img, (lefttop), (rightbottom), bgrcolor(255,0,0), thicknes, linetype
    # maybe draw name in rectangle topleft
    #x0,y0 = coords[0]
    #x1,y1 = coords[2]
    #cv2.rectangle(img, tuple(coords[0]), tuple(coords[2]), bgrcolor, 1, 4)
    #cv2 can't show chinese,only english 
    #cv2.putText(img,text,(x0,y0+fontheight),cv2.FONT_HERSHEY_SIMPLEX,1,bgrcolor)
    draw = ImageDraw.Draw(img)
    #draw.rectangle((x0,y0,x1,y1), outline=bgrcolor, width=width)
    draw.rectangle(xy, outline=bgrcolor, width=width)
    # Draw non-ascii text onto image
    #font = ImageFont.truetype("C:\Windows\Fonts\\simsun.ttc", fontsize)
    #draw.text((x0+400, y0), text, font=font,fill=bgrcolor)
    #draw.text((x0, y0), text, font=font,fill=bgrcolor)

def render_text_pillow(img, xy: Tuple[float, float], text="",fontsize=20,bgrcolor=(200,50,255)):
    
    if len(text)==0:
        return
    # windows song font 
    fontfile = "C:\Windows\Fonts\\simsun.ttc"
    draw = ImageDraw.Draw(img)
    
    # ubuntu font path
    if platform.system()=="Linux":
        fontfile = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    # Draw non-ascii text onto image
    font = ImageFont.truetype(fontfile, fontsize)
    #draw.text((x0+400, y0), text, font=font,fill=bgrcolor)
    draw.text(xy, text, font=font,fill=bgrcolor)


def render_rectangle_by_name(img, detections):
    #bgr_color = (0, 255, 0) # green
    bgr_color =(203, 192, 255) # pink
    #bgr_color = (0, 255, 255) # yellow
    #bgr_color = (255, 0, 0) # blue
    #fontsize = 20
    fontsize = 30
    for detection in detections:
        # draw rectangle 
        x0,y0 = detection.coords[0]
        x1,y1 = detection.coords[2]

        render_rectangle_pillow(img,(x0,y0,x1,y1),bgrcolor=bgr_color)
       
        if detection.class_name == "text":
            render_text_pillow(img,(x0,y0),
                            text = f"{detection.extra_data['ocr']}")
        elif detection.class_name == "image_quality_region":
            data = "\n".join([ f" {key}:{value}" 
                              for key, value in detection.extra_data.items()])
            render_text_pillow(img,(x0,y0),
                            text = f" {detection.class_name}:\n{data}",
                            fontsize = fontsize,bgrcolor=bgr_color)
        else: #detection.class_name == "person" or other detectors 
            attributes=""
            if detection.attributes :
                attributes = "\n".join([ f" {key}:{value}" 
                            for key, value in detection.attributes.items()])
            extra_data=""
            if detection.extra_data :
                extra_data = "\n".join([ f" {key}:{value:4.2f}" 
                            for key, value in detection.extra_data.items()])
            render_text_pillow(img,(x0,y0),
                            text = f" {detection.class_name}:\n{attributes}\n{extra_data}",
                            fontsize = fontsize ,bgrcolor=bgr_color)   

 
def save_image(filename,frame):
    ret = cv2.imwrite(filename,frame)
    if not ret:
        print(f"Failed to save frame to {filename}")
 


def process_frame(frame,api,capsule_names):
    names = ['person','vehicle','car','bus','truck']
    detections = detect_image(api, frame, capsule_names)
    # print(detections)
    # Could extend the feature here to render and crop persons
    render_by_names(frame,detections,names)


def process_frame_ex(frame,api,capsule_names): 
    detections = detect_image(api, frame, capsule_names)
    # render_by_names(frame,detections,names)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    render_rectangle_by_name(img,detections)
    new_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return new_frame


def process_video(filename, outputname, api, capsule_names,is_show=False):
    
    cap = cv2.VideoCapture(filename)
    
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 声明编码器和创建 VideoWrite 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #modified codec using 'avc3/h.264', for reduce mp4 file size
    #fourcc = cv2.VideoWriter_fourcc(*'avc3')
    print(f"input:{filename},output:{outputname}")
    out = cv2.VideoWriter(outputname, fourcc, fps, (width, height))

    frameindex = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # render frame like person or face 
        #process_frame(frame,api,capsule_names)
        new_frame = process_frame_ex(frame,api,capsule_names)
        if is_show:
             # 缩放图像,resize ration.
            scale_percent=20
            new_width = int(width * scale_percent / 100)
            new_height = int(height * scale_percent / 100)
            buffer=cv2.resize(new_frame, (new_width, new_height))
            cv2.imshow(filename,buffer)
            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # save frame to ouput
        out.write(new_frame)
        # print process ...
        frameindex += 1 
        print(f'\rprocess:frame={frameindex}, {(frameindex*100.0)/(framecount):4.2f}%',end=' ')
 
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
 
def main():

    # The capsules for person and face detection
    capsule_names = ["detector_person_administration","classifier_pose_closeup_ex"]
    #capsule_names = ["detector_person_and_vehicle_fast"]
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    output_path = "./output"
 
    parser = ArgumentParser()
    parser.add_argument("--stream", type=str, default=None, help="a video file")
    parser.add_argument("--show", action="store_true", help="show frame when processing")
    parser.add_argument("--server-url", default="http://localhost", help="url of brainframe server")

    args = parser.parse_args()
    if args.stream is None:
        print(f"usage: --stream videofile --show")
        exit(-1)
    
    stream_path = args.stream
    api = BrainFrameAPI(args.server_url)
    api.wait_for_server_initialization()
    
    # if not exsit ,os.path.dirname(output_path),create output directory
    if not os.path.exists(output_path):
        print(f"create directory: {output_path}")
        os.mkdir(output_path)
    
    output_filename = output_path +'/'+ os.path.basename(stream_path)
    
    print(f'start process {stream_path}:{strftime("%Y-%m-%d %H:%M:%S", localtime())}')
    process_video(stream_path, output_filename, api, capsule_names,is_show=args.show)
    print(f'\nsaved to {output_filename}:{strftime("%Y-%m-%d %H:%M:%S", localtime())}')
    
if __name__ == "__main__":
    main()

