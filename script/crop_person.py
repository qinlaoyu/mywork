
import cv2 
from brainframe.api import BrainFrameAPI
import os
from argparse import ArgumentParser
 
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
    
    
def render_rectangle(img,coords,bgrcolor,name=""):
    # img, (lefttop), (rightbottom), bgrcolor(255,0,0), thicknes, linetype
    # maybe draw name in rectangle topleft
    cv2.rectangle(img, tuple(coords[0]), tuple(coords[2]), bgrcolor, 1, 4)
    
    
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
            render_rectangle(img,detection.coords,bgr_dic[detection.class_name])
    
    
    
def save_image(filename,frame):
    ret = cv2.imwrite(filename,frame)
    if not ret:
        print(f"Failed to save frame to {filename}")
    else:
        print(f"Ok to save frame to {filename}")
    return ret
    
def crop_detection_object(frame,detections,name='person'):
    clips=[]
    for detection in detections:
        if detection.class_name == name:
            clip = corp_rectangle(frame,detection.coords)
            clips.append(clip)
    return clips
    

def save_clips_image(clips,pathname):
    num =0
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    for clip in clips:
        num += 1
        # save person image
        save_image(f"{pathname}/pic{num}.jpg",clip)
        

# check every detection, select available person including 'face' coords
def check_avalib_person(detections):
    # distinguish person or face 
    avalilable_detects = []
    persons,faces = [], []
    for detection in detections:
        if detection.class_name == 'person':
            persons.append(detection)
        elif detection.class_name == 'face':
            faces.append(detection)
    #check face maybe fit some person data
    for person in persons:
        #topleft(x1,y1),bottomright(x2,y2);
        x1, y1 = person.coords[0]
        x2, y2 = person.coords[2]
        for face in faces:
            bx1,by1 = face.coords[0]
            bx2,by2 = face.coords[2]
            # check face is fited in persons's coords
            if(bx1>=x1 and bx2<=x2 and by1>=y1 and by2<=y2):
                # fine, correct persion image including 'face'
                avalilable_detects.append(person)
    return avalilable_detects
    
 
def corp_person(stream_path, frameindex):
    # The capsules for person and face detection
    # capsule_names = ["detector_person_openvino", "detector_face_openvino"]
    capsule_names = ["detector_person_administration", "detector_face_fast"]
    # directory to save pic file
    outpath = "../images"
    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
    
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    if not stream_path:
        stream_path = "../video/20230113-102816.mp4"
 
    api = BrainFrameAPI(bf_server_url)
    api.wait_for_server_initialization()
    
    #frameindex = 5
    frame = read_frame(stream_path, frameindex)
    if frame is None:
        print(f"Failed to read frame in {stream_path}")
        return
 
    detections = detect_image(api, frame, capsule_names)
    # print(detections)
 
    # Could extend the feature here to render and crop persons
    #render_by_names(frame,detections,['person','face'])
    render_by_names(frame,detections,['face'])
    # check and return if perfect person image including 'face' 
    available_dects = check_avalib_person(detections)

    # corp persons and save image
    clips = crop_detection_object(frame, available_dects, 'person')
    
    save_clips_image(clips, outpath)
    
    print(f"{len(clips)} clips in frame index({frameindex})")
    
    
def main():
    parser = ArgumentParser()
    parser.add_argument("-file", type=str, default=None, help="name of video for detecton")
    parser.add_argument("-index", type=int, default=100, help="index of frame in video")
    args = parser.parse_args()
    
    corp_person(args.file,args.index)
    
 
if __name__ == "__main__":
    main()

