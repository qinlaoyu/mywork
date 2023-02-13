
import cv2
from brainframe.api import BrainFrameAPI
from time import strftime, localtime 
import os
 
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
        print(f"\nOK to save frame to {filename}")
    return ret


def process_frame(frame,api,capsule_names):
    names = ['person','vehicle','car','bus','truck']
    detections = detect_image(api, frame, capsule_names)
    # print(detections)
    # Could extend the feature here to render and crop persons
    render_by_names(frame,detections,names)



def process_video(filename, outputname, api, capsule_names):
    
    cap = cv2.VideoCapture(filename)
    
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 声明编码器和创建 VideoWrite 对象
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #modified codec using 'avc3/h.264', for reduce mp4 file size
    fourcc = cv2.VideoWriter_fourcc(*'avc3')
    out = cv2.VideoWriter(outputname, fourcc, fps, (width, height))

    frameindex = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {filename}")
            break

        # render frame like person or face 
        process_frame(frame,api,capsule_names)
        # save frame to ouput
        out.write(frame)
        # print process ...
        frameindex += 1 
        
        # only record 3 minutes 
        if frameindex%10 ==0: print(f'\rprocess:{(frameindex*100.0)/((fps*60*3)):4.2f}%',end=' ')
        if( frameindex>(fps*60*3) ):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
 
def main():

    # The capsules for person and face detection
    #capsule_names = ["detector_person_openvino", "detector_face_openvino"]
    capsule_names = ["detector_person_and_vehicle_fast"]
    # The video file name, it can be replaced by the other video file or rtsp/http streams
    #stream_path = "./video/20230113-102816.mp4"
    stream_path = "../video/street.mp4"
    output_path = "./output/output.mp4"
 
    # The url to access the brainframe server with rest api
    bf_server_url = "http://localhost"
 
    api = BrainFrameAPI(bf_server_url)
    api.wait_for_server_initialization()
    
    # if not exsit ,create output directory
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))

    print(f'start process {stream_path}:{strftime("%Y-%m-%d %H:%M:%S", localtime())}')
    process_video(stream_path, output_path, api, capsule_names)
    print(f'end process {stream_path}:{strftime("%Y-%m-%d %H:%M:%S", localtime())}')
    # corp persons
    
if __name__ == "__main__":
    main()

