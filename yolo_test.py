from ultralytics import YOLO
import sys

if __name__ == '__main__':
    # 消融模型
    #ablition_model = ["yolov8s","yolo11s","YOLOv11s+P2","YOLOv11s+P2+rP5","p2_rp5_new_yolo11s","p2_rp5_yolo11s"][-1]
    ablition_model = sys.argv[1]
    data_type = sys.argv[2]
    size = sys.argv[3]
    # 加载模型
    print("model:",ablition_model,"data:",data_type)
    model = YOLO('./runs/{}/weights/best.pt'.format(ablition_model))  # build a new model from scratch
    print('========================================================================')
    model.val(data=r"./dataset/{}.yaml".format(data_type), batch=32, imgsz =size, save_json=True,  split='test',conf=0.25, iou=0.5, name = ablition_model+"/time_"+data_type+'_'+str(size) )
    print('========================================================================')




