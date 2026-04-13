from ultralytics import YOLO
import sys

if __name__ == '__main__':
    # 加载模型
    model_cgf = sys.argv[1]
    data = sys.argv[2]
    model = YOLO('yolo12s.pt')  # build a new model from scratch
    imgsz = sys.argv[3]
    name = model_cgf.split('.')[0]+'_'+str(imgsz )+'_'+data.split('.')[0]
    print("-----------------------{}------------------".format(name))
    model.train(data=r"./dataset/{}".format(data),
                epochs=200,
                model="./Lib/{}".format(model_cgf),
                name = name,
                imgsz=imgsz,
                batch=16)