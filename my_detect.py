import os
from pathlib import Path
import argparse



####################################################################################################
#                                             Variable                                             #
####################################################################################################



####################################################################################################
#                                             Function                                             #
####################################################################################################
def run_detect(project_dir: str, model_name: str, source: str, imgsz: list, conf_thres: float, iou_thres: float, device: str, save_txt: int, save_conf: int, nosave: int, auto: int):
    project_dir = Path(project_dir)
    yolo_dir = project_dir / 'yolov5'
    inference_dir = project_dir / 'inference_data'

    # 創建 inference_data directory
    if not Path.exists(inference_dir):
        Path.mkdir(inference_dir)

    source = Path(source)
    name = source.stem
    weights_file = yolo_dir / 'runs/train' / model_name / 'weights/best.pt'
    imgsz = f'{imgsz[0]}' if len(imgsz) == 1 else f'{imgsz[0]} {imgsz[1]}'
    
    # cmd
    # General Mode
    if not auto:
        cmd = f'python detect.py --weights {weights_file} --source {source} --imgsz {imgsz} --conf-thres {conf_thres} --iou-thres {iou_thres} --device {device} --save-txt {save_txt} --save-conf {save_conf} --nosave {nosave} --project {inference_dir} --name {name}'
    # Auto Mode
    else:
        cmd = f'python detect_auto.py --weights {weights_file} --source {source} --imgsz {imgsz} --conf-thres {conf_thres} --iou-thres {iou_thres} --device {device} --save-txt {save_txt} --save-conf {save_conf} --nosave {nosave} --project {inference_dir} --name {name}'
    
    os.chdir(yolo_dir)
    print(yolo_dir)
    print(cmd)
    os.system(cmd)



def parse_opt() -> argparse.ArgumentParser.parse_args:
    ### Argument
    parser = argparse.ArgumentParser(
        prog='My Detect',
        description='My detect pipeline of yolov5'
    )
    # Self argument
    parser.add_argument('--model-name', type=str, help='model which want to use')
    parser.add_argument('--source', type=str, help='image directory of detect')
    parser.add_argument('--auto', type=int, default=0, help='auto mode. auto detect the new file in the specify directory.')
    # Yolo argument
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', type=int, default=1, help='save results to *.txt')
    parser.add_argument('--save-conf', type=int, default=1, help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', type=int, default=0, help='do not save images/videos')
    opt = parser.parse_args()

    return opt



####################################################################################################
#                                               Main                                               #
####################################################################################################
if __name__ == '__main__':

    # 專案目錄
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # 檢查專案結構是否符合
    if not Path.exists(Path(project_dir) / 'yolov5'):
        print('Error Directory Form.')
        os._exit(0)
    
    opt = parse_opt()
    model_name = opt.model_name
    source = opt.source
    imgsz = opt.imgsz
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    device = opt.device
    save_txt = opt.save_txt
    save_conf = opt.save_conf
    nosave = opt.nosave
    auto = opt.auto

    run_detect(project_dir=project_dir, model_name=model_name, source=source, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, device=device, save_txt=save_txt, save_conf=save_conf, nosave=nosave, auto=auto)
