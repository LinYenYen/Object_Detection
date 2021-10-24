import os
from pathlib import Path
import argparse
import logging
import datetime
from typing import Tuple
import pandas as pd



####################################################################################################
#                                             Variable                                             #
####################################################################################################



####################################################################################################
#                                             Function                                             #
####################################################################################################
def split_train_val(data_dir: str, train_size: float, image_type: str) -> None:
    data_dir = Path(data_dir)
    image_dir = data_dir / 'images'
    label_dir = data_dir / 'labels'
    
    # images / label files 
    label_files = [l for l in label_dir.glob('**/*.txt') if l.stem != 'classes']

    # label summary
    summary_data = []
    for file in label_files:
        with open(file, 'r') as fr:
            line = fr.readline()
            while line:
                temp = line.split(' ', 1)
                temp[1] = os.fspath(file).replace(f'{os.sep}labels{os.sep}', f'{os.sep}images{os.sep}').replace('.txt', f'.{image_type}')
                summary_data.append(temp)
                line = fr.readline()
    df_summary = pd.DataFrame(data=summary_data, columns=['Label', 'File'])

    ### StratifiedShuffleSplit by class
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    labels = df_summary['Label'].unique()
    for label in labels:
        df_temp = df_summary.loc[df_summary['Label'] == label].sample(frac=1, ignore_index=True)
        split_num = round(df_temp.shape[0] * train_size)
        train_temp = df_temp.loc[:split_num]
        val_temp = df_temp.loc[split_num:]
        df_train = pd.concat([df_train, train_temp])
        df_val = pd.concat([df_val, val_temp])
    df_train = df_train['File'].drop_duplicates(keep='first')
    df_val = df_val['File'].drop_duplicates(keep='first')

    # train / val txt
    df_train.to_csv(f"{os.fspath(data_dir / 'train.txt')}", index=False, header=None)
    df_val.to_csv(f"{os.fspath(data_dir / 'val.txt')}", index=False, header=None)



def create_yaml(project_dir: str, name: str, data_dir: str, class_path: str, weights=None, pretrain=None) -> None:
    project_dir = Path(project_dir)
    yolo_dir = project_dir / 'yolov5'
    data_dir = Path(data_dir)
    class_path = Path(class_path)

    # classes
    with open(class_path, 'r') as fr:
        classes = fr.read()
        classes = classes.strip().split('\n')
        num_class = len(classes)

    # data yaml
    yolo_data_dir = yolo_dir / 'data'

    data = [
        f'train: {os.fspath(data_dir / "train.txt")}\n',
        f'val: {os.fspath(data_dir / "val.txt")}\n',
        f'nc: {num_class}\n',
        f'names: {classes}\n'
    ]

    with open(yolo_data_dir / f'{name}.yaml', 'w', encoding='utf-8') as fw:
        fw.writelines(data)

    ### model yaml
    yolo_model_dir = yolo_dir / 'models'

    if weights:
        # use yolov5 pre-train model
        with open(yolo_model_dir / f'{weights}.yaml', 'r', encoding='utf-8') as fr:
            model_yaml = fr.readlines()
        model_yaml[3] = f'nc: {num_class}  # number of classes\n'

    elif pretrain:
        # use self pre-train model
        with open(yolo_model_dir / f'{pretrain}.yaml', 'r', encoding='utf-8') as fr:
            model_yaml = fr.readlines()
        
    with open(yolo_model_dir / f'{name}.yaml', 'w', encoding='utf-8') as fw:
        fw.writelines(model_yaml)



def create_logging(project_dir: str, name: str, epochs: int, batch_size: int, imgsz: int, weights: str, pretrain: str) -> Tuple[logging.Logger, logging.Handler]:
    project_dir = Path(project_dir)
    log_dir = project_dir / 'logs'

    # Create log directory
    if not Path.exists(log_dir):
        Path.mkdir(log_dir)
    
    # Basic Setting
    filename = log_dir / f'{name}.log'
    format = '%(asctime)s %(name)s %(levelname)s %(message)s'
    formatter = logging.Formatter(format)

    # Config: set the format of output and levelname
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # FileHandle
    fileHandler = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(f'name: {name}')
    logger.info(f'epochs: {epochs}')
    logger.info(f'batch-size: {batch_size}')
    logger.info(f'img-size: {imgsz}')
    logger.info(f'weights: {weights}')
    logger.info(f'pre-train: {pretrain}')
    logger.info(f'start time: {datetime.datetime.now()}')

    return logger, fileHandler
    
    

def run_train(project_dir: str, name:str, epochs: int, batch_size: int, imgsz: int, device: str, weights: str, pretrain: str, logger: logging.Logger, handler: logging.Handler):
    project_dir = Path(project_dir)
    yolo_dir = project_dir / 'yolov5'
    cfg = yolo_dir / 'models' / f'{name}.yaml'
    data = yolo_dir / 'data'/ f'{name}.yaml'

    # cmd
    # use yolov5 pre-train model
    if weights:
        weights_file = yolo_dir / f'{weights}.pt'
        cmd = f'python train.py --name {name} --weights {weights_file} --cfg {cfg} --data {data} --epochs {epochs} --batch {batch_size} --imgsz {imgsz} --device {device}'
    # use self pre-train model
    elif pretrain:
        pretrain_file = yolo_dir / 'runs/train' / pretrain / 'weights/best.pt'
        cmd = f'python train.py --name {name} --weights {pretrain_file} --cfg {cfg} --data {data} --epochs {epochs} --batch {batch_size} --imgsz {imgsz} --device {device}'

    os.chdir(yolo_dir)
    print(cmd)
    os.system(cmd)
    logger.info(f'finish time: {datetime.datetime.now()}')
    logger.removeHandler(handler)



def parse_opt() -> argparse.ArgumentParser.parse_args:
    ### Argument
    parser = argparse.ArgumentParser(
        prog='My Train',
        description='My training pipeline of yolov5'
    )
    # Self argument
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--train-size', type=float, default=0.8, help='How much percent of data you what to be use as training data. Rest of data will be use as validate. Ex. 0 ~ 1')
    parser.add_argument('--image-type', type=str, default='jpg', help='The type of image')
    parser.add_argument('--class-path', type=str, help='The path of classes.txt')
    # Yolo argument
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # Conflict argument
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--weights', type=str, help='initial weights name. Ex. yolov5x or yolov5s or yolov5m or yolov5l')
    group.add_argument('--pretrain', type=str, help='pre-train model name')
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
    data_dir = opt.data_dir
    train_size = opt.train_size
    image_type = opt.image_type
    class_path = opt.class_path
    name = opt.name
    epochs = opt.epochs
    batch_size = opt.batch_size
    imgsz = opt.imgsz
    device = opt.device
    weights = opt.weights
    pretrain = opt.pretrain

    logger, handler = create_logging(project_dir=project_dir, name=name, epochs=epochs, batch_size=batch_size, imgsz=imgsz, weights=weights, pretrain=pretrain)
    split_train_val(data_dir=data_dir, train_size=train_size, image_type=image_type)
    create_yaml(project_dir=project_dir, name=name, data_dir=data_dir, class_path=class_path, weights=weights, pretrain=pretrain)
    run_train(project_dir=project_dir, name=name, epochs=epochs, batch_size=batch_size, imgsz=imgsz, device=device, weights=weights, pretrain=pretrain, logger=logger, handler=handler)

