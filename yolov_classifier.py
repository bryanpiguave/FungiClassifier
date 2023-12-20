from ultralytics import YOLO
import argparse


DATASET_PATH = 'dataset'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', default=DATASET_PATH)
parser.add_argument('--epochs', default=100)
parser.add_argument('--batch-size', default=16)
args = parser.parse_args()


# Load a model
model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)
# Train the model
results = model.train(data=args.dataset_path, task="classify",
                      epochs=args.epochs, imgsz=224,
                      batch=args.batch_size)