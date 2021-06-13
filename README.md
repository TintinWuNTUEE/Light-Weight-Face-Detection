# Light-Weight-Face-Detection

## Evaluation
in [Pytorch_Retinaface-master](./Pytorch_Retinaface-master)
```
python3 test_widerface.py --trained_model weights/mobilenet0.25_Final.pth --network mobile0.25 --dataset_folder ../face_detection/CV_dataset/test/

cp widerface_evaluate/widerface_txt/solution.txt ../face_detection/test
```