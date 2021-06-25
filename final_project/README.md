# Light-Weight-Face-Detection

## Environments


## folder structure
.
├── README
├── best_model.pth
├── presentation file
└── final_project           # source codes
    ├── data/
    ├── layers/
    ├── utils/
    ├── models/
        ├── net.py
        └── retinaface.py
    ├── train.py
    └── test_wider_face.py

## Training
```
python3 train.py --training_dataset <put/your/training_data/label.txt>
```

for example
```
python3 train.py --training_dataset ../face_detection/CV_dataset/train/label.txt
```

## Evaluation
After training for 250 epoches, you can start evalute the model.
```
python3 test_widerface.py --dataset_folder <put/your/dataset_folder> --save_folder <folder/to_save/your_solution> -m <your_model>
```

for example
```
python3 test_widerface.py --dataset_folder ../face_detection/CV_dataset/test/ --save_folder ./ -m ./weights/mobilenet0.25_Final.pth
```