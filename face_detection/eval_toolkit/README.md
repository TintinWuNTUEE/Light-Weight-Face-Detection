## Usage

Requirement : numpy, tqdm

You can install the packages with 

````
pip3 install numpy tqdm
````

##### before evaluating ....

````
python3 setup.py build_ext --inplace
````

##### evaluating


````
python3 evaluation.py -p <your prediction file, ex. "./solution.txt"> -g <groud truth pickle file> [-lm]
````

