# svm
# Data Credit Card
creditcard.csv terlalu besar download from here:
https://drive.google.com/file/d/1cLgkQ7kyaugevljOELLInGWRIImcZ1xM/view?usp=sharing
Simpan file ini di dalam folder dataset/

# Prerequisite
pip install -r requirements.txt

# Balancing Data
python ./balancing_data.py

# custom
python ./balancing_data.py --dr [number_of_rows]

# svm
# default
python ./svm.py

# custom
python ./svm.py --filename [custom_file] --lr 0.02 
