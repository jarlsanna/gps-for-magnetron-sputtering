# gps-for-magnetron-sputtering
Code for characterising sputtering processes in our self-driving lab. See the paper https://arxiv.org/pdf/2506.05999

## Running the code
This code was developed with Python version 3.11. To use this code, you can follow these steps

### 1. Clone the repository
If you set up shh connection use
```
git clone git@github.com:jarlsanna/gps-for-magnetron-sputtering.git
```
otherwise clone via https
```
git clone https://github.com/jarlsanna/gps-for-magnetron-sputtering.git
```

### 2. Python environment
After cloning the repo, navigate to the directory
```
cd gps-for-magnetron-sputtering
```
then create a python virtual environment (here called my_env), and activate it
```
python -m venv my_env
source my_env/bin/activate
```
### 3. Install modules
Now you can install the modules needed and run the code in an isolated environment. 
```
pip install -r requirements.tex
```

### 4. Example usage
See the notebook `example.ipynb`. It shows how to read the data, and an example of an active learning loop, using the functions in the other files. 
