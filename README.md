#### ✔ Python 3.8 and Ubuntu 16.04 are required
## Enviroment setting
### 1. git install
```bash
$ sudo apt-get install git

$ git config --global user.name <user_name>
$ git config --global user.email <user_email>
```

<br>

### 2. Clone this Repository on your local path
```bash
$ cd <your_path>
$ git clone https://github.com/Artinto/hCG_multi_classification
```

<br>

### 3. Create the virtual enviroment (optional)
        
#### - install virtualenv
```bash
$ sudo apt-get install python3-venv
```

#### - Create your virtual enviroment (insert your venv_name)
```bash
$ python3 -m venv <venv_name>
```

#### - Activate your virtual environment
```bash
$ source ./<venv_name>/bin/activate
```
**&rarr; Terminal will be...**   ```(venv_name) $ ```
  
#### -  Install requirements packages
```bash
$ pip install -r requirements.txt
```
