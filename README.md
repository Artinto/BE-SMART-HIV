## Scalable Deep Learning Platform with Nanotechnology for High-Sensitivity, Point-of-Care HIV Detection 
#### ✔ Python $\geq$ 3.8 and Ubuntu $\geq$ 16.04 are required
#### ✔ If you have any issues, please pull request.

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

<br>

## Trained weight
- Pretrained with Covid-19 data: [[Google Drive]](https://drive.google.com/file/d/1u06XF4sE__bN0HkJx5bolHDqB53T5U9f/view?usp=drive_link)   
- Fine-tuned with HIV LFA kit 1: [[Google Drive]](https://drive.google.com/file/d/1J8QmemOmWXEx6OK62zciikWfBsK6BBUh/view?usp=drive_link)   
- Fine-tuned with HIV LFA kit 2: [[Google Drive]](https://drive.google.com/file/d/18tzK26vYdUnOclKq6OaYLGcMqep7fqoP/view?usp=drive_link)   
