# Study of Deep Learning & TensorFlow
모두를 위한 딥러닝 강의 by Sung Kim ( https://hunkim.github.io/ml/ )

## TensorFlow Installation
- https://www.tensorflow.org/get_started/os_setup

### Requirements
- Python 3.5.1
- pip
- pyenv
- virtualenv
- autoenv

### Virtualenv installation

#### Install python 3.5.1:
```
$ brew update
$ brew upgrade
$ pyenv install 3.5.1
```

#### Create a Virtualenv environment in the directory:
```
$ pyenv virtualenv 3.5.1 learntf
```

#### Create autoenv:
```
$ touch .env
$ vi .env

# /.env
echo "====================="
echo "TensorFlow Playground"
echo " Activate virtualenv "
echo "====================="

pyenv activate learntf

$ cd ./.
```

#### Install TensorFlow:
```
# Mac OS X, CPU only, Python 3.4 or 3.5:
(learntf)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl

$ pip instsall --upgrade pip

(learntf)$ pip install --upgrade $TF_BINARY_URL
```

#### Test your installation:
```
$ python
...
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> print(sess.run(a + b))
42
>>>
```

#### Deactivate the environment:
`$ pyenvv deactivate`


## Jupyter Notebook Installation
```
# install
$ pip install jupyter
# Launch
$ jupyter notebook
```
