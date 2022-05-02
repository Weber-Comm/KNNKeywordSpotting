# Project Introduction

[ A-Lightweight-Model-for-Keyword-Spotting](https://github.com/Webersan/A-Lightweight-Model-for-Keyword-Spotting) by [Webersan](https://github.com/Webersan)

Here presents a lightweight model for keyword spotting, which ismainly based on KNN method and MFCC. The project is in the development of continuous improvement currently.

# Environment

```
Python 3.10.3

LIBs				   VERSION
numpy                  1.21.6
scipy                  1.8.0
PyAudio                0.2.11
python-speech-features 0.6
matplotlib             3.5.1	(not necessary)
scikit-learn           1.0.2	(not necessary)
```



# A Way to Test the Model

```python
python main.py
```

## Modify test time duration

Change N to modify time duration, about 2.5 secs per 100 chunks.

N == 300 is suggested.

```python
for _ in range(N):	# main.py line 45
```

