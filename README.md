# Project Introduction

[ A-Lightweight-Model-for-Keyword-Spotting](https://github.com/Webersan/A-Lightweight-Model-for-Keyword-Spotting) by [Webersan](https://github.com/Webersan)

The project is in the development of continuous improvement.

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

## Result

While the program is running, you are supposed to speak to the microphone on your computer. The console displays a string containing the most probable phoneme.

```powershell
Listening...
c:\PythonProjects\DSP Project\main.py:50: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead data_int = np.fromstring(data_bytes, dtype=np.int16)

NNNNNNNNNNNNNNNNNNNNNNNNNssssssssoooooooorioNNotooooNNNNriririririririNNNNNNoooooooooooooooooooooooNNNNNtttNtNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNtttttNNttriNNsssssssoooooooNNNririririririririririririritNNNNooooooNNNNNttNtNNNNNririririNriNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNriNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN

Time cost: 7.512481212615967 s
Listening Complete.
```

Up to now, the model supports these phoneme:

's'

'o' 

'ri' (or pricisely 'i')

't'

N (nothing)

This model recognizes vowels better than consonants.



