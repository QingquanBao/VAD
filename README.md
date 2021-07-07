# Voice Activity Detection
This is a project in AI2615 in SJTU implemented by Qingquan Bao.
The code is still be continued.

## Requirements
All available in `requirements.txt`
To install them, you can run `pip install -r requirements.txt`.

## Temporal feature + naive classifier
In `utils/time_feature_extraction.py`, we implement two kinds of temporal feature ZCR (Zero Crossing Rate) and energy.
In `LRtest.py` and `model/state_machine.py`, we implement **Logisitic Resgression** and **State Machine classifier** to detect voice activity in develop dataset.

To predict labels in new data, run 
`python vad4test.py --model=LR --featType=Time --testdirPath=<your test file directory path> --outPath=<the output .txt path u wish>` 

## Spectral feature + GMM
Spectral data is extracted in `utils/spectralFeature.py` where we implement FBank and MFCC.
In `gmm.py`, we implement **MFCC+GMM** 

To predict labels in new data with **MFCC+GMM**, run
`python gmm.py`

To predict labels in new data, run 
`python vad4test.py --model=GMM --featType=MFCC --testdirPath=<your test file directory path> --outPath=<the output .txt path u wish>`

## Spectral feature + LSTM
The model is implementde in `model/lstm.py` and now the architecture **only support MEL40 feature**.

To predict labels in new data, run
`python vad4test.py --model=LSTM --featType=MEL --testdirPath=<your test file directory path> --outPath=<the output .txt path u wish>`

## Result
| model | auc | eer | acc(train) | acc(test) |
| ------| -----| ----|---- | -----|
| all 1                         | 0.5| 0.9999| 0.815|
| TimeFeat + LR                 |0.8575 | 0.2127| 0.902 |
| TimeFeat + preSmooth + LR     | 0.9291 | 0.1060| **0.9512**|
| TimeFeat + StateMachine + postSmooth| 0.9453| 0.0943| 0.9316|
| MFCC20 + GMM                  | 0.9185 | 0.0970 | 0.9282 | 0.9300
| MFCC20 + GMM + Smooth         | 0.9788 | 0.0700 | 0.9467| 0.9597
| Mel40 + LSTM + Smooth         | **0.9902** | **0.0389** |  | **0.9726** |
