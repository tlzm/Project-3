# Project-3

Deep learning for bearing remaining useful life prediction based on Femto bearing datasets (PHM-12)

This dataset has only the front 0.1s data per 10s. Therefore, inputting the whole data into the DL model directly is not allowed.

Furthermore, I have tried the follow methods:

1. Predicting the RUL with only stastisics features (including time domain, freq. domain and wavelets feature). However, the results are quite worse.

2. Predicting the RUL with raw data (setting the value of 0.1s as the value at that time, mainly use the TCN), the results are worse than several state-of-art methods.

3. Predicting the RUL with time & freq. feature (STFT). (Neural network and wavelets rebuild denoising were used, and the result is quite interesting). As a result, this method's result is almost as same as the raw data.

4. Using the datasets (with rawdata method) to predict the continous learning, the results show life long learning is available for the real experiment data.
