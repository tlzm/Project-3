# Project-3

Deep learning for bearing remaining useful life prediction based on Femto bearing datasets (PHM-12)

This dataset has only the front 0.1s data per 10s. Therefore, inputting the whole data into the DL model directly is not allowed.

Furthermore, I have tried the follow methods:

1. Predicting the RUL with only stastisics features (including time domain, freq. domain and wavelets feature). However, the results is quite worse.

2. Predicting the RUL with raw data (setting the value of 0.1s as the value at that time)
