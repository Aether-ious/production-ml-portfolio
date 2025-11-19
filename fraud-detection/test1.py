import pandas as pd
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/fraud_detection_train.csv"
df = pd.read_csv(url)
df.to_csv("data/train_sample.csv", index=False)
print("Downloaded! Shape:", df.shape)

n/train.csv
--2025-11-18 19:50:26--  https://github.com/kingabzpro/ieee-fraud-detection-simplified/raw/main/train.csv
Resolving github.com (github.com)... 20.207.73.82
Connecting to github.com (github.com)|20.207.73.82|:443... connected.
HTTP request sent, awaiting response... 404 Not Found
2025-11-18 19:50:26 ERROR 404: Not Found.