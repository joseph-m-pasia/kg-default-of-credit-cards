# install Kaggle API client
pip install kagglehub

# download data from Kaggle using Kaggle API
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# unzip file in PowerShell
Expand-Archive -Path default-of-credit-card-clients-dataset.zip -DestinationPath ../data