# Environment setup


## Conda env

To install the environment type:

`conda env create --flie tf-graphs-probability.yml` 

in your `Anaconda Prompt`.

It will install `spektral` and other dependencies.


## Data

After installing `spektral`, copy the folder `TensorcellDataset` to `C:\Users\~\.spektral\datasets`


## MLFlow tracking

To start MLflow UI, open a new instance of `Anaconda Prompt` and activate the environment:

`conda activate tf-graphs-probability`

Next, in your `Anaconda Prompt` change the directory to the directory where you found this readme file.

Finally, type:

`mlflow ui` and copy the url to your browser.


