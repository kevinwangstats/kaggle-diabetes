# kaggle-diabetes

# Setting up
```
pip install kaggle
kaggle competitions download -c playground-series-s5e12
mkdir data
unzip playground-series-s5e12.zip -d data
echo "playground-series-s5e12.zip" >> .gitignore
echo "data/*.csv" >> .gitignore
```

```
pip install pandas matplotlib seaborn scipy scikit-learn
```

# Submission 
```
kaggle competitions submit -c playground-series-s5e12 -f data/test_data_linear_regression_output.csv -m "Linear Regression Model"
kaggle competitions submit -c playground-series-s5e12 -f data/test_data_h2o_automl_output.csv -m "H2O AutoML Model"
```
