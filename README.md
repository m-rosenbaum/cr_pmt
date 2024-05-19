# Costa Rica Proxy Means Test
This repo includes a model and data for the CAPP 30254 - Machine Learning for Public Policy Class Supervised Learning Project. It includes work from Magdalena Barros, Paula Cadena, and Michael Rosenbaum.

[Assignment details](https://docs.google.com/document/d/1IYufSd4UT8MuyYqZS3j7u4YLjTr788-QhyVLXdI8Oeo/edit#heading=h.6ort0f9f7ej2)


## Project Summary
We design a data flow to implement a machine learning-based proxy means test (PMT) to classify poverty status into 4 categories. This work uses a [Kaggle Dataset](https://www.kaggle.com/competitions/costa-rican-household-poverty-prediction/data) including partially cleaned data from the Inter-american Development Bank (IDB) on 2,988 Costa Rican households. 

We implement a grid search of hyperparameters across penalized multinomial regressions, random forests, and KNN. Based on this procedure, we select a random forest model to predict poverty status. We achieved an "F1 macro" score of 37% predictive power on our test sample measured by F1 scores, which average the F1 score of each label. This corresponds to 62% overall accuracy due to the sample imbalance. Although the model has substantively higher than random chance estimates of poverty status, it tends to predict false negatives for the higher poverty categories at higher rates, which would not be useful in the PMT context. 

Our complete approach and details are available in a [project report](https://github.com/m-rosenbaum/cr_pmt/blob/267ed3fc37b76b038184fe442067ee0eba9b0bef/report/report.ipynb).  

## Acknowledgments
Professor: Chenhao Tan

Teaching Assistants: Karen Zhou, Katherine Dumais