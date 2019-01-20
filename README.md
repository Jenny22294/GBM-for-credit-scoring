
# Introduction:
The Gradient Booting Machine (known as GBM, or sometimes simply Gradient Booting) is an approach of Machine Learning for both regression and classification problems. For more details about this model, please read https://en.wikipedia.org/wiki/Gradient_boosting.
In this post we will use the data set hmeq.csv from Credit Risk Analytics book: Measurement Techniques, Applications, and Examples in SAS. This is the data set used many times in this book for some classification models.
Our goal is based on information about customers applying for credit (occupation, loan purpose, loan amount, criminal record...) to build the model, in order to classify records for granting credit to customers.
With classification problems as just presented, Logistic is often the first thought model. However, the classification quality of this model is not good.
Using the GBM model, the classification results of the model achieved much higher accuracy than Logistic for the data set hmeq.csv. Specifically, quality classification of GBM model for test data set: (1) Accuracy is 94.43%, (2) the rate of misclassing bad records into good records (this is important) at a low level of 17.65% , (3) AUC = 0.97 (very high level according to the classification criteria for quality classification model./
Let's get started!

# Data Manipulation
```
import pandas as pd
credit = pd.read_csv("C:/Users/mlcl.local/Desktop/Self-Studied-R-and-Python/Python/hmeq.csv")

credit = credit.assign(MORTDUE = credit["MORTDUE"].fillna(credit["MORTDUE"]).mean(),
                      VALUE = credit["VALUE"].fillna(credit["VALUE"].mean()),
                      DEBTINC = credit["DEBTINC"].fillna(credit["DEBTINC"].mean()),
                      JOB = credit["JOB"].fillna("Other"),
                      REASON = credit["REASON"].fillna("Unknown"))
credit.head(10)


# Re-label for BAD and drop NAs:
df = credit.dropna(how = "any", axis = 0)
df2 = df.assign(BAD = df["BAD"].map(lambda x: "B" if x == 1 else "G"))
df2.head()

```
# GBM default

```
# Loading h2o:
import h2o 
h2o.init(nthreads = 2, max_mem_size = 6)

# Transform to h2o objects:
df = h2o.H2OFrame(df2)

# Declare input output:
response = "BAD"
predictors = df.names
predictors.remove("BAD")

# Split dataset:
train, test = df.split_frame(ratios = [0.8], seed = 1234)


# Use GBM and cross-validation with k folds = 5:

from h2o.estimators.gbm import H2OGradientBoostingEstimator
cv_gbm = H2OGradientBoostingEstimator(nfolds = 5, seed = 31)
cv_gbm.train(x = predictors, y = response, training_frame = train)

# Model summary:
cv_summary = cv_gbm.cross_validation_metrics_summary().as_data_frame()
cv_summary

```

## Model performance on test data

```
# Model performance on test data:
perf_cv_test = cv_gbm.model_performance(test)

# AUC on testdata:
perf_cv_test.auc()

# Gini:
perf_cv_test.gini()

# Confusion matrix:
perf_cv_test.confusion_matrix()

# Accuracy of model on test data:
1-0.0846

```
Although the model delivered 91.54% accuracy and Gini indicator is high, this default GBM forecasts false 40 bad records (label B) into good records (label G) equivalent to ~ 20% error. Miss clasification of good and bad records the banks can lose an opportunity to make a profit, but misleading bad records into good records is much more severe for financial institutions. So we need to improve the default GBM.


## Refining GBM

```
# Setup parameters for GBM:

gbm_tuned = H2OGradientBoostingEstimator(

    # number of ntree:
    ntrees = 10000,
    
    # Learning rate 
    learn_rate = 0.01,
    
    # Stop the algorithm early if the AUC validation does not improve 
    # at least 0.01% after 5 attempts
    stopping_rounds = 5, stopping_tolerance = 1e-04, stopping_metric = "AUC",
    
    # For each tree, input 80% variables:
    sample_rate = 0.8,
    
    # 80% of variables for every split:
    col_sample_rate = 0.8,
    
    # samples selection must ensure that the ratio of profile labels is preserved
    fold_assignment = "Stratified",
    
    # regenerate the results:
    seed = 1234,
    score_tree_interval = 10,
    
    # Cross validation with k = 5:
    
    nfolds = 5)

```


```
# Tune the model:
gbm_tuned.train(x = predictors, y = response, training_frame = train)

# Model performance on test data:
perf_for_test = gbm_tuned.model_performance(test)

# Confusion matrix:
perf_for_test.confusion_matrix()

# AUC:
perf_for_test.auc()

# Visualise AUC:
perf_for_test.plot()

# Gini:
perf_for_test.gini()

# Accuracy on testdata:
1-0.061

```

# Evaluate models on 30 different samples

Although the statistical criteria show that GBM classification quality is good (it is actually too good, we need to check more carefully). However we need some more solid and convincing evidence to use this model. To solve this problem, we will run and test this model for 30 different samples then get the information based on: 
BB - bad records correctly classified a bad record.
BG - misclassifying bad records into good ones (this is really bad).
GB - misclassify good records into bad ones (also bad but not equal to BG).
GG - correctly classified good record is a good record.

```
# Create a loop to evaluate the quality of the forecast model on 30 samples:

BB = []
BG = []
GB = []
GG = []


for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
    
    # Split data:
    train, test = df.split_frame(ratios = [0.5], seed = i)
    perf_for_test = gbm_tuned.model_performance(test)
    u = perf_for_test.confusion_matrix()
    v = u.table.as_data_frame()
    BB.append(v.loc[0, "B"])
    BG.append(v.loc[1, "B"])
    GB.append(v.loc[0, "G"])
    GG.append(v.loc[1, "G"])


# Convert into Series of pandas:     
BB = pd.Series(BB)
BG = pd.Series(BG)
GB = pd.Series(GB)
GG = pd.Series(GG)


# Get the results in Data frame: 
results = pd.DataFrame({"BB": BB, 
                        "BG": BG,
                        "GB": GB, 
                        "GG": GG}, 
                       columns = ["BB", "BG", "GB", "GG"])

results.head()


# Statistical description:
results.describe()


# Calculate accuracy on average over 30 runs of the model:

Accuracy = (results["BB"] + results["GG"]) / (results["BB"] + results["GG"] + results["BG"] + results["GB"])
Accuracy.describe()


### Visualisation:

%matplotlib notebook
import matplotlib as plt    
import matplotlib.pyplot as plt


plt.plot(Accuracy, label = "Accuracy Rate for 30 samples")
plt.legend()

```

If it is assumed that:
* a GG profile generates a 10% interest rate for a bank
* a BG profile makes a complete loss of capital
* each loan profile is granted 1 USD, 
then the profit statistics are:

```
profit = GG*0.1 - BG
profit.describe()

# Stop h2o: 
h2o.shutdown()
```

