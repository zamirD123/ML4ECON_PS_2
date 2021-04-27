---
title: "PS2"
subtitle: "ml4econ, HUJI 2021"
author: "Doron Zamir"
date: "4/26/2021"
output: 
  html_document: 
    keep_md: yes
---



# Load Packages

```r
if (!require("pacman")) install.packages("pacman")

pacman::p_load(
  tidyverse,
  tidymodels,
  vip,
  here,
  readxl,
  DataExplorer,
  caret,
  glmnet
)
```
set seed:

```r
set.seed(100)
```


# Linear regression
### 1. *Can we use the data for prediction without assumptions? Why?* 
Since we are not interested in the casual effect of the `x`'s, nor the underlying model related to the DGP, the only assumption we need to make in order to make predictions is  that of a **Stable DGP**.

### 2. *What is the downside in adding interactions? *
When adding interactions, we are increasing the number of featuers(`x`'s) in our model. The $k$ in $n/k$ ratio increases and our model becomes more saturated. This might cause to overfitting of the model. 

### 3. *Why is $E(u|X) \sim N(0,\sigma)$ assumptions strong?*
This assumptions are strong:
* although it could be the case that the residual converges in distribution to th Normal distribution (from CLT), it's not the only option: for example, if we observe the frequency that an event occurs, it's more likley that it will be from the Poisson distribution.

* it might not be the case that the error term has a mean of zero, but you can always ajust it to be zero by changing $\beta_0$.

### 4. *The $\beta$s CI:* 
From the above assumptions on the distribution of the error term, we can conclude the distribution of the $\beta$s, and by using the sample standart errors as an estimatior to it's SD, we can calculate the CI. 

## Emprical Exercise - The Wine Dataset

### Loading the Data:


```r
wine_raw <- 
  here("Data","winequality_red.csv") %>% 
  read_csv()
```

### Exploring the data
Note the data has no `na`.

plot continuous variables:

```r
wine_raw %>% 
  plot_histogram()
```

![](ps_2_files/figure-html/unnamed-chunk-4-1..svg)<!-- -->

plot variables against `quality`:

```r
wine_raw %>% 
  plot_boxplot(by = "quality")
```

![](ps_2_files/figure-html/unnamed-chunk-5-1..svg)<!-- -->

### Model

train/test split (70/30):

```r
wine_split <- wine_raw %>% initial_split(prop = 0.7)
wine_train <- wine_split %>% training()
wine_test <- wine_split %>% testing()
```

Simple Linear Model:

```r
wine_lm <- wine_train %>% 
  lm(quality~.,
     data = .)

wine_lm %>% tidy()
```

```
## # A tibble: 12 x 5
##    term                   estimate std.error statistic  p.value
##    <chr>                     <dbl>     <dbl>     <dbl>    <dbl>
##  1 (Intercept)             8.82    26.5          0.333 7.39e- 1
##  2 `fixed acidity`         0.0201   0.0322       0.625 5.32e- 1
##  3 `volatile acidity`     -1.19     0.153       -7.74  2.30e-14
##  4 `citric acid`          -0.246    0.184       -1.34  1.82e- 1
##  5 `residual sugar`        0.0177   0.0197       0.899 3.69e- 1
##  6 chlorides              -2.30     0.545       -4.21  2.76e- 5
##  7 `free sulfur dioxide`   0.00396  0.00273      1.45  1.47e- 1
##  8 `total sulfur dioxide` -0.00291  0.000876    -3.32  9.32e- 4
##  9 density                -4.35    27.0         -0.161 8.72e- 1
## 10 pH                     -0.483    0.239       -2.02  4.36e- 2
## 11 sulphates               0.988    0.144        6.87  1.10e-11
## 12 alcohol                 0.274    0.0332       8.24  4.83e-16
```

Prediction:

```r
wine_test_pred <- wine_test %>% 
  mutate(pred =predict(
    wine_lm,
    newdata = wine_test)) %>% 
  mutate(pred = round(pred,digits = 0))

wine_test_pred %>% select(pred) %>% sample_n(6)
```

```
## # A tibble: 6 x 1
##    pred
##   <dbl>
## 1     6
## 2     5
## 3     6
## 4     5
## 5     6
## 6     5
```


calculate metrics:

```r
wine_test_pred %>% 
  summarise(
    RMSE = mean((pred - quality)^2)^0.5,
    "R^2" = cov(pred,quality)^2,
    MAE = mean(abs(pred-quality))
)
```

```
## # A tibble: 1 x 3
##    RMSE  `R^2`   MAE
##   <dbl>  <dbl> <dbl>
## 1 0.645 0.0628 0.386
```

### RMSE vs. t-test

while t test checks for significant of the $\beta$s, the `RMSE` checks for the accuracy of the entire model. 


# Emprical Exercise - The Heart Dataset

### *Can we use linear regression for binary outcomes?*
Yes, we can use binary regression model (i.e., OLS when the $Y$ is binary). We might accure some problems, as the predictions are not limtied to the $[0,1]$ segment. It is more common to use `probit/logit`.
### Loading the Data:


```r
heart_raw <- 
  here("Data","heart.csv") %>% 
  read_csv()

heart_raw %>% head()
```

```
## # A tibble: 6 x 14
##     age   sex    cp trestbps  chol   fbs restecg thalach exang oldpeak slope
##   <dbl> <dbl> <dbl>    <dbl> <dbl> <dbl>   <dbl>   <dbl> <dbl>   <dbl> <dbl>
## 1    63     1     3      145   233     1       0     150     0     2.3     0
## 2    37     1     2      130   250     0       1     187     0     3.5     0
## 3    41     0     1      130   204     0       0     172     0     1.4     2
## 4    56     1     1      120   236     0       1     178     0     0.8     2
## 5    57     0     0      120   354     0       1     163     1     0.6     2
## 6    57     1     0      140   192     0       1     148     0     0.4     1
## # ... with 3 more variables: ca <dbl>, thal <dbl>, target <dbl>
```

### Exploring the data
Note the data has no `na`.

plot continuous variables:

```r
heart_raw %>% 
  plot_histogram()
```

![](ps_2_files/figure-html/unnamed-chunk-11-1..svg)<!-- -->
### Model

train/test split (70/30):

```r
heart_split <- heart_raw %>% initial_split(prop = 0.7)
heart_train <- heart_split %>% training()
heart_test <- heart_split %>% testing()
```


#### Linear regression:

```r
heart_lm <-  lm(target ~.,
                data =heart_train)

heart_lm %>% tidy()
```

```
## # A tibble: 14 x 5
##    term         estimate std.error statistic   p.value
##    <chr>           <dbl>     <dbl>     <dbl>     <dbl>
##  1 (Intercept)  0.757     0.359        2.11  0.0361   
##  2 age         -0.00105   0.00321     -0.327 0.744    
##  3 sex         -0.231     0.0579      -3.99  0.0000929
##  4 cp           0.115     0.0272       4.22  0.0000365
##  5 trestbps    -0.00157   0.00158     -0.990 0.323    
##  6 chol        -0.000798  0.000539    -1.48  0.140    
##  7 fbs          0.0110    0.0717       0.154 0.878    
##  8 restecg      0.0364    0.0481       0.757 0.450    
##  9 thalach      0.00440   0.00144      3.06  0.00249  
## 10 exang       -0.143     0.0614      -2.32  0.0213   
## 11 oldpeak     -0.0609    0.0304      -2.00  0.0464   
## 12 slope        0.0356    0.0503       0.707 0.480    
## 13 ca          -0.0739    0.0258      -2.86  0.00462  
## 14 thal        -0.119     0.0436      -2.73  0.00693
```

predict:

```r
p <- 0.9
heart_test_pred <- heart_test %>% 
  mutate(
    pred = predict(heart_lm,heart_test,type="response"),
    target_fct = factor(target),
    pred_class = if_else(pred > p,1,0),
    pred_class = factor(pred_class))

heart_test_pred$pred %>% max()
```

```
## [1] 1.226172
```

```r
heart_test_pred$pred %>% min()
```

```
## [1] -0.3533657
```
It's easy to se the prediction are not in [0,1].

ROC curve

```r
heart_test_pred %>%
  roc_curve(target_fct,pred) %>% autoplot()
```

![](ps_2_files/figure-html/unnamed-chunk-15-1..svg)<!-- -->

```r
 heart_test_pred  %>%
  conf_mat(target_fct,pred_class)
```

```
##           Truth
## Prediction  0  1
##          0 42 33
##          1  1 14
```

The model is not good, at all...

#### Logit

estimating logistic model:

```r
heart_logit <- glm(formula = target ~.,
                   data = heart_train,
                   family = "binomial")

heart_logit %>% summary()
```

```
## 
## Call:
## glm(formula = target ~ ., family = "binomial", data = heart_train)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.3655  -0.3939   0.1532   0.5855   2.8860  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  2.935738   3.017185   0.973  0.33055    
## age         -0.010563   0.027693  -0.381  0.70287    
## sex         -2.076254   0.554251  -3.746  0.00018 ***
## cp           0.866018   0.221887   3.903  9.5e-05 ***
## trestbps    -0.013461   0.012364  -1.089  0.27627    
## chol        -0.008680   0.004755  -1.825  0.06793 .  
## fbs         -0.131686   0.626322  -0.210  0.83347    
## restecg      0.363787   0.407502   0.893  0.37200    
## thalach      0.034392   0.012724   2.703  0.00688 ** 
## exang       -1.048060   0.489902  -2.139  0.03241 *  
## oldpeak     -0.451549   0.258939  -1.744  0.08119 .  
## slope        0.264766   0.411086   0.644  0.51953    
## ca          -0.550719   0.214671  -2.565  0.01031 *  
## thal        -0.958959   0.355134  -2.700  0.00693 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 292.79  on 212  degrees of freedom
## Residual deviance: 151.91  on 199  degrees of freedom
## AIC: 179.91
## 
## Number of Fisher Scoring iterations: 6
```

```r
logit_coef <- heart_logit %>%
  tidy() %>% select(estimate) %>% 
  rename(Logit = estimate)
```

making predictions:

```r
heart_log_predict <- heart_test %>% 
  mutate(
    pred=predict(heart_logit,heart_test,type="response")
  )

heart_log_predict$pred %>% max()
```

```
## [1] 0.9974865
```

```r
heart_log_predict$pred %>% min()
```

```
## [1] 0.001112794
```

calsify predictions for p = 0.9

```r
p <- 0.9
heart_log_predict <- heart_log_predict %>%
  mutate(
    pred_class = if_else(pred > p,"positive","negative"),
    pred_class = factor(pred_class),
    target_fct = if_else(target == 1,"positive","negative"),
    target_fct = factor(target_fct)
) 
```


matrics:

```r
conf_mtx <- heart_log_predict  %>%
  conf_mat(target_fct,pred_class)

conf_mtx %>% summary() %>% 
  pivot_wider(names_from = .metric, values_from = .estimate) %>% 
  select(accuracy,spec,sens)
```

```
## # A tibble: 1 x 3
##   accuracy  spec  sens
##      <dbl> <dbl> <dbl>
## 1    0.667 0.404 0.953
```

## Regularization

### *Penalty*

1. The term in the penalty must be in absolute or square terms, beacuse we want it to be positive, even if some of the $\beta$s are negative.
2. Since LASSO uses the absolute value in the penalty temm, the optimum related to it is usually when some of the $\beta$ are zero (triangle Vs. circle), and therefor it functions as "variable selection"


### *Modeling*

preparing data for glmnet:

```r
X_heart <- heart_test %>% select(-target) %>% as.matrix
Y_heart <- heart_test %>% select(target) %>% as.matrix
```

Fitting a model

```r
fit_ridge <- glmnet(
  x = X_heart,
  y = Y_heart,
  alpha = 0
)
plot(fit_ridge,xvar = "lambda")
```

![](ps_2_files/figure-html/unnamed-chunk-21-1..svg)<!-- -->


```r
cv_ridge <- cv.glmnet(
  x = X_heart,
  y = Y_heart,
  alpha = 0 
)
plot (cv_ridge)
```

![](ps_2_files/figure-html/unnamed-chunk-22-1..svg)<!-- -->

Coeffitients:

```r
coef_1 <- coef(cv_ridge, s = "lambda.min") %>%
  tidy() %>%
  as_tibble() %>%
  select(value) %>%
  rename(lambda_min = value)

coef(cv_ridge, s = "lambda.1se") %>%
  tidy() %>%
  as_tibble %>%
  select(-column) %>% 
  rename("lambda_1se" = value) %>% 
  bind_cols(coef_1,logit_coef)
```

```
## # A tibble: 14 x 4
##    row         lambda_1se lambda_min    Logit
##    <chr>            <dbl>      <dbl>    <dbl>
##  1 (Intercept)  0.682       0.769     2.94   
##  2 age         -0.000490    0.00139  -0.0106 
##  3 sex         -0.0725     -0.128    -2.08   
##  4 cp           0.0474      0.0688    0.866  
##  5 trestbps    -0.00133    -0.00221  -0.0135 
##  6 chol         0.0000960   0.000311 -0.00868
##  7 fbs          0.00381     0.00547  -0.132  
##  8 restecg      0.0444      0.0815    0.364  
##  9 thalach      0.00132     0.00112   0.0344 
## 10 exang       -0.111      -0.165    -1.05   
## 11 oldpeak     -0.0367     -0.0442   -0.452  
## 12 slope        0.0761      0.122     0.265  
## 13 ca          -0.0732     -0.138    -0.551  
## 14 thal        -0.0670     -0.104    -0.959
```

The coefficients are very different from the one form the Logistic regression, and some of them are really close to zero.
  
### problem with covariats = zero
In Econometrics, we are interested in the causual effect of an observable. Reducing the coeffitiant of a featcher to zero for reasons of model simplicity might cause us to lose important information.

###prediction:

```r
heart_predict <- heart_test %>%
  select(target) %>%
  mutate(
    log_pred = predict(heart_logit, newdata = heart_test),
    min_lam_pred = predict(cv_ridge,
                           as.matrix(select(heart_test,-target))
                           ,s = "lambda.min")[,1],
    "1se_lam_pred" = predict(cv_ridge,
                           as.matrix(select(heart_test,-target))
                           ,s = "lambda.1se")[,1]
  )
```


*The code for this file is on Github, and can be found [here](https://github.com/zamirD123/ML4ECON_PS_2.git)*
