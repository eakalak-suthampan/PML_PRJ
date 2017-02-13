# Classification in Weight Lifting Exercises Dataset
Eakalak Suthampan  
February 12, 2017  



#Overview
Dataset for this report is "Weight Lifting Exercises Dataset" which comes from [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

In this dataset, Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

The goal for this report will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise. This is the "classe" variable in the dataset.

#Loading And Preprocessing The Data
The training data for this report are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)


```r
if(!file.exists("pml-training.csv"))
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv")
if(!file.exists("pml-testing.csv"))
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
rawTrain <- read.table("pml-training.csv", header = TRUE, sep = ",", na.strings = c("","NA"))
rawTest <- read.table("pml-testing.csv", header = TRUE, sep = ",", na.strings = c("","NA"))
```

I found that there are many columns that have almost entirely "NA" values so I filtered these columns out (the remaining columns are the same on both training and testing).


```r
no_na_cols_train <- colnames(rawTrain[,colSums(is.na(rawTrain)) < 0.5*nrow(rawTrain)])
no_na_cols_test <- colnames(rawTest[,colSums(is.na(rawTest)) < 0.5*nrow(rawTest)])
training <- rawTrain[,no_na_cols_train]
testing <- rawTest[,no_na_cols_test]
```

I also filter out unnecessary columns which are not measurements from accelerometers.

```r
training <- training[,8:60]
testing <- testing[,8:59]
```

Since the testing data does not include classe variable so I will split training into training and pretesting (pretesting is used for evaluate accuracy).


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
set.seed(999)
inTrain <- createDataPartition(y=training$classe,p=0.8,list=FALSE)
pretesting <- training[-inTrain,]
training <- training[inTrain,]
str(training)
```

```
## 'data.frame':	15699 obs. of  53 variables:
##  $ roll_belt           : num  1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 1.43 ...
##  $ pitch_belt          : num  8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 8.18 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
##  $ gyros_belt_y        : num  0 0 0 0.02 0 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 -0.02 ...
##  $ accel_belt_x        : int  -21 -20 -22 -21 -21 -22 -22 -20 -21 -22 ...
##  $ accel_belt_y        : int  4 5 3 2 4 3 4 2 4 2 ...
##  $ accel_belt_z        : int  22 23 21 24 21 21 21 24 22 23 ...
##  $ magnet_belt_x       : int  -3 -2 -6 -6 0 -4 -2 1 -3 -2 ...
##  $ magnet_belt_y       : int  599 600 604 600 603 599 603 602 609 602 ...
##  $ magnet_belt_z       : int  -313 -305 -310 -302 -312 -311 -313 -312 -308 -319 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
##  $ pitch_arm           : num  22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 21.5 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0 0.02 0 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 0 ...
##  $ accel_arm_x         : int  -288 -289 -289 -289 -289 -289 -289 -288 -288 -288 ...
##  $ accel_arm_y         : int  109 110 111 111 111 111 111 109 110 111 ...
##  $ accel_arm_z         : int  -123 -126 -123 -123 -122 -125 -124 -122 -124 -123 ...
##  $ magnet_arm_x        : int  -368 -368 -372 -374 -369 -373 -372 -369 -376 -363 ...
##  $ magnet_arm_y        : int  337 344 344 337 342 336 338 341 334 343 ...
##  $ magnet_arm_z        : int  516 513 512 506 513 509 510 518 516 520 ...
##  $ roll_dumbbell       : num  13.1 12.9 13.4 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.3 -70.4 -70.4 -70.8 ...
##  $ yaw_dumbbell        : num  -84.9 -85.1 -84.9 -84.9 -84.5 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 -0.02 0 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -232 -232 -233 -234 -232 -234 -232 -235 -233 ...
##  $ accel_dumbbell_y    : int  47 46 48 48 48 47 46 47 48 47 ...
##  $ accel_dumbbell_z    : int  -271 -270 -269 -270 -269 -270 -272 -269 -270 -270 ...
##  $ magnet_dumbbell_x   : int  -559 -561 -552 -554 -558 -551 -555 -549 -558 -554 ...
##  $ magnet_dumbbell_y   : int  293 298 303 292 294 295 300 292 291 291 ...
##  $ magnet_dumbbell_z   : num  -65 -63 -60 -68 -66 -70 -74 -65 -69 -65 ...
##  $ roll_forearm        : num  28.4 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 27.5 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 -63.8 ...
##  $ yaw_forearm         : num  -153 -152 -152 -152 -152 -152 -152 -152 -152 -152 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 0.02 ...
##  $ gyros_forearm_y     : num  0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 0.02 ...
##  $ gyros_forearm_z     : num  -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 -0.03 ...
##  $ accel_forearm_x     : int  192 196 189 189 193 195 193 193 190 191 ...
##  $ accel_forearm_y     : int  203 204 206 206 203 205 205 204 205 203 ...
##  $ accel_forearm_z     : int  -215 -213 -214 -214 -215 -215 -213 -214 -215 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -16 -17 -9 -18 -9 -16 -22 -11 ...
##  $ magnet_forearm_y    : num  654 658 658 655 660 659 660 653 656 657 ...
##  $ magnet_forearm_z    : num  476 469 469 473 478 470 474 476 473 478 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
dim(pretesting)
```

```
## [1] 3923   53
```

#Classification Algorithms Selection
From the dataset, I expected that Random Forest and GBM (Gradient Boosting) will be good candidate algorithms. Decision tree is also included in this comparison just for showing the simple model. 

For each model I will use **5 K-Fold crossvalidation** to select its optimal model.

For speeding purpose I will use smaller training and testing data to perform the comparison.

```r
set.seed(7436)
trainingSmall <- training[sample(nrow(training), nrow(training)),]
testingSmall <- trainingSmall[1:1000,]
trainingSmall <- trainingSmall[1001:5000,]
train_control <- trainControl(method="cv", number=5)
```

### Decision Tree

```r
set.seed(12345)
model1 <- train(classe ~ ., data=trainingSmall, trControl=train_control, method="rpart")
```

```
## Loading required package: rpart
```

```r
model1
```

```
## CART 
## 
## 4000 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 3200, 3200, 3199, 3201, 3200 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.02909218  0.5672497  0.44581075
##   0.04731861  0.4637975  0.29143924
##   0.11251314  0.3295177  0.06625623
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.02909218.
```

Notice that, Decision Tree uses crossvalidation to find its optimal model accross the tuning parameter "cp".

### Random Forest

```r
set.seed(12345)
ptm <- proc.time() # start timer
model2 <- train(classe ~ ., data=trainingSmall, trControl=train_control, method="rf")
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
proc.time() - ptm  # stop timer
```

```
##    user  system elapsed 
##  142.68    1.04  143.95
```

```r
model2
```

```
## Random Forest 
## 
## 4000 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 3200, 3200, 3199, 3201, 3200 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9607453  0.9502754
##   27    0.9639994  0.9544131
##   52    0.9534981  0.9411034
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

Notice that, Random Forest uses crossvalidation to find its optimal model accross the tuning parameter "mtry".

### GBM

```r
set.seed(12345)
ptm <- proc.time() # start timer
model3 <- train(classe ~ ., data=trainingSmall, trControl=train_control, method="gbm", verbose = FALSE)
```

```
## Loading required package: gbm
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```
## Loading required package: plyr
```

```r
proc.time() - ptm  # stop timer
```

```
##    user  system elapsed 
##   69.34    0.22   69.64
```

```r
model3
```

```
## Stochastic Gradient Boosting 
## 
## 4000 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 3200, 3200, 3199, 3201, 3200 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7394961  0.6691761
##   1                  100      0.8077464  0.7563497
##   1                  150      0.8414940  0.7991396
##   2                   50      0.8412452  0.7985278
##   2                  100      0.8957446  0.8678719
##   2                  150      0.9219949  0.9011655
##   3                   50      0.8794968  0.8472478
##   3                  100      0.9282465  0.9091127
##   3                  150      0.9464965  0.9322460
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

Notice that, GBM uses crossvalidation to find its optimal model accross the tuning parameters "n.trees" and "interaction.depth".

### Models Comparison
Next, I will evaluate each model using testingSmall data and compare their accuracy.


```r
model1Cm <- confusionMatrix(testingSmall$classe, predict(model1,testingSmall))
model2Cm <- confusionMatrix(testingSmall$classe, predict(model2,testingSmall))
model3Cm <- confusionMatrix(testingSmall$classe, predict(model3,testingSmall))

paste("Model1 Decision Tree accuracy is",model1Cm$overall[1]*100,"%")
```

```
## [1] "Model1 Decision Tree accuracy is 56.7 %"
```

```r
paste("Model2 Random Forest accuracy is",model2Cm$overall[1]*100,"%")
```

```
## [1] "Model2 Random Forest accuracy is 98.2 %"
```

```r
paste("Model3 GBM accuracy is",model3Cm$overall[1]*100,"%")
```

```
## [1] "Model3 GBM accuracy is 95.7 %"
```

You can see that Random Forest is slightly better than GBM. So I select Random Forest to be used as classification algorithm. 

#The Selected Algorithm: Random Forest 

I will apply Random Forest to the large training data to build model then evaluate accuracy using pretesting data. For "randomForest" package, there are no need to do crossvalidation as stated in [http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr](http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr).

Caution: this execution chunk may takes a long time to executed. 


```r
library(randomForest)
#training <- training[sample(nrow(training), nrow(training)),]
#training <- training[1:5000,]
ptm <- proc.time() # start timer
set.seed(555)
rfFit <- randomForest(classe ~ ., data=training, proximity = TRUE, importance = TRUE)
proc.time() - ptm  # stop timer
```

```
##    user  system elapsed 
##  303.78    1.69  305.92
```

```r
confusionMatrix(pretesting$classe,predict(rfFit,pretesting))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    2  755    2    0    0
##          C    0    2  682    0    0
##          D    0    0    0  642    1
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9982          
##                  95% CI : (0.9963, 0.9993)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9977          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9974   0.9971   1.0000   0.9986
## Specificity            1.0000   0.9987   0.9994   0.9997   1.0000
## Pos Pred Value         1.0000   0.9947   0.9971   0.9984   1.0000
## Neg Pred Value         0.9993   0.9994   0.9994   1.0000   0.9997
## Prevalence             0.2850   0.1930   0.1744   0.1637   0.1840
## Detection Rate         0.2845   0.1925   0.1738   0.1637   0.1838
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9991   0.9980   0.9982   0.9998   0.9993
```

```r
plot(rfFit)
```

![](index_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

As you can see from the plot. The default parameter "ntree" is 500 but Error seems to be in steady state since trees < 100. So I can adjust "ntree" parameter to be 100 so that it can execute faster. 

For overfitting, You can see that accuracy on the training data is 99.6241799% and accuracy on pretesting data is 99.8215651%. Since both accuracies from training and pretesting are comparable So I hope it will not overfit :). 

### Outlier
From "randomForest" package, I can plot outliers as follow. 

```r
plot(outlier(rfFit),type="h",col=c("red", "green", "blue", "yellow", "black")[as.numeric(training$classe)], main="Outlier By Classe")
legend("topright",  lwd = 1, col = c("red", "green", "blue", "yellow", "black"), legend = levels(training$classe))
```

![](index_files/figure-html/unnamed-chunk-11-1.png)<!-- -->

### Variable Importance
From "randomForest" package, I can show the ploting of top 20 Variable Importance as follow.

```r
varImpPlot(rfFit, n.var = 20)
```

![](index_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

There are two metrics for measure Variable Importance. First is "MeanDecreaseAccuracy" which show that if you drop that variable out how much the accuracy will be decreased. So Higher "MeanDecreaseAccuracy" means more importance for that variable. Second is "MeanDecreaseGini" which is used to measure how well the variable can classify data. So Higher "MeanDecreaseGini" means more importance for that variable.

### Class Center

The class Center will show the centroid of features that represent each classe. 

```r
classCenter(training[,-53], training[,53], rfFit$prox)
```

```
##   roll_belt pitch_belt yaw_belt total_accel_belt gyros_belt_x gyros_belt_y
## A      1.45      5.690  -87.800                5         0.02         0.00
## B    115.50      6.370   -6.585               17         0.06         0.10
## C      1.92      5.490  -87.400                5         0.03         0.00
## D    118.00      7.030   -6.470               17         0.03         0.02
## E    134.00      5.115   30.450               19         0.03         0.06
##   gyros_belt_z accel_belt_x accel_belt_y accel_belt_z magnet_belt_x
## A        -0.03          -16            5           23            45
## B        -0.15          -17           44         -157            19
## C        -0.07          -15            7           17            30
## D        -0.15          -19           43         -157            21
## E        -0.07          -11           44         -170            40
##   magnet_belt_y magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm
## A           605          -314     0.00     0.000     0.0              32
## B           603          -315    10.85    -0.100     0.0              28
## C           599          -322     8.39     0.000     0.0              27
## D           603          -315     0.00    -7.275     0.0              21
## E           568          -412   -12.10   -12.500    -0.3              23
##   gyros_arm_x gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z
## A       0.020       -0.03       0.020      -275.0         111       -97.0
## B       0.400       -0.43       0.480       -15.0         -26       -56.5
## C       0.185       -0.27       0.230       -82.5          23       -32.0
## D       0.030       -0.29       0.305        53.0          -2       -26.0
## E       0.180       -0.37       0.480         9.0         -26       -25.0
##   magnet_arm_x magnet_arm_y magnet_arm_z roll_dumbbell pitch_dumbbell
## A       -368.0        331.0        512.0      45.29072     -31.464653
## B        391.0        173.5        386.5      81.85420      12.514867
## C        168.5        244.0        479.0     -15.28926     -31.571755
## D        600.0         76.5        354.0      66.65402       9.155322
## E        444.0        106.0        390.0      49.47230     -14.240075
##   yaw_dumbbell total_accel_dumbbell gyros_dumbbell_x gyros_dumbbell_y
## A   -69.212062                   18             0.16             0.08
## B    30.108832                   14             0.03             0.10
## C     7.967921                    7             0.11             0.06
## D   -10.997441                    8             0.13            -0.02
## E   -13.390701                   13             0.14             0.08
##   gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y accel_dumbbell_z
## A            -0.13              -47             48.0              -18
## B            -0.08               13             80.0               29
## C            -0.11              -22             -7.5                4
## D            -0.15                3             33.0               -3
## E            -0.16               -9             39.0               -7
##   magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z roll_forearm
## A              -523             311.0               -16         25.3
## B              -254             457.0               -23          0.0
## C              -504             238.0                27        133.0
## D              -459             352.0               -11          0.0
## E              -389             340.5                15          0.0
##   pitch_forearm yaw_forearm total_accel_forearm gyros_forearm_x
## A        -32.10        70.2                  34            0.02
## B         11.80         0.0                  36            0.02
## C         11.80        83.4                  36            0.02
## D         36.20         0.0                  35            0.05
## E         13.15         0.0                  37            0.22
##   gyros_forearm_y gyros_forearm_z accel_forearm_x accel_forearm_y
## A           0.000            0.00           104.0             188
## B           0.555            0.23           -73.0             111
## C           0.100            0.07           -40.0             256
## D           0.050            0.03          -173.5             147
## E           0.310            0.13           -86.5             170
##   accel_forearm_z magnet_forearm_x magnet_forearm_y magnet_forearm_z
## A          -161.0            -17.0            653.0            512.0
## B           -11.0           -309.5            518.0            487.5
## C          -102.5           -374.5            666.0            550.0
## D          -157.0           -604.0            373.0            493.0
## E           -26.0           -419.0            442.5            468.0
```

### out of sample error
out of sample error is an error on a new testing data which is different data from the training. Thus out of sample error in this Random Forest is the error on the pretesting data which is 

```r
paste((1 - confusionMatrix(pretesting$classe,predict(rfFit,pretesting))$overall[1])*100,"%")
```

```
## [1] "0.178434871271982 %"
```

# Prediction Test Cases
Here are predictions on 20 different test cases of the testing data.

```r
predict(rfFit,testing)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

