setwd("~/Homework/Homework Year 3/Data Mining Homework/DM HW 3")

spambase.df <- read.csv("spambase.csv", stringsAsFactors = TRUE)

#EDA
class(spambase.df)
dim(spambase.df)
nrow(spambase.df)
ncol(spambase.df)
names(spambase.df)
str(spambase.df)
head(spambase.df)
summary(spambase.df)

#aggragating data for averages of spam and nonspam
aggregate(. ~ Spam, spambase.df, mean)

# partition data
set.seed(2)
train.index <- sample(nrow(spambase.df), nrow(spambase.df) * 0.6)
train.df <- spambase.df[train.index, ]
valid.df <- spambase.df[-train.index, ]

# run logistic model and show coefficients and odds
log.reg.fit <- glm(Spam ~ ., data = train.df, family = "binomial")
summary(log.reg.fit)
data.frame(summary(log.reg.fit)$coefficient, odds = exp(coef(log.reg.fit)))

# evaluating model performance
library(gains)
pred <- predict(log.reg.fit, valid.df)
gain <- gains(valid.df$Spam, pred, groups = 100)
library(caret)
confusionMatrix(factor(ifelse(pred >= 0.5, 1, 0)), factor(valid.df$Spam), positive = "1")
par(mfrow = c(1, 1))

# lift chart
plot(c(0, gain$cume.pct.of.total * sum(valid.df$Spam)) ~ c(0, gain$cume.obs), 
     xlab = "# of cases", ylab = "Cumulative", type = "l")
lines(c(0, sum(valid.df$Spam)) ~ c(0, nrow(valid.df)), lty = 2)
r <- roc(valid.df$Spam, pred)
library(pROC)
auc(r)

# decile lift chart
gain <- gains(valid.df$Spam, pred)
heights <- gain$mean.resp / mean(valid.df$Spam)
dec.lift <- barplot(heights, names.arg = gain$depth, ylim = c(0, 3),
                    xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise Lift Chart")

#show coefficients and odds
data.frame(summary(log.reg.fit)$coefficient, odds = exp(coef(log.reg.fit)))


#selecting two different variable selection models

# create model with no predictors for bottom of search rangeS
spam.lm <- lm(Spam ~ ., data = train.df)
spam.lm.null <- lm(Spam ~ 1, data = train.df)

# use step() to run forward selection
library(forecast)
spam.lm.fwd <- step(spam.lm.null, scope = list(spam.lm.null, upper = spam.lm), 
                    direction = "forward")

summary(spam.lm.fwd)

train.lm.fwd <- predict(spam.lm.fwd, train.df)
accuracy(train.lm.fwd, train.df$Spam)

# evaluating model performance 2
pred2 <- predict(spam.lm.fwd, valid.df)
gain2 <- gains(valid.df$Spam, pred2, groups = 100)
confusionMatrix(factor(ifelse(pred2 >= 0.5, 1, 0)), factor(valid.df$Spam), positive = "1")

# lift chart 2
plot(c(0, gain2$cume.pct.of.total * sum(valid.df$Spam)) ~ c(0, gain2$cume.obs), 
     xlab = "# of cases", ylab = "Cumulative", type = "l")
lines(c(0, sum(valid.df$Spam)) ~ c(0, nrow(valid.df)), lty = 2)
r2 <- roc(valid.df$Spam, pred2)
library(pROC)
auc(r2)

# decile lift chart 2
gain2 <- gains(valid.df$Spam, pred2)
heights2 <- gain2$mean.resp / mean(valid.df$Spam)
dec.lift <- barplot(heights2, names.arg = gain2$depth, ylim = c(0, 3),
                    xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise Lift Chart")

#stepwise regression
spam.lm.step <- step(spam.lm.null, scope = list(spam.lm.null, upper = spam.lm), 
                       direction = "both")

summary(spam.lm.step)

# evaluating model performance 3
pred3 <- predict(spam.lm.step, valid.df)
gain3 <- gains(valid.df$Spam, pred, groups = 100)
confusionMatrix(factor(ifelse(pred3 >= 0.5, 1, 0)), factor(valid.df$Spam), positive = "1")

# lift chart 3
plot(c(0, gain3$cume.pct.of.total * sum(valid.df$Spam)) ~ c(0, gain3$cume.obs), 
     xlab = "# of cases", ylab = "Cumulative", type = "l")
lines(c(0, sum(valid.df$Spam)) ~ c(0, nrow(valid.df)), lty = 2)
r3 <- roc(valid.df$Spam, pred3)
library(pROC)
auc(r3)

# decile lift chart 3
gain3 <- gains(valid.df$Spam, pred3)
heights3 <- gain3$mean.resp / mean(valid.df$Spam)
dec.lift <- barplot(heights3, names.arg = gain3$depth, ylim = c(0, 3),
                    xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise Lift Chart")

