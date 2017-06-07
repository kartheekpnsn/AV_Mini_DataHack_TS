setwd('C:\\Users\\ka294056\\Desktop\\time-series')

library(data.table)
library(xgboost)
library(ggplot2)

train_df_ts = fread('Data\\Train_JPXjxg6.csv')
test_df_ts = fread('Data\\Test_mvj827l.csv', sep = ',')

# # plot the time series
p = ggplot() +
		geom_line(data = train_df_ts, aes(x = 1:nrow(train_df_ts), y = Count), color = 'coral') +
		geom_point(data = train_df_ts, aes(x = 1:nrow(train_df_ts), y = Count), color = 'skyblue')

# # inferences # #
# There is an increasing trend
# The trend at the initial part is different from the later part of the training data

# # engineer features # #
engineerFeatures = function(data, flag = TRUE) {
	library(lubridate)
	Date = strptime(data$Datetime, '%d-%m-%Y %H:%M')
	orderDate = order(Date)
	data = data[orderDate, ]
	data[, Year := factor(format(Date, '%Y'))]
	data[, Hour := factor(format(Date, "%H"))]
	data[, WeekDay := factor(weekdays(Date))]

	if(flag) {
		data[, day := factor(format(Date, '%d'))]
		data[, month := factor(format(Date, '%m'))]
		data[, week_of_month := factor(ifelse(ceiling(mday(Date) / 7) == 5, 4, ceiling(mday(Date) / 7)))]
		data[, weekend_flag := factor(ifelse(WeekDay %in% c('Saturday', 'Sunday'), 1, 0))]
		data[, DayCount := floor(as.numeric(Date - Date[1])/(24 * 60 * 60))]
		data[, week_of_year := ifelse(months(Date) == "December", 
			as.numeric(format(Date-4, "%U"))+1, 
			as.numeric(format(Date+3, "%U")))]
	}
	data[, Datetime := NULL]
	return(data)
}

train_df = engineerFeatures(train_df_ts)
test_df = engineerFeatures(test_df_ts)

# # Data Split # #
dataSplit = function(data, p = 0.75) {
	index = 1:nrow(data)
	index = sample(index)
	return(index[1:round(length(index) * p)])
}

index = dataSplit(train_df)
otrain = train_df[index, ]
otest = train_df[-index, ]

# # EVALUATION MEASURE
rmse = function(original, predicted) {
	sqrt(sum((original - predicted)**2)/length(original))
}

# # XGBOOST # #
XGB = function(otrain, otest, seed = 1){
	library(Matrix)
	library(xgboost)
	# # one hot encoding
	train_matrix = sparse.model.matrix(Count ~. -1, data = otrain)
	test_matrix = sparse.model.matrix(Count ~. -1, data = otest)

	dtrain = xgb.DMatrix(data = as.matrix(train_matrix), label = as.numeric(otrain$Count))
	dtest = xgb.DMatrix(data = as.matrix(test_matrix), label = as.numeric(otest$Count))

	hyper_params = list(booster = "gbtree", # default
		objective = "reg:linear",
		eta = 0.02,
		gamma = 1,
		scale_pos_weight = 85,
		max_depth = 8,
		min_child_weight = 8,
		subsample = 0.9,
		colsample_bytree = 0.8,
		silent = 1,
		seed = seed
	)
	watchlist = list(eval = dtest, train = dtrain)
	fit = xgb.train(param = hyper_params, data = dtrain, nrounds = 500, print_every_n = 100, watchlist = watchlist)
	return(predict(fit, dtest))
}

# # ARIMA # #
train_df_ts = fread('Data\\Train_JPXjxg6.csv')
Date = order(strptime(train_df_ts$Datetime, '%d-%m-%Y %H:%M'))
train_df_ts = train_df_ts[Date, ]
train_df_ts[, Count := Count/2]
train_df_ts = ts(train_df_ts$Count)
index = 1:round(length(train_df_ts) * 0.7)
otrain_ts = train_df_ts[index]
otest_ts = train_df_ts[-index]

plot.ts(otrain_ts)
x11()
plot.ts(otest_ts)
library(forecast)
fit = auto.arima(log10(otrain_ts), approximation = FALSE, trace = FALSE) # 
print(summary(fit))
ARIMAPredict = (10 ** (forecast.Arima(fit, h = length(otest_ts))$mean))
print(rmse(original = otest_ts , predicted = ARIMAPredict))

# 1. There appeared to be a seasonality which was multiplicative in nature. 
#		Hence the data did not increase linearly.
# 2. The count remained at 2 for the first fraction of observations. 
#		As such the trend did not start immediately along with the data.
# 3. There appeared to be an impact of the day of the week on the data. 
#		It seemed to indicate a cyclic behavior in general.
# 4. Tune alpha (indicates level), beta (indicates trend) and gamma (indicates seasonality) variables as well.

# # Holt Winters # #
library(zoo)
library(TTR)

train_df_ts = fread('Data\\Train_JPXjxg6.csv')
Date = order(strptime(train_df_ts$Datetime, '%d-%m-%Y %H:%M'))
train_df_ts = train_df_ts[Date, ]
train_df_ts[, Count := Count/2]
train_df_ts = ts(train_df_ts$Count, freq = 365, start = c(2012))

hw1 = HoltWinters(train_df_ts, seasonal = 'multiplicative')
print(sqrt((hw1$SSE)/length(train_df_ts)))

# # Linear Regression # #
lmFit = lm(Count ~ ., data = otrain)
lmPredict = predict(lmFit, newdata = otest)
print(rmse(original = otest$Count, predicted = lmPredict))

# # XGBOOST # #
xgbPredict = XGB(otrain = otrain, otest = otest, seed = 294056)
print(rmse(original = otest$Count, predicted = xgbPredict))

# # Ensemble # #
print(rmse(original = otest$Count, predicted = (.2 * lmPredict + .8 * xgbPredict)))