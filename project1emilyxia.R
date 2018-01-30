library(data.table)
library(sqldf)
library(doMC)
library(ROCR)
library(glmnet)
library(caret)
library(plyr)

#load data
customer_table <- fread("/Users/lulu/Downloads/customer_table.csv")
order_table <- fread("/Users/lulu/Downloads/order_table.csv")
product_table <- fread("/Users/lulu/Downloads/product_table.csv")
category_table <- fread("/Users/lulu/Downloads/category_table.csv")

##select target customer (who puchased once before 2016/11/22 and then were dormant for the following 3 months)

# change from scientific notation to the actual number
customer_table$customer_id <- as.character(customer_table$customer_id)
order_table$customer_id <- as.character(order_table$customer_id)
order_table$order_id <- as.character(order_table$order_id)
order_table$product_id <- as.character(order_table$product_id)
product_table$product_id <- as.character(product_table$product_id)

# find customers who only made one purchase before 2016/11/22
one_time_purchaser <- subset( 
  order_table[
    order_date<'20161122'&order_amount>0,
    .(count=.N, order_date=max(order_date), order_amount=max(order_amount),product_id=max(product_id)),
    by=customer_id
    ],
  count==1)

#find customers who made purchase between 2016/11/22 and 2017/02/22
purchase_again_customer <- unique(subset(order_table, order_date>='20161122'&order_date<'20170222'&order_amount>0),by='customer_id')

#find customers who were dormant between 2016/11/22 and 2017/02/22
dormant_3month <- sqldf("SELECT * 
                        FROM one_time_purchaser
                        WHERE customer_id NOT IN 
                        (SELECT customer_id FROM purchase_again_customer);")

target_user <- customer_table[customer_table$customer_id %in% dormant_3month$customer_id]
target_user <- cbind(target_user,dormant_3month[,c("order_date","order_amount","product_id")])

#find customers who purchased again between 2017/02/22 and 2017/05/22
purchase_again2_customer_id <- unique(subset(order_table, order_date>='2017022'&order_date<'20170522'&order_amount>0),by='customer_id')$customer_id

target_user$flag <- as.factor(ifelse(target_user$customer_id %in% purchase_again2_customer_id, 'YES','NO'))

rm(one_time_purchaser,dormant_3month,purchase_again_customer,purchase_again2_customer_id)


##data pre-processing

#remove and add features
target_user[,c("customer_id","last_visit_date","age")] <- NULL

target_user$tenure <- as.numeric(as.Date("2017-02-22") - as.Date(target_user$first_visit_date))
target_user$days_since_first_order <- as.numeric(as.Date("2017-02-22") - as.Date(as.character(target_user$order_date),format='%Y%m%d'))

target_user <- join(target_user, product_table, by='product_id', type='left')

target_user[,c("first_visit_date","order_date","product_id")] <- NULL

#deal with missing value
target_user$category_id[is.na(target_user$category_id)] <- 'unknown'
target_user$latest_device_class[is.na(target_user$latest_device_class)] <- 'unknown'
target_user[is.na(target_user)] <- 0

#dummy coding
top_countries <- as.character(as.matrix(target_user[,list(Count=.N), by = country][order(-Count)][1:30][,.(country)]))

target_user$country_updated <- ifelse(target_user$country %in% top_countries,target_user$country,'others')
target_user$country <- NULL

country_dummy <- model.matrix( ~ country_updated - 1, data=target_user)
gender_dummy <- model.matrix( ~ gender - 1, data=target_user)
category_dummy <- model.matrix( ~ category_id - 1, data=target_user)
deviceclass_dummy <- model.matrix( ~ latest_device_class - 1, data=target_user)

target_user_combined <- cbind(target_user,country_dummy,gender_dummy,category_dummy,deviceclass_dummy)
target_user_combined[,c("country_updated","gender","category_id","latest_device_class")] <- NULL
rm(country_dummy,gender_dummy,category_dummy,deviceclass_dummy)

#filter out zero variance variable
zero_variance <- names(Filter(function(x)(length(unique(x)) == 1),target_user_combined))
target_user_combined <- Filter(function(x)(length(unique(x)) > 1),target_user_combined)

## transform continuous variables
transform_columns <- c("user_feature","phone_feature","tablet_feature","family_size","number_of_devices","tenure","order_amount","days_since_first_order")
transformed_column     <- target_user_combined[ ,grepl(paste(transform_columns, collapse = "|"),names(target_user_combined)),with = FALSE]
non_transformed_column <- target_user_combined[ ,-grepl(paste(transform_columns, collapse = "|"),names(target_user_combined)),with = FALSE]

transformed_column_processed <- predict(preProcess(transformed_column, method = c("BoxCox","scale")),transformed_column)
transformedAll <- cbind(non_transformed_column, transformed_column_processed)

rm(customer_table,order_table,product_table,target_user,transformed_column_processed,transformed_column,non_transformed_column)


##train and evaluate model

#set training/test datasets
set.seed(99)

train_rate <- 0.6
training_index <- createDataPartition(transformedAll$flag, p = train_rate, list = FALSE, times = 1)

train_data <- transformedAll[training_index,]
test_data <- transformedAll[-training_index,]

train_x <- subset(train_data, select = -c(flag))
train_y <- as.factor(apply(subset(train_data, select = c(flag)), 2, as.factor))

test_x <- subset(test_data, select = -c(flag))
test_y <- as.factor(apply(subset(test_data, select = c(flag)), 2, as.factor))

registerDoMC(cores=6)

#Lasso Logistic Regression
model_glm_cv_lasso <- cv.glmnet(data.matrix(train_x),train_y,alpha = 1,family="binomial",type.measure="auc",parallel=TRUE)

lasso_predict <- predict(model_glm_cv_lasso, data.matrix(test_x),type='response')
lasso_pred <- prediction(lasso_predict,test_y)
lasso_perf_recall <- performance(lasso_pred,"prec","rec")
lasso_perf_roc <- performance(lasso_pred,"tpr","fpr")
lasso_perf_auc <- performance(lasso_pred,"auc")

plot(lasso_perf_recall)
plot(lasso_perf_roc)
lasso_perf_auc@y.values

#Ridge Logistic Regression
model_glm_cv_ridge <- cv.glmnet(data.matrix(train_x),train_y,alpha = 0,family="binomial",type.measure="auc",parallel=TRUE)

ridge_predict <- predict(model_glm_cv_ridge, data.matrix(test_x),type='response')
ridge_pred <- prediction(ridge_predict,test_y)
ridge_perf_recall <- performance(ridge_pred,"prec","rec")
ridge_perf_roc <- performance(ridge_pred,"tpr","fpr")
ridge_perf_auc <- performance(ridge_pred,"auc")

plot(ridge_perf_recall)
plot(ridge_perf_roc)
ridge_perf_auc@y.values

#Random Forest
rf <- foreach(ntree=rep(200, 6), .combine='c', .multicombine=TRUE,
              .packages='randomForest') %dopar% {
                randomForest(train_x, train_y, ntree=ntree)
              }

rf_predict <- predict(rf, data.matrix(test_x),type='response')
rf_pred <- prediction(rf_predict,test_y)
rf_perf_recall <- performance(rf_pred,"prec","rec")
rf_perf_roc <- performance(rf_pred,"tpr","fpr")
rf_perf_auc <- performance(rf_pred,"auc")

plot(rf_perf_recall)
plot(rf_perf_roc)
rf_perf_auc@y.values