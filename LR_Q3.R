# Install necessary packages if not already installed
install.packages("pROC")  # For ROC curve calculation

# Load required libraries
library(caret)
library(glmnet)
library(ggplot2)
library(pROC)

# Load your data (assuming you have the data in 'data')
data <- read.csv("Finalsurvey.csv")

# Keep only the relevant columns
data_relevant <- data[, c('BREXIT_VOTE', 'NHS_WAIT_TIME', 'NHS_AFFECT_TRUST', 'NHS_AFFECT_VOTE', 'NHS_DURING_COVID19', 'FUTURE_VOTE_NHS')]

# Data Cleaning: Fix inconsistencies in 'NHS_WAIT_TIME' column
data_relevant$NHS_WAIT_TIME <- gsub("haven?", "haven't", data_relevant$NHS_WAIT_TIME)

# Convert all categorical variables to factors
data_relevant$BREXIT_VOTE <- as.factor(data_relevant$BREXIT_VOTE)
data_relevant$NHS_WAIT_TIME <- as.factor(data_relevant$NHS_WAIT_TIME)
data_relevant$NHS_AFFECT_TRUST <- as.factor(data_relevant$NHS_AFFECT_TRUST)
data_relevant$NHS_AFFECT_VOTE <- as.factor(data_relevant$NHS_AFFECT_VOTE)
data_relevant$NHS_DURING_COVID19 <- as.factor(data_relevant$NHS_DURING_COVID19)
data_relevant$FUTURE_VOTE_NHS <- as.factor(data_relevant$FUTURE_VOTE_NHS)

# Fix factor levels of 'BREXIT_VOTE' to be valid R variable names
levels(data_relevant$BREXIT_VOTE) <- make.names(levels(data_relevant$BREXIT_VOTE))

# Split data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(data_relevant$BREXIT_VOTE, p = .7, list = FALSE)
trainData <- data_relevant[trainIndex, ]
testData  <- data_relevant[-trainIndex, ]

# Set up training control for cross-validation
traincross <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)

# Define the hyperparameter grid for logistic regression
grid <- expand.grid(
  alpha = 0,  # alpha = 0 corresponds to ridge regression, use 1 for lasso
  lambda = seq(0.001, 0.1, by = 0.01)
)

# Train the logistic regression model with hyperparameter tuning
set.seed(42)
logregmodel <- train(
  BREXIT_VOTE ~ .,
  data = trainData,
  method = "glmnet",
  trControl = traincross,
  tuneGrid = grid,
  family = "multinomial"
)

# Print the best hyperparameters
print(logregmodel$bestTune)

# Predict on the test data
predictions <- predict(logregmodel, newdata = testData)
probabilities <- predict(logregmodel, newdata = testData, type = "prob")

# Confusion matrix to evaluate the model
confmatrix <- confusionMatrix(predictions, testData$BREXIT_VOTE)
print(confmatrix)

# 1. ROC Curve Calculation for Each Class
roc_curve_leave <- roc(testData$BREXIT_VOTE, as.numeric(probabilities$Voted.to.Leave))
roc_curve_remain <- roc(testData$BREXIT_VOTE, as.numeric(probabilities$Voted.to.Remain))
roc_curve_no_vote <- roc(testData$BREXIT_VOTE, as.numeric(probabilities$Did.not.vote))

# Plot ROC Curves for each class
plot(roc_curve_leave, col = "blue", main = "ROC Curves for Brexit Vote")
plot(roc_curve_remain, col = "green", add = TRUE)
plot(roc_curve_no_vote, col = "red", add = TRUE)

legend("bottomright", legend = c("Voted to Leave", "Voted to Remain", "Did not Vote"),
       col = c("blue", "green", "red"), lwd = 2)

# 2. Visualization of Logistic Regression Coefficients
# Extract coefficients from the best model (using the best lambda)
coef_sparse <- coef(logregmodel$finalModel, s = logregmodel$bestTune$lambda)

# Convert the sparse matrix to a regular matrix for each class
coef_matrix_leave <- as.matrix(coef_sparse[[2]])  # Coefficients for "Voted to Leave"
coef_matrix_remain <- as.matrix(coef_sparse[[3]])  # Coefficients for "Voted to Remain"
coef_matrix_no_vote <- as.matrix(coef_sparse[[1]])  # Coefficients for "Did not Vote"

# Create dataframes for each class
coef_data_leave <- as.data.frame(coef_matrix_leave)
coef_data_remain <- as.data.frame(coef_matrix_remain)
coef_data_no_vote <- as.data.frame(coef_matrix_no_vote)

coef_data_leave$Feature <- rownames(coef_data_leave)
coef_data_remain$Feature <- rownames(coef_data_remain)
coef_data_no_vote$Feature <- rownames(coef_data_no_vote)

# Rename the coefficient columns for each class
names(coef_data_leave)[1] <- "Coefficient"
names(coef_data_remain)[1] <- "Coefficient"
names(coef_data_no_vote)[1] <- "Coefficient"

# Plot the coefficients for "Voted to Leave"
ggplot(coef_data_leave, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "blue") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients for 'Voted to Leave'",
       x = "Feature",
       y = "Coefficient") +
  theme_minimal()

# Plot the coefficients for "Voted to Remain"
ggplot(coef_data_remain, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "green") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients for 'Voted to Remain'",
       x = "Feature",
       y = "Coefficient") +
  theme_minimal()

# Plot the coefficients for "Did not Vote"
ggplot(coef_data_no_vote, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "red") +
  coord_flip() +
  labs(title = "Logistic Regression Coefficients for 'Did not Vote'",
       x = "Feature",
       y = "Coefficient") +
  theme_minimal()
