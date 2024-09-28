# Load necessary libraries
library(tidyverse)
library(caret)
library(glmnet)
library(ggplot2)

# Step 1: Load the dataset
surveydata <- read.csv("Finalsurvey.csv", encoding = "ISO-8859-1")

# Step 2: Drop unnecessary column 'SOCIALMEDIA_TYPE_OTHERS'
surveydataclean <- surveydata %>%
  select(-SOCIALMEDIA_TYPE_OTHERS)

# Step 3: Convert 'BREXIT_VOTE' to binary (1 for "Voted to Leave", 0 otherwise)
surveydataclean$BREXIT_VOTE <- ifelse(surveydataclean$BREXIT_VOTE == "Voted to Leave", 1, 0)

# Step 4: One-Hot Encoding for Categorical Variables
X <- model.matrix(~ . - BREXIT_VOTE, data = surveydataclean)[, -1]

# Target variable 'BREXIT_VOTE'
y <- surveydataclean$BREXIT_VOTE

# Step 5: Train-Test Split
set.seed(42)
trainIndex <- createDataPartition(y, p = .8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

# Step 6: Define train control for cross-validation
traincross <- trainControl(method = "cv", number = 5)

# Step 7: Define the tuning grid for alpha (Lasso vs Ridge) and lambda (regularization strength)
tune_grid <- expand.grid(
  alpha = c(0, 0.5, 1),  # 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
  lambda = 10^seq(-3, 3, length = 10)  # Range of lambda values to search
)

# Step 8: Train the logistic regression model with regularization using glmnet
logregmodel <- train(
  as.data.frame(X_train), as.factor(y_train),
  method = "glmnet", 
  trControl = traincross,
  tuneGrid = tune_grid
)

# Step 9: Extracting Coefficients for Visualization
best_lambda <- logregmodel$bestTune$lambda
best_alpha <- logregmodel$bestTune$alpha

# Extract coefficients from the glmnet model
final_model <- logregmodel$finalModel
coef_matrix <- as.matrix(coef(final_model, s = best_lambda))

# Create a data frame with feature names and corresponding coefficients
coeffdf <- data.frame(
  Feature = rownames(coef_matrix),
  Coefficient = as.numeric(coef_matrix)
)

# Step 10: Filter non-zero coefficients for easier visualization
coeffdf <- coeffdf %>%
  filter(Coefficient != 0)  # Keep only non-zero coefficients

# Step 11: Visualization
ggplot(coeffdf, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +  # Flip axes to make it easier to read
  labs(
    title = "Effect of Factors on Brexit Vote (Voted to Leave)",
    x = "Feature",
    y = "Coefficient (Impact on Voting to Leave)"
  ) +
  theme_minimal()
