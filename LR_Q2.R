# Load necessary libraries
library(caret)
library(glmnet)
library(dplyr)
library(ggplot2)

# Load the data
data <- read.csv("Finalsurvey.csv")

# Clean the dataset - Remove rows with missing or NULL values
cleaneddata <- na.omit(data)

# Convert BREXIT_VOTE to binary (1 = Leave, 0 = Remain)
cleaneddata <- cleaneddata %>%
  filter(BREXIT_VOTE %in% c("Voted to Leave", "Voted to Remain"))
cleaneddata$BREXIT_VOTE <- ifelse(cleaneddata$BREXIT_VOTE == "Voted to Leave", 1, 0)

# Use exact age ranges like 18-24, 25-34 etc.
# Assume the column "Age_Range" exists in the dataset with values like "18-24", "25-34"
# If this column does not exist, modify the dataset to include these ranges
cleaneddata$Age <- as.factor(cleaneddata$Age)

# Convert social media type to factor
cleaneddata$SOCIALMEDIA_TYPE <- as.factor(cleaneddata$SOCIALMEDIA_TYPE)

# Convert other social media-related variables to numeric
cleaneddata$INFLUENCE_SOCIAL_MEDIA <- as.numeric(factor(cleaneddata$INFLUENCE_SOCIAL_MEDIA))
cleaneddata$ENGAGE_SOCAIL_MEDIA <- as.numeric(factor(cleaneddata$ENGAGE_SOCAIL_MEDIA))
cleaneddata$SOCIALMEDIA_VS_TRAD_NEWS <- as.numeric(factor(cleaneddata$SOCIALMEDIA_VS_TRAD_NEWS))
cleaneddata$POLITICAL_AD <- as.numeric(factor(cleaneddata$POLITICAL_AD))
cleaneddata$LOYALTY_CHANGE <- as.numeric(factor(cleaneddata$LOYALTY_CHANGE))

# ---- Step 2: Logistic Regression with Correct Age Ranges ----

# Select features for the model, including age range and social media type
X <- cleaneddata %>%
  select(Age, SOCIALMEDIA_TYPE, INFLUENCE_SOCIAL_MEDIA, ENGAGE_SOCAIL_MEDIA, 
         SOCIALMEDIA_VS_TRAD_NEWS, POLITICAL_AD, LOYALTY_CHANGE)
y <- cleaneddata$BREXIT_VOTE

# Convert categorical variables to dummy variables for logistic regression
X <- model.matrix(~ Age + SOCIALMEDIA_TYPE + INFLUENCE_SOCIAL_MEDIA + 
                    ENGAGE_SOCAIL_MEDIA + SOCIALMEDIA_VS_TRAD_NEWS + POLITICAL_AD + 
                    LOYALTY_CHANGE - 1, data = X)

# Split data into training and test sets
set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_index,]
y_train <- y[train_index]
X_test <- X[-train_index,]
y_test <- y[-train_index]

# Train logistic regression using glmnet (Lasso and Ridge)
trainedcross <- trainControl(method = "cv", number = 5)
tune_grid <- expand.grid(
  alpha = c(0, 0.5, 1),  # 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
  lambda = 10^seq(-3, 3, length = 10)  # Regularization strength
)

logregmodel <- train(
  x = X_train,
  y = as.factor(y_train),
  method = "glmnet",
  trControl = trainedcross,
  tuneGrid = tune_grid
)

# Best hyperparameters
optimal_parameters <- logregmodel$bestTune
print(optimal_parameters)

# Predict on test set
y_pred <- predict(logregmodel, X_test)

# Model performance
confusionMatrix(as.factor(y_pred), as.factor(y_test))

# ---- 1. Interaction Plot: Media Type and Age Impact on Voting ----
# This plot shows how media type and age range interact and influence the BREXIT vote (Leave or Remain).

ggplot(cleaneddata, aes(x = Age, fill = as.factor(BREXIT_VOTE))) +
  geom_bar(position = "fill") +
  facet_wrap(~SOCIALMEDIA_TYPE) +
  scale_fill_brewer(palette = "Set1", labels = c("Remain", "Leave")) +
  labs(title = "Impact of Age Range and Social Media Type on Voting Behavior",
       x = "Age Range", y = "Proportion of Voters",
       fill = "BREXIT Vote") +
  theme_minimal()

# ---- 2. Stacked Bar Plot: Social Media Type and Voting by Age Range ----
# This plot shows how different social media types influence the voting decision for different age ranges.

ggplot(cleaneddata, aes(x = Age, fill = SOCIALMEDIA_TYPE)) +
  geom_bar(position = "fill") +
  facet_wrap(~BREXIT_VOTE) +
  scale_fill_brewer(palette = "Set2", name = "Social Media Type") +
  labs(title = "Voting Decision by Age Range and Social Media Type",
       x = "Age Range", y = "Proportion of Voters") +
  theme_minimal()

# ---- 3. Coefficient Plot: Visualizing Logistic Regression Coefficients ----
# Extract coefficients from the model
best_logreg <- logregmodel$finalModel
best_lambda <- logregmodel$bestTune$lambda
coefficients <- coef(best_logreg, s = best_lambda)

# Prepare coefficient data for plotting
coeff_df <- as.data.frame(as.matrix(coefficients))
coeff_df$features <- rownames(coeff_df)
coeff_df <- coeff_df %>% filter(features != "(Intercept)")

ggplot(coeff_df, aes(x = reorder(features, s1), y = s1)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Impact of Social Media Features on Voting Behavior",
       x = "Features", y = "Coefficient Value (Impact on Voting)") +
  theme_minimal()
