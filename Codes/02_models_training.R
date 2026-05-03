# ==============================================================================
# PHASE 2: MODEL TRAINING AND EVALUATION
# ==============================================================================
# This script loads the preprocessed data from Phase 1 and trains 12 models
# (6 Linear, 6 Non-Linear) to predict readmission.
# ==============================================================================

suppressPackageStartupMessages({
  library(caret)        # Main modeling framework
  library(dplyr)        # Data manipulation
  library(ggplot2)      # Plotting
  library(glmnet)       # Lasso, Ridge, Elastic Net
  library(MASS)         # LDA, QDA
  library(pls)          # PLSDA
  library(e1071)        # SVM, Naive Bayes
  library(mda)          # Mixture Discriminant Analysis (for NDA)
  library(earth)        # Flexible Discriminant Analysis (FDA)
  library(nnet)         # Neural Networks, Multinomial Logistic
  library(kernlab)      # SVM Radial
  library(naivebayes)   # Naive Bayes
})

# ------------------------------------------------------------------------------
# 1. SETUP & DATA LOADING
# ------------------------------------------------------------------------------

# Configuration
CFG <- list(
  train_path = "outputs/train_processed_boxcox_cs_corr.csv", # Loading the "best" processed version
  test_path  = "outputs/test_processed_boxcox_cs_corr.csv",
  out_dir    = "outputs_models",
  seed       = 123,
  cv_folds   = 5   # Using 5-fold CV for speed (increase to 10 for more robustness)
)

if (!dir.exists(CFG$out_dir)) dir.create(CFG$out_dir, recursive = TRUE)
set.seed(CFG$seed)

# Load Data
cat("Loading data...\n")
train_df <- read.csv(CFG$train_path)
test_df  <- read.csv(CFG$test_path)

# Ensure the response variable is a factor (Critical for classification)
# Checking your previous script, the response column name is 'readmitted'
response_col <- "readmitted"
train_df[[response_col]] <- as.factor(train_df[[response_col]])
test_df[[response_col]]  <- as.factor(test_df[[response_col]])

# FIX: Rename levels to be valid R variable names (Required for caret classProbs)
# The levels likely contain symbols like "<" or ">" (e.g., "<30", ">30") which break caret.
# We replace them with text (LT, GT) to ensure valid, distinct names.
valid_level_names <- function(l) {
  l <- gsub("<", "LT", l)
  l <- gsub(">", "GT", l)
  make.names(l)
}

levels(train_df[[response_col]]) <- valid_level_names(levels(train_df[[response_col]]))
levels(test_df[[response_col]])  <- valid_level_names(levels(test_df[[response_col]]))

# Check levels to ensure consistency
cat("Class levels:", paste(levels(train_df[[response_col]]), collapse = ", "), "\n")

# Define Training Control
# We use Cross-Validation to estimate Kappa and Accuracy
fitControl <- trainControl(
  method = "cv",
  number = CFG$cv_folds,
  savePredictions = "final",
  classProbs = TRUE,  # Calculate probabilities for ROC if needed later
  verboseIter = TRUE  # Print progress
)

# List to store trained models
models_list <- list()

# ------------------------------------------------------------------------------
# 2. LINEAR MODELS (6 Models)
# ------------------------------------------------------------------------------
cat("\n--- Training Linear Models ---\n")

# 2.1 Logistic Regression (Multinomial since we have 3 classes)
cat("1. Logistic Regression (Multinomial)...\n")
set.seed(CFG$seed)
models_list[["Logistic"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "multinom", 
  trControl = fitControl, 
  trace = FALSE
)

# 2.2 Linear Discriminant Analysis (LDA)
cat("2. Linear Discriminant Analysis (LDA)...\n")
set.seed(CFG$seed)
models_list[["LDA"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "lda", 
  trControl = fitControl
)

# 2.3 Partial Least Squares Discriminant Analysis (PLSDA)
cat("3. PLSDA...\n")
set.seed(CFG$seed)
models_list[["PLSDA"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "pls", 
  trControl = fitControl,
  tuneLength = 5 # Auto-tune number of components
)

# 2.4 Lasso (Penalized - Alpha = 1)
cat("4. Lasso...\n")
set.seed(CFG$seed)
models_list[["Lasso"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "glmnet", 
  trControl = fitControl,
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 0.1, length = 10))
)

# 2.5 Ridge (Penalized - Alpha = 0)
cat("5. Ridge...\n")
set.seed(CFG$seed)
models_list[["Ridge"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "glmnet", 
  trControl = fitControl,
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.0001, 0.1, length = 10))
)

# 2.6 Elastic Net (Penalized - Mix of Lasso and Ridge)
cat("6. Elastic Net...\n")
set.seed(CFG$seed)
models_list[["ElasticNet"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "glmnet", 
  trControl = fitControl,
  tuneLength = 5 # Let caret search for optimal alpha and lambda
)

# ------------------------------------------------------------------------------
# 3. NON-LINEAR MODELS (6 Models)
# ------------------------------------------------------------------------------
cat("\n--- Training Non-Linear Models ---\n")

# 3.1 Nonlinear Discriminant Analysis 
# We use Mixture Discriminant Analysis (MDA) as the flexible/nonlinear extension of LDA.
# (Alternatively, QDA could be used here).
cat("7. Nonlinear Discriminant Analysis (MDA)...\n")
set.seed(CFG$seed)
models_list[["NDA_MDA"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "mda", 
  trControl = fitControl
)

# 3.2 Neural Networks
cat("8. Neural Networks...\n")
set.seed(CFG$seed)
models_list[["NeuralNet"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "nnet", 
  trControl = fitControl,
  trace = FALSE,
  linout = FALSE, # False for classification
  tuneLength = 3  # Keep small for speed, increase for better performance
)

# 3.3 Flexible Discriminant Analysis (FDA)
cat("9. Flexible Discriminant Analysis...\n")
set.seed(CFG$seed)
models_list[["FDA"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "fda", 
  trControl = fitControl
)

# 3.4 Support Vector Machines (Radial Basis Function)
cat("10. SVM (Radial)...\n")
set.seed(CFG$seed)
models_list[["SVM"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "svmRadial", 
  trControl = fitControl,
  tuneLength = 3
)

# 3.5 K-Nearest Neighbors (KNN)
cat("11. K-Nearest Neighbors...\n")
set.seed(CFG$seed)
models_list[["KNN"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "knn", 
  trControl = fitControl,
  tuneLength = 5
)

# 3.6 Naive Bayes
cat("12. Naive Bayes...\n")
set.seed(CFG$seed)
models_list[["NaiveBayes"]] <- train(
  as.formula(paste(response_col, "~ .")), 
  data = train_df, 
  method = "naive_bayes", 
  trControl = fitControl
)

# ------------------------------------------------------------------------------
# 4. MODEL COMPARISON & SELECTION
# ------------------------------------------------------------------------------
cat("\n--- Comparing Models ---\n")

# Collect resampling results
results <- resamples(models_list)

# Summary of results (Kappa and Accuracy)
summary_stats <- summary(results)
print(summary_stats)

# Plot performance
png(file.path(CFG$out_dir, "01_model_comparison_kappa.png"), width = 1000, height = 800)
dotplot(results, metric = "Kappa", main = "Model Comparison: Cohen's Kappa")
dev.off()

# Extract mean Kappa for sorting
model_kappas <- as.data.frame(summary_stats$statistics$Kappa)
model_kappas$Model <- rownames(model_kappas)
model_kappas <- model_kappas %>% arrange(desc(Mean))

cat("\nTop Models by Mean Kappa:\n")
print(head(model_kappas))

# Select Best 2 Models
best_model_name <- model_kappas$Model[1]
second_best_model_name <- model_kappas$Model[2]

cat(paste("\nBest Model:", best_model_name, "\n"))
cat(paste("Second Best Model:", second_best_model_name, "\n"))

best_model <- models_list[[best_model_name]]
second_best_model <- models_list[[second_best_model_name]]

# ------------------------------------------------------------------------------
# 5. FINAL EVALUATION ON TEST SET
# ------------------------------------------------------------------------------
cat("\n--- Evaluating Best 2 Models on Test Set ---\n")

evaluate_model <- function(model, name, test_data, ref_col) {
  pred_raw <- predict(model, newdata = test_data)
  
  # Create Confusion Matrix
  cm <- confusionMatrix(pred_raw, test_data[[ref_col]])
  
  cat(paste0("\n>>> Confusion Matrix: ", name, " <<<\n"))
  print(cm)
  
  # Save CM to text file
  sink(file.path(CFG$out_dir, paste0("confusion_matrix_", name, ".txt")))
  print(cm)
  sink()
  
  return(cm)
}

cm1 <- evaluate_model(best_model, best_model_name, test_df, response_col)
cm2 <- evaluate_model(second_best_model, second_best_model_name, test_df, response_col)

# Save the models for future use
saveRDS(models_list, file.path(CFG$out_dir, "all_trained_models.rds"))
cat("\nAll models and results saved to:", CFG$out_dir, "\n")