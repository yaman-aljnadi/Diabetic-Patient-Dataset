# install.packages("openxlsx")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(caret)
  library(reshape2)
  library(e1071)
  library(stringr)
  library(rlang)
  # Added specific modeling libraries to prevent runtime errors
  library(MLmetrics)
  library(naivebayes)
  library(klaR)
  library(mda)    # Required for Non-Linear Discriminant Analysis
  library(earth)  # Required for Flexible Discriminant Analysis (FDA)
  library(glmnet) # Required for Lasso/Ridge/ElasticNet
  library(pls)    # Required for PLS
  library(nnet)   # Required for Neural Networks
  library(kernlab)# Required for SVM
})

CFG <- list(
  data_path = "../Datasets/diabetic_data_project.csv",
  response  = "readmitted",
  seed      = 123,
  out_dir   = "outputs",
  save_plots = TRUE,
  train_prop = 0.80,
  k_knn = 5,
  nzv_freq_cut = 95/5,
  cor_cutoff = 0.90,
  pca_var_thresh = 0.95
)

set.seed(CFG$seed)
if (!dir.exists(CFG$out_dir)) dir.create(CFG$out_dir, recursive = TRUE)

gsave <- function(filename, plot = last_plot(), width = 8, height = 6, dpi = 300) {
  if (isTRUE(CFG$save_plots)) ggsave(file.path(CFG$out_dir, filename), plot = plot, width = width, height = height, dpi = dpi)
}

replace_q_na <- function(df) { df[df == "?"] <- NA; df }

split_types <- function(df, response) {
  list(
    cat = setdiff(names(Filter(function(x) is.character(x) || is.factor(x), df)), response),
    num = setdiff(names(Filter(is.numeric, df)), response)
  )
}

plot_response <- function(df, response) {
  a <- df %>% count(!!rlang::sym(response), name = "Count") %>% mutate(Perc = round(100*Count/sum(Count),1))
  p <- ggplot(a, aes(!!rlang::sym(response), Count, fill = !!rlang::sym(response))) +
    geom_col(width = .65, show.legend = FALSE) +
    geom_text(aes(label = paste0(Count, " (", Perc, "%)")), vjust = -0.4, size = 4) +
    labs(title = "Class distribution", x = NULL, y = "Count") +
    theme_minimal(base_size = 13)
  print(p); gsave("01_class_distribution.png", p, 7, 5)
  invisible(a)
}

missing_plot <- function(df) {
  m <- colSums(is.na(df))/nrow(df)*100
  d <- tibble(Column = names(m), Missing = as.numeric(m)) %>% filter(Missing>0) %>% arrange(desc(Missing))
  if (nrow(d)==0) return(invisible(d))
  p <- ggplot(d, aes(reorder(Column, Missing), Missing)) +
    geom_col(width=.7, fill="#FF6347") + coord_flip() +
    geom_text(aes(label=paste0(round(Missing,1),"%")), hjust=-0.1) +
    labs(title="Missing values by column", x=NULL, y="%") +
    theme_minimal(base_size = 13)
  print(p); gsave("02_missingness.png", p, 8, 6)
  invisible(d)
}

nzv_drop <- function(X, freqCut = 95/5) {
  nzv_idx <- nearZeroVar(X, freqCut = freqCut, saveMetrics = FALSE)
  if (length(nzv_idx)) X <- X[, -nzv_idx, drop = FALSE]
  X
}

cor_drop <- function(X, cutoff = 0.90) {
  if (ncol(X) < 2) return(list(X = X, removed = character()))
  cmat <- suppressWarnings(cor(X, use = "pairwise.complete.obs"))
  bad <- findCorrelation(cmat, cutoff = cutoff, names = TRUE, exact = TRUE)
  if (length(bad)) X <- X[, setdiff(colnames(X), bad), drop = FALSE]
  list(X = X, removed = bad)
}

cor_heatmap <- function(X, fname="03_corr_heatmap.png") {
  if (ncol(X) < 2) return(invisible(NULL))
  C <- suppressWarnings(cor(X, use="pairwise.complete.obs"))
  # C[upper.tri(C, diag=TRUE)] <- NA
  d <- reshape2::melt(C, na.rm = TRUE, varnames=c("Var1","Var2"), value.name="r")
  p <- ggplot(d, aes(Var1, Var2, fill = r)) +
    geom_raster() +
    scale_fill_gradient2(limits=c(-1,1), low="blue", mid="white", high="red", midpoint=0) +
    theme_minimal(base_size = 10) + theme(axis.text.x=element_text(angle=90,vjust=0.5,hjust=1)) +
    labs(title="Correlation heatmap", x=NULL, y=NULL, fill="r")
  print(p); gsave(fname, p, 9, 7)
}

scree_plot <- function(pc, fname="06_pca_scree.png") {
  var <- pc$sdev^2
  pvar <- var/sum(var)
  df <- tibble(PC = seq_along(pvar), Prop = pvar, Cum = cumsum(pvar))
  p <- ggplot(df, aes(PC, Prop)) + geom_point() + geom_line() + 
    geom_line(aes(y=Cum), linetype="dashed") +
    labs(title="PCA variance (points) & cumulative (dashed)", x="Component", y="Variance proportion") +
    theme_minimal(base_size = 13)
  print(p); gsave(fname, p, 7, 5)
  invisible(df)
}

class_panels <- function(train_df, up_df, down_df, response) {
  z <- bind_rows(
    train_df %>% count(!!rlang::sym(response), name="Count") %>% mutate(Set="Train"),
    up_df    %>% count(!!rlang::sym(response), name="Count") %>% mutate(Set="Upsampled"),
    down_df  %>% count(!!rlang::sym(response), name="Count") %>% mutate(Set="Downsampled")
  ) %>% group_by(Set) %>% mutate(Perc = round(100*Count/sum(Count),1)) %>% ungroup()
  p <- ggplot(z, aes(!!rlang::sym(response), Count, fill=!!rlang::sym(response))) +
    geom_col(width=.65, show.legend = FALSE) +
    geom_text(aes(label=paste0(Count," (",Perc,"%)")), vjust=-0.4, size=3.6) +
    facet_wrap(~Set, ncol = 1, scales = "free_y") +
    labs(title="Class distribution: original vs resampled", x=NULL, y="Count") +
    theme_minimal(base_size = 12)
  print(p); gsave("07_resampling_panels.png", p, 7, 9)
}

skewness_plot <- function(sk_df, fname = "09_skewness_distribution.png") {
  sk_long <- sk_df %>%
    select(Before, After) %>%
    tidyr::pivot_longer(cols = everything(), names_to = "State", values_to = "Skewness")
  
  p <- ggplot(sk_long, aes(x = Skewness, fill = State)) +
    geom_density(alpha = 0.6, na.rm = TRUE, adjust = 1.5) +
    scale_fill_manual(values = c("Before" = "#FF6347", "After" = "#4682B4")) +
    labs(
      title = "Distribution of Skewness Before and After Box-Cox",
      subtitle = "Applied to numeric predictors. 'After' values are much closer to zero.",
      x = "Skewness Value", y = "Density", fill = "Transformation State"
    ) +
    theme_minimal(base_size = 13)
  print(p)
  gsave(fname, p, 9, 6)
  invisible(p)
}

outlier_plot <- function(df_before, df_after, fname = "10_outliers_before_after.png") {
  kept_cols <- intersect(colnames(df_before), colnames(df_after))
  if (length(kept_cols) == 0) return(invisible(NULL))
  
  df_before <- df_before[, kept_cols, drop = FALSE]
  df_after <- df_after[, kept_cols, drop = FALSE]
  
  long_before <- df_before %>%
    tidyr::pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
    mutate(State = "Before")
  
  long_after <- df_after %>%
    as.data.frame() %>% 
    tidyr::pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
    mutate(State = "After")
  
  combined <- bind_rows(long_before, long_after) %>%
    mutate(State = factor(State, levels = c("Before", "After")))
  
  p <- ggplot(combined, aes(x = State, y = Value, fill = State)) +
    geom_boxplot(show.legend = FALSE, outlier.size = 1) +
    facet_wrap(~Variable, scales = "free_y") +
    scale_fill_manual(values = c("Before" = "#FF6347", "After" = "#4682B4")) +
    labs(
      title = "Outlier Comparison Before and After Transformation",
      subtitle = "Boxplots per variable. 'After' is Box-Cox transformed, centered, and scaled.",
      x = NULL, y = "Value (Variable-specific Scale)"
    ) +
    theme_minimal(base_size = 15) +
    theme(strip.text = element_text(size = 8))
  
  print(p)
  gsave(fname, p, width = 12, height = 9, dpi = 300)
  invisible(p)
}

skewness_hist_plots <- function(df_before, df_after, fname = "11_skewness_histograms.png") {
  kept_cols <- intersect(colnames(df_before), colnames(df_after))
  if (length(kept_cols) == 0) return(invisible(NULL))
  
  df_before <- df_before[, kept_cols, drop = FALSE]
  df_after <- df_after[, kept_cols, drop = FALSE]
  
  long_before <- df_before %>%
    tidyr::pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
    mutate(State = "Before")
  
  long_after <- df_after %>%
    as.data.frame() %>%
    tidyr::pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
    mutate(State = "After")
  
  combined <- bind_rows(long_before, long_after) %>%
    mutate(State = factor(State, levels = c("Before", "After")))
  
  max_facets <- 6
  var_chunks <- split(kept_cols, ceiling(seq_along(kept_cols) / max_facets))
  
  for (i in seq_along(var_chunks)) {
    chunk_vars <- var_chunks[[i]]
    plot_data <- filter(combined, Variable %in% chunk_vars)
    
    p <- ggplot(plot_data, aes(x = Value, fill = State)) +
      geom_histogram(bins = 30, show.legend = FALSE) +
      facet_grid(Variable ~ State, scales = "free") +
      scale_fill_manual(values = c("Before" = "#FF6347", "After" = "#4682B4")) +
      labs(
        title = paste("Distribution of Numeric Predictors (Part", i, ")"),
        subtitle = "before vs. after.",
        x = "Value (Scales are independent for each plot)",
        y = "Frequency"
      ) +
      theme_minimal(base_size = 20) +
      theme(strip.text.y = element_text(angle = 0, hjust = 1))
    
    if (length(var_chunks) > 1) {
      part_fname <- stringr::str_replace(fname, "\\.png$", paste0("_part", i, ".png"))
    } else {
      part_fname <- fname
    }
    
    print(p)
    gsave(part_fname, p, width = 8, height = max(8, 1.5 * length(chunk_vars)), dpi = 150)
  }
  invisible(NULL)
}

outlier_count_plot <- function(df_before, df_after, fname = "13_outlier_counts.png") {
  kept_cols <- intersect(colnames(df_before), colnames(df_after))
  if (length(kept_cols) == 0) return(invisible(NULL))
  
  count_outliers <- function(x) {
    if (!is.numeric(x)) return(NA_integer_)
    q1 <- quantile(x, 0.25, na.rm = TRUE)
    q3 <- quantile(x, 0.75, na.rm = TRUE)
    iqr <- q3 - q1
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
  }
  
  before_counts <- sapply(df_before[, kept_cols, drop = FALSE], count_outliers)
  after_counts <- sapply(as.data.frame(df_after)[, kept_cols, drop = FALSE], count_outliers)
  
  count_df <- tibble(
    Variable = rep(kept_cols, 2),
    Count = c(before_counts, after_counts),
    State = rep(c("Before", "After"), each = length(kept_cols))
  ) %>%
    filter(!is.na(Count) & Count > 0)
  
  if (nrow(count_df) == 0) {
    cat("No outliers detected in any variables.\n")
    return(invisible(NULL))
  }
  
  var_order <- count_df %>% 
    filter(State == "Before") %>% 
    arrange(desc(Count)) %>% 
    pull(Variable)
  
  count_df$Variable <- factor(count_df$Variable, levels = var_order)
  count_df$State <- factor(count_df$State, levels = c("Before", "After"))
  
  p <- ggplot(count_df, aes(x = Variable, y = Count, fill = State)) +
    geom_col(position = "dodge") +
    geom_text(aes(label = Count), position = position_dodge(width = 0.9), vjust = -0.3, size = 3) +
    scale_fill_manual(values = c("Before" = "#FF6347", "After" = "#4682B4")) +
    labs(
      title = "Number of Outliers Before and After Transformation",
      subtitle = "Outliers defined by the 1.5 * IQR rule. Variables with no outliers are hidden.",
      x = NULL, y = "Outlier Count"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
  gsave(fname, p, width = 10, height = 7)
  invisible(p)
}

ohe_prepare <- function(train_cat_df, test_cat_df, out_dir, report = "03_removed_single_level_cats.csv") {
  if (!ncol(train_cat_df)) {
    return(list(train = data.frame(), test = data.frame(), removed = character()))
  }
  
  norm_chars <- function(df) {
    df <- mutate(df, across(everything(), ~ as.character(.)))
    df <- mutate(df, across(everything(), ~ stringr::str_squish(.)))
    df[] <- lapply(df, function(x) { x[is.na(x) | x == ""] <- "Unknown"; x })
    df
  }
  train_cat_df <- norm_chars(train_cat_df)
  test_cat_df  <- norm_chars(test_cat_df)
  
  keep <- vapply(train_cat_df, function(x) dplyr::n_distinct(x) > 1, logical(1))
  removed <- names(train_cat_df)[!keep]
  train_cat_df <- train_cat_df[, keep, drop = FALSE]
  test_cat_df  <- test_cat_df[,  keep, drop = FALSE]
  if (length(removed)) {
    write.csv(data.frame(removed_single_level = removed),
              file.path(out_dir, report), row.names = FALSE)
  }
  if (!ncol(train_cat_df)) return(list(train = data.frame(), test = data.frame(), removed = removed))
  
  for (nm in names(train_cat_df)) {
    tr <- train_cat_df[[nm]]
    te <- test_cat_df[[nm]]
    tr_levs <- sort(unique(tr))
    all_levs <- unique(c(tr_levs, "Other"))
    te2 <- ifelse(te %in% tr_levs, te, "Other")
    train_cat_df[[nm]] <- factor(tr, levels = all_levs)
    test_cat_df[[nm]]  <- factor(te2, levels = all_levs)
  }
  
  form <- as.formula(paste("~", paste(sprintf("`%s`", names(train_cat_df)), collapse = " + "), "- 1"))
  X_train <- as.data.frame(model.matrix(form, data = train_cat_df))
  X_test  <- as.data.frame(model.matrix(form, data = test_cat_df))
  list(train = X_train, test = X_test, removed = removed)
}

# ------------------------------------------------------------------------------

data_raw <- read.csv(CFG$data_path, stringsAsFactors = FALSE)
data <- replace_q_na(data_raw)

if (CFG$response %in% names(data)) {
  data[[CFG$response]] <- as.character(data[[CFG$response]])
  
  # Rename explicit levels
  data[[CFG$response]][data[[CFG$response]] == "<30"] <- "Lt30"   # "Less than 30"
  data[[CFG$response]][data[[CFG$response]] == ">30"] <- "Gt30"   # "Greater than 30"
  data[[CFG$response]][data[[CFG$response]] == "NO"]  <- "No"     # "No Readmission"
  
  # Convert back to factor
  data[[CFG$response]] <- as.factor(data[[CFG$response]])
}

goal <- "Predict patient readmission"
sample_size <- nrow(data)
response <- CFG$response
if (!response %in% names(data)) stop("Response not found.")
data[[response]] <- as.factor(data[[response]])
types <- split_types(data, response)
cat("Goal:", goal, "\n")
cat("Sample size:", sample_size, "\n")
cat("Response variable:", response, "\n")
cat("# predictors:", ncol(data)-1, "\n")
cat("# categorical predictors:", length(types$cat), "\n")
cat("# numeric predictors:", length(types$num), "\n")

plot_response(data, response)
miss_tbl <- missing_plot(data)
write.csv(miss_tbl, file.path(CFG$out_dir, "00_missing_table.csv"), row.names = FALSE)

data <- data %>% mutate(across(all_of(types$cat), ~ ifelse(is.na(.), "Unknown", as.character(.))))

idx <- caret::createDataPartition(y = data[[response]], p = CFG$train_prop, list = FALSE)
train <- data[idx, , drop = FALSE]
test  <- data[-idx, , drop = FALSE]
cat("Train rows:", nrow(train), " Test rows:", nrow(test), "\n")

train_x_cat <- train[, types$cat, drop = FALSE]
test_x_cat  <- test[,  types$cat, drop = FALSE]
train_x_num <- train[, types$num, drop = FALSE]
test_x_num  <- test[,  types$num, drop = FALSE]

ohe <- ohe_prepare(train_x_cat, test_x_cat, out_dir = CFG$out_dir)
train_cat_ohe <- ohe$train
test_cat_ohe  <- ohe$test

train_num <- train_x_num
test_num  <- test_x_num
X_train_all <- bind_cols(train_num, train_cat_ohe)
X_test_all  <- bind_cols(test_num,  test_cat_ohe)

X_train_all <- nzv_drop(X_train_all, CFG$nzv_freq_cut)
X_test_all  <- X_test_all[, colnames(X_train_all), drop = FALSE]

if (anyNA(X_train_all)) {
  imputer <- caret::preProcess(X_train_all, method = "knnImpute", k = CFG$k_knn)
  X_train_all <- predict(imputer, X_train_all)
  X_test_all  <- predict(imputer, X_test_all)
}

shift_vec <- vapply(X_train_all, function(x) ifelse(min(x, na.rm=TRUE) <= 0, abs(min(x, na.rm=TRUE)) + 1e-6, 0), numeric(1))
X_train_pos <- sweep(X_train_all, 2, shift_vec, `+`)
X_test_pos  <- sweep(X_test_all,  2, shift_vec, `+`)

pp_box <- caret::preProcess(X_train_pos, method = c("BoxCox","center","scale"))
X_train_bc <- predict(pp_box, X_train_pos)
X_test_bc  <- predict(pp_box, X_test_pos)

# --- MODIFIED: Generate heatmaps before and after pruning ---
# 1. Plot heatmap BEFORE removing highly correlated predictors
cor_heatmap(X_train_bc, "03_corr_heatmap_before_removal.png")

# 2. Identify and remove predictors with correlation > cutoff
cor_pruned <- cor_drop(X_train_bc, cutoff = CFG$cor_cutoff)
X_train_bc <- cor_pruned$X
X_test_bc  <- X_test_bc[, colnames(X_train_bc), drop = FALSE]
write.csv(data.frame(removed_high_corr = cor_pruned$removed), file.path(CFG$out_dir,"04_removed_high_corr.csv"), row.names = FALSE)

# 3. Plot heatmap AFTER removal for comparison (if any were removed)
if (length(cor_pruned$removed) > 0) {
  cor_heatmap(X_train_bc, "03a_corr_heatmap_after_removal.png")
}
# -------------------------------------------------------------

pp_ss <- caret::preProcess(X_train_bc, method = "spatialSign")
X_train_ss <- predict(pp_ss, X_train_bc)
X_test_ss  <- predict(pp_ss,  X_test_bc)

pc_fit <- prcomp(X_train_bc, center = FALSE, scale. = FALSE)
scree <- scree_plot(pc_fit, "06_pca_scree.png")
cumvar <- cumsum(pc_fit$sdev^2)/sum(pc_fit$sdev^2)
k_pc <- which(cumvar >= CFG$pca_var_thresh)[1]
if (is.na(k_pc)) k_pc <- min(1, ncol(pc_fit$x))
X_train_pc <- pc_fit$x[, 1:k_pc, drop = FALSE]
X_test_pc  <- scale(X_test_bc, center = pc_fit$center, scale = pc_fit$scale) %*% pc_fit$rotation[, 1:k_pc, drop = FALSE]

cor_heatmap(as.data.frame(X_train_pc), fname = "12_corr_heatmap_pca.png")

train_base <- bind_cols(as.data.frame(X_train_bc), tibble(!!response := train[[response]]))
test_base  <- bind_cols(as.data.frame(X_test_bc),  tibble(!!response := test[[response]]))
train_spatial <- bind_cols(as.data.frame(X_train_ss), tibble(!!response := train[[response]]))
test_spatial  <- bind_cols(as.data.frame(X_test_ss),  tibble(!!response := test[[response]]))
train_pca <- bind_cols(as.data.frame(X_train_pc), tibble(!!response := train[[response]]))
test_pca  <- bind_cols(as.data.frame(X_test_pc),  tibble(!!response := test[[response]]))

imb_tbl <- train %>% count(!!rlang::sym(response), name="n") %>% mutate(p = n/sum(n))
write.csv(imb_tbl, file.path(CFG$out_dir,"05_class_imbalance_train.csv"), row.names = FALSE)

up <- upSample(x = train_base %>% dplyr::select(-all_of(response)), y = train_base[[response]], yname = response)
down <- downSample(x = train_base %>% dplyr::select(-all_of(response)), y = train_base[[response]], yname = response)
class_panels(train_base, up, down, response)

write.csv(train_base,   file.path(CFG$out_dir, "train_processed_boxcox_cs_corr.csv"), row.names = FALSE)
write.csv(test_base,    file.path(CFG$out_dir, "test_processed_boxcox_cs_corr.csv"),  row.names = FALSE)
write.csv(train_spatial,file.path(CFG$out_dir, "train_processed_boxcox_cs_corr_spatial.csv"), row.names = FALSE)
write.csv(test_spatial, file.path(CFG$out_dir, "test_processed_boxcox_cs_corr_spatial.csv"),  row.names = FALSE)
write.csv(train_pca,    file.path(CFG$out_dir, "train_processed_boxcox_cs_corr_pca.csv"), row.names = FALSE)
write.csv(test_pca,     file.path(CFG$out_dir, "test_processed_boxcox_cs_corr_pca.csv"),  row.names = FALSE)

sk_before <- if (length(types$num)) sapply(train_x_num, function(x) e1071::skewness(x, na.rm=TRUE)) else numeric(0)
sk_after  <- if (length(types$num)) {
  kept_num <- colnames(train_x_num)[colnames(train_x_num) %in% colnames(X_train_bc)]
  sapply(as.data.frame(X_train_bc)[, kept_num, drop = FALSE], function(x) e1071::skewness(x, na.rm=TRUE))
} else numeric(0)
sk_tab <- tibble(Variable = names(sk_before), Before = as.numeric(sk_before)) %>%
  left_join(tibble(Variable = names(sk_after), After = as.numeric(sk_after)), by="Variable") %>%
  mutate(Delta = After - Before) %>% arrange(desc(abs(Before)))
write.csv(sk_tab, file.path(CFG$out_dir,"08_skewness_before_after_boxcox.csv"), row.names = FALSE)

# --- CALLS TO PLOTTING FUNCTIONS ---
skewness_plot(sk_tab)
outlier_plot(train_x_num, X_train_bc)
skewness_hist_plots(train_x_num, X_train_bc)
outlier_count_plot(train_x_num, X_train_bc)

# Decisions summary
dec_lines <- c(
  paste("Goal:", goal),
  paste("Sample size:", sample_size),
  paste("Response:", response),
  paste("# predictors:", ncol(data)-1),
  paste("# categorical predictors:", length(types$cat)),
  paste("# numeric predictors:", length(types$num)),
  paste("Train/Test split:", paste0(round(CFG$train_prop*100), "% / ", round((1-CFG$train_prop)*100), "%")),
  paste("Imputation:", paste0("kNN (k=", CFG$k_knn, ") on numeric after one-hot")),
  paste("Dummy variables:", "robust OHE via model.matrix with 'Other' mapping; single-level train cats dropped"),
  paste("Near-zero variance removal:", "applied"),
  paste("Box-Cox:", "applied after shifting variables to positive; with centering/scaling"),
  paste("High correlation removal:", paste0("cutoff |r|>", CFG$cor_cutoff, "; removed ", length(cor_pruned$removed), " predictors")),
  paste("PCA:", paste0(k_pc, " components for ≥", round(CFG$pca_var_thresh*100), "% variance")),
  paste("Spatial sign:", "alternate train/test set produced"),
  paste("Class imbalance (train):", paste(imb_tbl[[response]], collapse="/"), "counts =", paste(imb_tbl$n, collapse="/")),
  paste("Resampling sets:", "upsampled and downsampled versions created for model comparison")
)
writeLines(dec_lines, con = file.path(CFG$out_dir, "09_decisions_summary.txt"))

# Rmd deck
rmd <- c(
  "---",
  "title: \"Preprocessing Presentation\"",
  "subtitle: \"Diabetes Readmission\"",
  "author: \"Group\"",
  "output: beamer_presentation",
  "fontsize: 11pt",
  "---",
  "",
  "## Goal",
  "Predict patient readmission using EHR predictors.",
  "",
  "## Data Structure",
  paste0("- Sample size: ", sample_size),
  paste0("- Response: ", response),
  paste0("- Predictors: ", ncol(data)-1, " (", length(types$cat), " categorical, ", length(types$num), " numeric)"),
  "",
  "## Class Distribution",
  "![](01_class_distribution.png){height=0.6\\textheight}",
  "",
  "## Missingness",
  "![](02_missingness.png){height=0.6\\textheight}",
  "",
  "## One-Hot + Impute + Box-Cox + Center/Scale",
  "- Robust OHE (single-level train cats dropped; unseen test levels → Other)",
  paste0("- kNN imputation (k=", CFG$k_knn, ")"),
  "- Box-Cox (shifted to positive), then center/scale",
  "",
  "## Correlations",
  "![](03_corr_heatmap_before_removal.png){height=0.6\\textheight}",
  paste0("- High-correlation removal at |r|>", CFG$cor_cutoff),
  "",
  "## Skewness Reduction",
  "![](09_skewness_distribution.png){height=0.6\\textheight}",
  "- Box-Cox transform significantly reduced skewness in numeric predictors.",
  "",
  "## Outlier Mitigation (Boxplots)",
  "![](10_outliers_before_after.png){height=0.6\\textheight}",
  "- Transformations also help in managing the influence of outliers.",
  "",
  "## Outlier Reduction (Counts)",
  "![](13_outlier_counts.png){height=0.6\\textheight}",
  "- The number of outliers is significantly reduced after transformation.",
  "",
  "## PCA",
  "![](06_pca_scree.png){height=0.55\\textheight}",
  paste0("- Retained ", k_pc, " PCs (≥", round(CFG$pca_var_thresh*100), "% variance)"),
  "",
  "## PCA Correlation Check",
  "![](12_corr_heatmap_pca.png){height=0.6\\textheight}",
  "- Heatmap confirms PCA components are uncorrelated.",
  "",
  "## Spatial Sign",
  "- Alternate set to mitigate outliers",
  "",
  "## Train/Test & Resampling",
  paste0("- Split: ", round(CFG$train_prop*100), "% train / ", round((1-CFG$train_prop)*100), "% test (stratified)"),
  "![](07_resampling_panels.png){height=0.6\\textheight}",
  "- Upsampled and downsampled sets prepared",
  "",
  "## What We'll Use For Modeling",
  "- Start with Box-Cox + center/scale + correlation pruning",
  paste0("- Compare: with/without PCA (", k_pc, " PCs)"),
  "- Evaluate with original vs upsampled training",
  "- Keep spatial-sign variant for robustness checks"
)
writeLines(rmd, con = file.path(CFG$out_dir, "presentation.Rmd"))

cat("Files written to:", normalizePath(CFG$out_dir), "\n")







#  MODEL TRAINING FOR 12 CLASSIFIERS ORIGINAL SAMPLE SIZE
#  Uses caret::train(), Kappa, Confusion Matrix, and Test Set

set.seed(123)

#============================================================
# Choose which processed dataset to model
# Recommended: Box-Cox + Center/Scale + Correlation-Pruned
#============================================================

# ============================================================
# FIX CLASS LABELS (IMPORTANT FOR multinom AND OTHER MODELS)
# ============================================================

# Extract original levels
original_levels <- levels(train_base[[response]])
cat("Original response levels:\n")
print(original_levels)

# Create safe R variable names
safe_levels <- make.names(original_levels)

cat("Renamed (safe) levels:\n")
print(safe_levels)

# Apply renaming to ALL processed training/test sets
train_base[[response]]    <- factor(train_base[[response]],    levels = original_levels, labels = safe_levels)
test_base[[response]]     <- factor(test_base[[response]],     levels = original_levels, labels = safe_levels)

train_spatial[[response]] <- factor(train_spatial[[response]], levels = original_levels, labels = safe_levels)
test_spatial[[response]]  <- factor(test_spatial[[response]],  levels = original_levels, labels = safe_levels)

train_pca[[response]]     <- factor(train_pca[[response]],     levels = original_levels, labels = safe_levels)
test_pca[[response]]      <- factor(test_pca[[response]],      levels = original_levels, labels = safe_levels)


train_set <- train_base
test_set  <- test_base

response <- CFG$response

x_train <- dplyr::select(train_set, -!!sym(response))
x_test  <- dplyr::select(test_set,  -!!sym(response))

y_train <- train_set[[response]]
y_test  <- test_set[[response]]

# Common Training Control (K-fold CV)
library(doParallel)

# 1. Register Parallel Backend. This is makes it run the NN model faster
num_cores <- parallel::detectCores() - 1 
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# 2. Update Control for Monitoring
ctrl <- trainControl(
  method = "cv",
  number = 10,                 
  summaryFunction = multiClassSummary,
  classProbs = TRUE,
  allowParallel = TRUE,        
  verboseIter = TRUE           
)

# Storage structure
model_list <- list()
results_list <- list()
conf_list <- list()

add_model <- function(name, model, prediction) {
  model_list[[name]] <<- model
  conf_list[[name]] <<- confusionMatrix(prediction, y_test)
  results_list[[name]] <<- data.frame(
    Model = name,
    Kappa = conf_list[[name]]$overall["Kappa"],
    Accuracy = conf_list[[name]]$overall["Accuracy"]
  )
}

# LINEAR MODELS


# 1. Logistic Regression ("multinom" for multi-class)
set.seed(123)
m_log <- train(
  x_train, y_train,
  method = "multinom",
  trControl = ctrl
)
p_log <- predict(m_log, x_test)
add_model("Logistic Regression", m_log, p_log)



# 2. Linear Discriminant Analysis
set.seed(123)
m_lda <- train(
  x_train, y_train,
  method = "lda",
  trControl = ctrl
)
p_lda <- predict(m_lda, x_test)
add_model("LDA", m_lda, p_lda)


# 3. PLS-DA  ("pls" chooses PLSDA when outcome is factor)
set.seed(123)
m_pls <- train(
  x_train, y_train,
  method = "pls",
  tuneLength = 15,
  trControl = ctrl
)
p_pls <- predict(m_pls, x_test)
add_model("PLS-DA", m_pls, p_pls)


# 4. Lasso (glmnet α=1)
set.seed(123)
m_lasso <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl
)
p_lasso <- predict(m_lasso, x_test)
add_model("Lasso", m_lasso, p_lasso)


# 5. Ridge (glmnet α=0)
set.seed(123)
m_ridge <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl
)
p_ridge <- predict(m_ridge, x_test)
add_model("Ridge", m_ridge, p_ridge)



# 6. Elastic Net (α ∈ [0,1])
set.seed(123)
m_en <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = seq(0, 1, length = 5),
    lambda = seq(0.0001, 1, length = 20)
  ),
  trControl = ctrl
)
p_en <- predict(m_en, x_test)
add_model("Elastic Net", m_en, p_en)



# NON-LINEAR MODELS 
# 7. Nonlinear Discriminant Analysis (method = "mda")
# Fixed: changed 'nda' to 'mda' (Mixture Discriminant Analysis)
set.seed(123)
m_nda <- train(
  x_train, y_train,
  method = "mda",
  trControl = ctrl
)
p_nda <- predict(m_nda, x_test)
add_model("NDA", m_nda, p_nda)



# 8. Neural Network (nnet)
set.seed(123)
m_nnet <- train(
  x_train, y_train,
  method = "nnet",
  trace = FALSE,
  tuneLength = 3,
  trControl = ctrl
)
p_nnet <- predict(m_nnet, x_test)
add_model("Neural Network", m_nnet, p_nnet)


# 9. Flexible Discriminant Analysis
set.seed(123)
m_fda <- train(
  x_train, y_train,
  method = "fda",
  tuneLength = 10,
  trControl = ctrl
)
p_fda <- predict(m_fda, x_test)
add_model("FDA", m_fda, p_fda)


# 10. Support Vector Machine (RBF)
set.seed(123)
m_svm <- train(
  x_train, y_train,
  method = "svmRadial",
  tuneLength = 3,
  trControl = ctrl
)
p_svm <- predict(m_svm, x_test)
add_model("SVM (RBF)", m_svm, p_svm)


# 11. KNN classifier
set.seed(123)
m_knn <- train(
  x_train, y_train,
  method = "knn",
  tuneLength = 15,
  trControl = ctrl
)
p_knn <- predict(m_knn, x_test)
add_model("KNN", m_knn, p_knn)


# 12. Naive Bayes
set.seed(123)
m_nb <- train(
  x_train, y_train,
  method = "naive_bayes",
  trControl = ctrl
)
p_nb <- predict(m_nb, x_test)
add_model("Naive Bayes", m_nb, p_nb)



# Generating Table Results 
results_all <- do.call(rbind, results_list)
results_all <- results_all[order(-results_all$Kappa), ]

print(results_all)

write.csv(results_all, file.path("Final_Project_Phase2_Results/model_kappa_results_original.csv"), row.names = FALSE)



counts_list <- list()
stats_list  <- list()

for(name in names(conf_list)) {
  cm <- conf_list[[name]]
  
  # --- Part A: Raw Counts (Prediction vs Reference) ---
  # Converts the matrix to a "Long" format: Model | Prediction | Reference | Freq
  df_counts <- as.data.frame(cm$table)
  df_counts$Model <- name
  counts_list[[name]] <- df_counts
  
  # --- Part B: Detailed Class Stats (Sensitivity, Specificity, etc.) ---
  # If there are 3 classes, this table will have 3 rows per model
  df_stats <- as.data.frame(cm$byClass)
  df_stats$Class <- rownames(df_stats) # Capture the class names
  df_stats$Model <- name
  
  # Reorder columns so Model and Class are first
  df_stats <- df_stats %>% dplyr::select(Model, Class, everything())
  stats_list[[name]] <- df_stats
}

# Combine all models into single dataframes
all_counts_df <- do.call(rbind, counts_list)
all_stats_df  <- do.call(rbind, stats_list)

# Write to CSV
write.csv(all_counts_df, file.path("Final_Project_Phase2_Results/model_confusion_matrix_counts_original.csv"), row.names = FALSE)
write.csv(all_stats_df,  file.path("Final_Project_Phase2_Results/model_detailed_class_metrics_original.csv"),  row.names = FALSE)






print(m_log)
print(m_lda)
print(m_pls)
print(m_lasso)
print(m_ridge)
print(m_en)
print(m_nda)
print(m_nnet)
print(m_fda)
print(m_svm)
print(m_knn)
print(m_nb)

plot(m_log)
plot(m_lda)
plot(m_pls)
plot(m_lasso)
plot(m_ridge)
plot(m_en)
plot(m_nda)
plot(m_nnet)
plot(m_fda)
plot(m_svm)
plot(m_knn)
plot(m_nb)


confusionMatrix(m_fda)

confusionMatrix(m_nda)

pred_fda <- predict(m_fda, newdata = x_test)
confusionMatrix(pred_fda, y_test)

pred_fda <- predict(m_nda, newdata = x_test)
confusionMatrix(pred_fda, y_test)

imp_fda <- varImp(m_fda)
top10_fda <- head(imp_fda$importance[order(-imp_fda$importance$Overall), , drop = FALSE], 15)
top10_fda



imp_fda <- varImp(m_nda)
top10_fda <- head(imp_fda$importance[order(-imp_fda$importance$Overall), , drop = FALSE], 15)
top10_fda



imp_nda <- varImp(m_nda)$importance

# Average importance across classes (for multi-class)
imp_nda$Mean <- rowMeans(imp_nda)

# Sort by the Mean column
top10_nda <- head(imp_nda[order(-imp_nda$Mean), , drop = FALSE], 10)

top10_nda











get_top10 <- function(model) {
  imp <- varImp(model)$importance
  
  # If the model returned one column (e.g., "Overall")
  if (ncol(imp) == 1) {
    imp_sorted <- imp[order(-imp[,1]), , drop = FALSE]
  } else {
    # Multi-class: mean across columns
    imp$Mean <- rowMeans(imp)
    imp_sorted <- imp[order(-imp$Mean), , drop = FALSE]
  }
  
  head(imp_sorted, 10)
}



get_top10(m_fda)
get_top10(m_nda)







library(ggplot2)
library(caret) # Ensure caret is loaded for varImp

# Function to get top n predictors and prepare for plotting
get_top_n_data <- function(model, n = 5) {
  imp <- varImp(model)$importance
  
  # Handle different importance structures
  if (ncol(imp) == 1) {
    # Rename column to "Importance" for consistency
    names(imp) <- "Importance"
    imp_sorted <- imp[order(-imp$Importance), , drop = FALSE]
  } else {
    # Multi-class: calculate mean across classes
    imp$Importance <- rowMeans(imp)
    imp_sorted <- imp[order(-imp$Importance), , drop = FALSE]
  }
  
  # Slice top n
  top_n <- head(imp_sorted, n)
  
  # Convert row names to a proper column for ggplot
  top_n$Feature <- rownames(top_n)
  
  return(top_n)
}

# Function to plot the data
plot_top_predictors <- function(model, model_name) {
  # Get data
  plot_data <- get_top_n_data(model, n = 5)
  
  # Plot
  ggplot(plot_data, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +  # Flip coordinates for readable labels
    theme_minimal() +
    labs(title = paste("Top 5 Predictors:", model_name),
         x = "Predictor",
         y = "Importance") +
    theme(axis.text.y = element_text(size = 11))
}

# --- Usage ---

# 1. Plot for FDA model
plot_top_predictors(m_fda, "FDA Model")

# 2. Plot for NDA model
plot_top_predictors(m_nda, "NDA Model")
























































#  MODEL TRAINING FOR 12 CLASSIFIERS UP SAMPLE SIZE
#  Uses caret::train(), Kappa, Confusion Matrix, and Test Set

set.seed(123)

#============================================================
# Choose which processed dataset to model
# Recommended: Box-Cox + Center/Scale + Correlation-Pruned
#============================================================

# ============================================================
# FIX CLASS LABELS (IMPORTANT FOR multinom AND OTHER MODELS)
# ============================================================

# Extract original levels
original_levels <- levels(train_base[[response]])
cat("Original response levels:\n")
print(original_levels)

# Create safe R variable names
safe_levels <- make.names(original_levels)

cat("Renamed (safe) levels:\n")
print(safe_levels)

# Apply renaming to ALL processed training/test sets
train_base[[response]]    <- factor(train_base[[response]],    levels = original_levels, labels = safe_levels)
test_base[[response]]     <- factor(test_base[[response]],     levels = original_levels, labels = safe_levels)

train_spatial[[response]] <- factor(train_spatial[[response]], levels = original_levels, labels = safe_levels)
test_spatial[[response]]  <- factor(test_spatial[[response]],  levels = original_levels, labels = safe_levels)

train_pca[[response]]     <- factor(train_pca[[response]],     levels = original_levels, labels = safe_levels)
test_pca[[response]]      <- factor(test_pca[[response]],      levels = original_levels, labels = safe_levels)


train_set <- train_base
test_set  <- test_base

response <- CFG$response

x_train <- dplyr::select(train_set, -!!sym(response))
x_test  <- dplyr::select(test_set,  -!!sym(response))

y_train <- train_set[[response]]
y_test  <- test_set[[response]]

# Common Training Control (K-fold CV)
library(doParallel)

# 1. Register Parallel Backend. This is makes it run the NN model faster
num_cores <- parallel::detectCores() - 1 
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# 2. Update Control for Monitoring
ctrl <- trainControl(
  method = "cv",
  number = 10,                 
  summaryFunction = multiClassSummary,
  classProbs = TRUE,
  allowParallel = TRUE,        
  verboseIter = TRUE,
  sampling = "up"
)

# Storage structure
model_list <- list()
results_list <- list()
conf_list <- list()

add_model <- function(name, model, prediction) {
  model_list[[name]] <<- model
  conf_list[[name]] <<- confusionMatrix(prediction, y_test)
  results_list[[name]] <<- data.frame(
    Model = name,
    Kappa = conf_list[[name]]$overall["Kappa"],
    Accuracy = conf_list[[name]]$overall["Accuracy"]
  )
}

# LINEAR MODELS


# 1. Logistic Regression ("multinom" for multi-class)
set.seed(123)
m_log <- train(
  x_train, y_train,
  method = "multinom",
  trControl = ctrl
)
p_log <- predict(m_log, x_test)
add_model("Logistic Regression", m_log, p_log)

print(m_log)

# 2. Linear Discriminant Analysis
set.seed(123)
m_lda <- train(
  x_train, y_train,
  method = "lda",
  trControl = ctrl
)
p_lda <- predict(m_lda, x_test)
add_model("LDA", m_lda, p_lda)


# 3. PLS-DA  ("pls" chooses PLSDA when outcome is factor)
set.seed(123)
m_pls <- train(
  x_train, y_train,
  method = "pls",
  tuneLength = 15,
  trControl = ctrl
)
p_pls <- predict(m_pls, x_test)
add_model("PLS-DA", m_pls, p_pls)


# 4. Lasso (glmnet α=1)
set.seed(123)
m_lasso <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl
)
p_lasso <- predict(m_lasso, x_test)
add_model("Lasso", m_lasso, p_lasso)


# 5. Ridge (glmnet α=0)
set.seed(123)
m_ridge <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl
)
p_ridge <- predict(m_ridge, x_test)
add_model("Ridge", m_ridge, p_ridge)



# 6. Elastic Net (α ∈ [0,1])
set.seed(123)
m_en <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = seq(0, 1, length = 5),
    lambda = seq(0.0001, 1, length = 20)
  ),
  trControl = ctrl
)
p_en <- predict(m_en, x_test)
add_model("Elastic Net", m_en, p_en)



# NON-LINEAR MODELS 
# 7. Nonlinear Discriminant Analysis (method = "mda")
# Fixed: changed 'nda' to 'mda' (Mixture Discriminant Analysis)
set.seed(123)
m_nda <- train(
  x_train, y_train,
  method = "mda",
  trControl = ctrl
)
p_nda <- predict(m_nda, x_test)
add_model("NDA", m_nda, p_nda)



# 8. Neural Network (nnet)
#set.seed(123)
#m_nnet <- train(
#  x_train, y_train,
#  method = "nnet",
#  trace = FALSE,
#  tuneLength = 3,
#  trControl = ctrl
#)
#p_nnet <- predict(m_nnet, x_test)
#add_model("Neural Network", m_nnet, p_nnet)


# 9. Flexible Discriminant Analysis
set.seed(123)
m_fda <- train(
  x_train, y_train,
  method = "fda",
  tuneLength = 10,
  trControl = ctrl
)
p_fda <- predict(m_fda, x_test)
add_model("FDA", m_fda, p_fda)


# 10. Support Vector Machine (RBF)
#set.seed(123)
#m_svm <- train(
#  x_train, y_train,
#  method = "svmRadial",
#  tuneLength = 3,
#  trControl = ctrl
#)
#p_svm <- predict(m_svm, x_test)
#add_model("SVM (RBF)", m_svm, p_svm)


# 11. KNN classifier
set.seed(123)
m_knn <- train(
  x_train, y_train,
  method = "knn",
  tuneLength = 15,
  trControl = ctrl
)
p_knn <- predict(m_knn, x_test)
add_model("KNN", m_knn, p_knn)


# 12. Naive Bayes
set.seed(123)
m_nb <- train(
  x_train, y_train,
  method = "naive_bayes",
  trControl = ctrl
)
p_nb <- predict(m_nb, x_test)
add_model("Naive Bayes", m_nb, p_nb)



# Generating Table Results 
results_all <- do.call(rbind, results_list)
results_all <- results_all[order(-results_all$Kappa), ]

print(results_all)

write.csv(results_all, file.path("Final_Project_Phase2_Results/model_kappa_results_upsampled.csv"), row.names = FALSE)



counts_list <- list()
stats_list  <- list()

for(name in names(conf_list)) {
  cm <- conf_list[[name]]
  
  # --- Part A: Raw Counts (Prediction vs Reference) ---
  # Converts the matrix to a "Long" format: Model | Prediction | Reference | Freq
  df_counts <- as.data.frame(cm$table)
  df_counts$Model <- name
  counts_list[[name]] <- df_counts
  
  # --- Part B: Detailed Class Stats (Sensitivity, Specificity, etc.) ---
  # If there are 3 classes, this table will have 3 rows per model
  df_stats <- as.data.frame(cm$byClass)
  df_stats$Class <- rownames(df_stats) # Capture the class names
  df_stats$Model <- name
  
  # Reorder columns so Model and Class are first
  df_stats <- df_stats %>% dplyr::select(Model, Class, everything())
  stats_list[[name]] <- df_stats
}

# Combine all models into single dataframes
all_counts_df <- do.call(rbind, counts_list)
all_stats_df  <- do.call(rbind, stats_list)

# Write to CSV
write.csv(all_counts_df, file.path("Final_Project_Phase2_Results/model_confusion_matrix_counts_upsampled.csv"), row.names = FALSE)
write.csv(all_stats_df,  file.path("Final_Project_Phase2_Results/model_detailed_class_metrics_upsampled.csv"),  row.names = FALSE)






















#  MODEL TRAINING FOR 12 CLASSIFIERS DOWN SAMPLE SIZE
#  Uses caret::train(), Kappa, Confusion Matrix, and Test Set

set.seed(123)

#============================================================
# Choose which processed dataset to model
# Recommended: Box-Cox + Center/Scale + Correlation-Pruned
#============================================================

# ============================================================
# FIX CLASS LABELS (IMPORTANT FOR multinom AND OTHER MODELS)
# ============================================================

# Extract original levels
original_levels <- levels(train_base[[response]])
cat("Original response levels:\n")
print(original_levels)

# Create safe R variable names
safe_levels <- make.names(original_levels)

cat("Renamed (safe) levels:\n")
print(safe_levels)

# Apply renaming to ALL processed training/test sets
train_base[[response]]    <- factor(train_base[[response]],    levels = original_levels, labels = safe_levels)
test_base[[response]]     <- factor(test_base[[response]],     levels = original_levels, labels = safe_levels)

train_spatial[[response]] <- factor(train_spatial[[response]], levels = original_levels, labels = safe_levels)
test_spatial[[response]]  <- factor(test_spatial[[response]],  levels = original_levels, labels = safe_levels)

train_pca[[response]]     <- factor(train_pca[[response]],     levels = original_levels, labels = safe_levels)
test_pca[[response]]      <- factor(test_pca[[response]],      levels = original_levels, labels = safe_levels)


train_set <- train_base
test_set  <- test_base

response <- CFG$response

x_train <- dplyr::select(train_set, -!!sym(response))
x_test  <- dplyr::select(test_set,  -!!sym(response))

y_train <- train_set[[response]]
y_test  <- test_set[[response]]

# Common Training Control (K-fold CV)
library(doParallel)

# 1. Register Parallel Backend. This is makes it run the NN model faster
num_cores <- parallel::detectCores() - 1 
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# 2. Update Control for Monitoring
ctrl <- trainControl(
  method = "cv",
  number = 10,                 
  summaryFunction = multiClassSummary,
  classProbs = TRUE,
  allowParallel = TRUE,        
  verboseIter = TRUE,
  sampling = "down"
)

# Storage structure
model_list <- list()
results_list <- list()
conf_list <- list()

add_model <- function(name, model, prediction) {
  model_list[[name]] <<- model
  conf_list[[name]] <<- confusionMatrix(prediction, y_test)
  results_list[[name]] <<- data.frame(
    Model = name,
    Kappa = conf_list[[name]]$overall["Kappa"],
    Accuracy = conf_list[[name]]$overall["Accuracy"]
  )
}

# LINEAR MODELS


# 1. Logistic Regression ("multinom" for multi-class)
set.seed(123)
m_log <- train(
  x_train, y_train,
  method = "multinom",
  trControl = ctrl
)
p_log <- predict(m_log, x_test)
add_model("Logistic Regression", m_log, p_log)

print(m_log)

# 2. Linear Discriminant Analysis
set.seed(123)
m_lda <- train(
  x_train, y_train,
  method = "lda",
  trControl = ctrl
)
p_lda <- predict(m_lda, x_test)
add_model("LDA", m_lda, p_lda)


# 3. PLS-DA  ("pls" chooses PLSDA when outcome is factor)
set.seed(123)
m_pls <- train(
  x_train, y_train,
  method = "pls",
  tuneLength = 15,
  trControl = ctrl
)
p_pls <- predict(m_pls, x_test)
add_model("PLS-DA", m_pls, p_pls)


# 4. Lasso (glmnet α=1)
set.seed(123)
m_lasso <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl
)
p_lasso <- predict(m_lasso, x_test)
add_model("Lasso", m_lasso, p_lasso)


# 5. Ridge (glmnet α=0)
set.seed(123)
m_ridge <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.0001, 1, length = 20)),
  trControl = ctrl
)
p_ridge <- predict(m_ridge, x_test)
add_model("Ridge", m_ridge, p_ridge)



# 6. Elastic Net (α ∈ [0,1])
set.seed(123)
m_en <- train(
  x_train, y_train,
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = seq(0, 1, length = 5),
    lambda = seq(0.0001, 1, length = 20)
  ),
  trControl = ctrl
)
p_en <- predict(m_en, x_test)
add_model("Elastic Net", m_en, p_en)



# NON-LINEAR MODELS 
# 7. Nonlinear Discriminant Analysis (method = "mda")
# Fixed: changed 'nda' to 'mda' (Mixture Discriminant Analysis)
set.seed(123)
m_nda <- train(
  x_train, y_train,
  method = "mda",
  trControl = ctrl
)
p_nda <- predict(m_nda, x_test)
add_model("NDA", m_nda, p_nda)



# 8. Neural Network (nnet)
#set.seed(123)
#m_nnet <- train(
#  x_train, y_train,
#  method = "nnet",
#  trace = FALSE,
#  tuneLength = 3,
#  trControl = ctrl
#)
#p_nnet <- predict(m_nnet, x_test)
#add_model("Neural Network", m_nnet, p_nnet)


# 9. Flexible Discriminant Analysis
set.seed(123)
m_fda <- train(
  x_train, y_train,
  method = "fda",
  tuneLength = 10,
  trControl = ctrl
)
p_fda <- predict(m_fda, x_test)
add_model("FDA", m_fda, p_fda)


# 10. Support Vector Machine (RBF)
#set.seed(123)
#m_svm <- train(
#  x_train, y_train,
#  method = "svmRadial",
#  tuneLength = 3,
#  trControl = ctrl
#)
#p_svm <- predict(m_svm, x_test)
#add_model("SVM (RBF)", m_svm, p_svm)


# 11. KNN classifier
set.seed(123)
m_knn <- train(
  x_train, y_train,
  method = "knn",
  tuneLength = 15,
  trControl = ctrl
)
p_knn <- predict(m_knn, x_test)
add_model("KNN", m_knn, p_knn)


# 12. Naive Bayes
set.seed(123)
m_nb <- train(
  x_train, y_train,
  method = "naive_bayes",
  trControl = ctrl
)
p_nb <- predict(m_nb, x_test)
add_model("Naive Bayes", m_nb, p_nb)



# Generating Table Results 
results_all <- do.call(rbind, results_list)
results_all <- results_all[order(-results_all$Kappa), ]

print(results_all)

write.csv(results_all, file.path("Final_Project_Phase2_Results/model_kappa_results_downsampled.csv"), row.names = FALSE)



counts_list <- list()
stats_list  <- list()

for(name in names(conf_list)) {
  cm <- conf_list[[name]]
  
  # --- Part A: Raw Counts (Prediction vs Reference) ---
  # Converts the matrix to a "Long" format: Model | Prediction | Reference | Freq
  df_counts <- as.data.frame(cm$table)
  df_counts$Model <- name
  counts_list[[name]] <- df_counts
  
  # --- Part B: Detailed Class Stats (Sensitivity, Specificity, etc.) ---
  # If there are 3 classes, this table will have 3 rows per model
  df_stats <- as.data.frame(cm$byClass)
  df_stats$Class <- rownames(df_stats) # Capture the class names
  df_stats$Model <- name
  
  # Reorder columns so Model and Class are first
  df_stats <- df_stats %>% dplyr::select(Model, Class, everything())
  stats_list[[name]] <- df_stats
}

# Combine all models into single dataframes
all_counts_df <- do.call(rbind, counts_list)
all_stats_df  <- do.call(rbind, stats_list)

# Write to CSV
write.csv(all_counts_df, file.path("Final_Project_Phase2_Results/model_confusion_matrix_counts_downsampled.csv"), row.names = FALSE)
write.csv(all_stats_df,  file.path("Final_Project_Phase2_Results/model_detailed_class_metrics_downsampled.csv"),  row.names = FALSE)




