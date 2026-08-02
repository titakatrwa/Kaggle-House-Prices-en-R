options(stringsAsFactors = FALSE)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/")
  project_dir <- dirname(script_path)
} else if (dir.exists(file.path(getwd(), "house_prices"))) {
  project_dir <- normalizePath(file.path(getwd(), "house_prices"), winslash = "/")
} else {
  project_dir <- normalizePath(getwd(), winslash = "/")
}

if (!requireNamespace("rpart", quietly = TRUE)) {
  stop("Le package rpart est requis. Installez-le avec install.packages('rpart').", call. = FALSE)
}

raw_dir <- file.path(project_dir, "data", "raw")
output_dir <- file.path(project_dir, "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
train <- read.csv(file.path(raw_dir, "train.csv"), na.strings = "NA", check.names = FALSE)

numeric_features <- c(
  "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF",
  "1stFlrSF", "FullBath", "YearBuilt", "YearRemodAdd", "LotArea",
  "Fireplaces", "TotRmsAbvGrd"
)
categorical_features <- c(
  "Neighborhood", "ExterQual", "KitchenQual", "BsmtQual",
  "GarageFinish", "HouseStyle", "CentralAir"
)

rmse_log <- function(actual, predicted) {
  sqrt(mean((log(actual) - log(pmax(predicted, 1)))^2))
}

make_folds <- function(y, k = 5L, seed = 2026) {
  breaks <- unique(quantile(y, probs = seq(0, 1, 0.1), na.rm = TRUE))
  strata <- cut(y, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  set.seed(seed)
  folds <- integer(length(y))
  for (indices in split(seq_along(y), strata)) {
    folds[indices] <- sample(rep(seq_len(k), length.out = length(indices)))
  }
  folds
}

fit_preprocessor <- function(data, rare_minimum = 20L) {
  medians <- vapply(data[numeric_features], median, numeric(1), na.rm = TRUE)
  levels_map <- lapply(data[categorical_features], function(x) {
    x[is.na(x)] <- "None"
    counts <- table(x)
    kept <- names(counts[counts >= rare_minimum])
    unique(c(kept, "Other"))
  })
  list(medians = medians, levels = levels_map)
}

apply_preprocessor <- function(data, prep) {
  result <- data.frame(Id = data$Id, check.names = FALSE)
  for (variable in numeric_features) {
    values <- data[[variable]]
    values[is.na(values)] <- prep$medians[[variable]]
    result[[variable]] <- values
  }
  for (variable in categorical_features) {
    values <- data[[variable]]
    values[is.na(values)] <- "None"
    values[!values %in% prep$levels[[variable]]] <- "Other"
    result[[variable]] <- factor(values, levels = prep$levels[[variable]])
  }
  if ("SalePrice" %in% names(data)) result$SalePrice <- data$SalePrice
  result
}

linear_formula <- as.formula(paste(
  "log(SalePrice) ~",
  paste(c(sprintf("`%s`", numeric_features), categorical_features), collapse = " + ")
))
tree_formula <- as.formula(paste(
  "log(SalePrice) ~",
  paste(c(sprintf("`%s`", numeric_features), categorical_features), collapse = " + ")
))

fold_id <- make_folds(train$SalePrice)
models <- c("Moyenne géométrique", "Régression linéaire log", "Arbre de régression")
fold_results <- data.frame()
oof <- data.frame(
  Id = train$Id,
  SalePrice = train$SalePrice,
  Fold = fold_id,
  Naive = NA_real_,
  Linear = NA_real_,
  Tree = NA_real_
)

for (fold in sort(unique(fold_id))) {
  analysis_raw <- train[fold_id != fold, ]
  validation_raw <- train[fold_id == fold, ]
  prep <- fit_preprocessor(analysis_raw)
  analysis <- apply_preprocessor(analysis_raw, prep)
  validation <- apply_preprocessor(validation_raw, prep)

  naive <- rep(exp(mean(log(analysis$SalePrice))), nrow(validation))
  linear_model <- lm(linear_formula, data = analysis)
  linear <- exp(predict(linear_model, newdata = validation))
  tree_model <- rpart::rpart(
    tree_formula,
    data = analysis,
    method = "anova",
    control = rpart::rpart.control(cp = 0.003, minsplit = 20, maxdepth = 8, xval = 0)
  )
  tree <- exp(predict(tree_model, newdata = validation))

  row_indices <- which(fold_id == fold)
  oof$Naive[row_indices] <- naive
  oof$Linear[row_indices] <- linear
  oof$Tree[row_indices] <- tree

  current <- data.frame(
    Fold = fold,
    Modele = models,
    RMSE_log = c(
      rmse_log(validation$SalePrice, naive),
      rmse_log(validation$SalePrice, linear),
      rmse_log(validation$SalePrice, tree)
    )
  )
  fold_results <- rbind(fold_results, current)
}

summary_results <- do.call(rbind, lapply(split(fold_results, fold_results$Modele), function(x) {
  data.frame(
    Modele = x$Modele[1],
    RMSE_moyenne = mean(x$RMSE_log),
    RMSE_ecart_type = sd(x$RMSE_log),
    RMSE_minimum = min(x$RMSE_log),
    RMSE_maximum = max(x$RMSE_log),
    row.names = NULL
  )
}))
summary_results <- summary_results[order(summary_results$RMSE_moyenne), ]

cat("=== RMSE LOG PAR PLI (PLUS PETIT = MEILLEUR) ===\n")
print(reshape(fold_results, idvar = "Fold", timevar = "Modele", direction = "wide"), row.names = FALSE)

cat("\n=== SYNTHÈSE 5 PLIS ===\n")
print(transform(
  summary_results,
  RMSE_moyenne = round(RMSE_moyenne, 4),
  RMSE_ecart_type = round(RMSE_ecart_type, 4),
  RMSE_minimum = round(RMSE_minimum, 4),
  RMSE_maximum = round(RMSE_maximum, 4)
), row.names = FALSE)

cat("\n=== DIAGNOSTIC DE L'OBSERVATION ID 1299 ===\n")
outlier_row <- oof[oof$Id == 1299, ]
print(outlier_row, row.names = FALSE)
cat(sprintf("RMSE OOF linéaire avec toutes les maisons : %.4f\n", rmse_log(oof$SalePrice, oof$Linear)))
cat(sprintf("RMSE OOF linéaire sans Id 1299 (diagnostic uniquement) : %.4f\n",
            rmse_log(oof$SalePrice[oof$Id != 1299], oof$Linear[oof$Id != 1299])))

oof$LinearAbsLogError <- abs(log(oof$SalePrice) - log(pmax(oof$Linear, 1)))
oof <- oof[order(oof$LinearAbsLogError, decreasing = TRUE), ]
write.csv(fold_results, file.path(output_dir, "cross_validation_folds.csv"), row.names = FALSE)
write.csv(summary_results, file.path(output_dir, "cross_validation_summary.csv"), row.names = FALSE)
write.csv(oof, file.path(output_dir, "oof_predictions.csv"), row.names = FALSE)

cat("\nRésultats enregistrés dans outputs/.\n")
