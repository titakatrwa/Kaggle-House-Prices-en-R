options(stringsAsFactors = FALSE)

# Compatible avec Rscript, Source dans RStudio et source() depuis Projet_R.
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", script_arg), winslash = "/")
  project_dir <- dirname(script_path)
} else if (dir.exists(file.path(getwd(), "house_prices"))) {
  project_dir <- normalizePath(file.path(getwd(), "house_prices"), winslash = "/")
} else {
  project_dir <- normalizePath(getwd(), winslash = "/")
}

raw_dir <- file.path(project_dir, "data", "raw")
output_dir <- file.path(project_dir, "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
train <- read.csv(file.path(raw_dir, "train.csv"), na.strings = "NA", check.names = FALSE)

# Variables retenues pour une baseline lisible, rapide et explicable.
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

mae <- function(actual, predicted) mean(abs(actual - predicted))

# Pour une régression, la stratification se fait sur des tranches de prix.
stratified_regression_split <- function(y, validation_fraction = 0.20, seed = 2026) {
  breaks <- unique(quantile(y, probs = seq(0, 1, 0.1), na.rm = TRUE))
  strata <- cut(y, breaks = breaks, include.lowest = TRUE, labels = FALSE)
  set.seed(seed)
  validation <- unlist(lapply(split(seq_along(y), strata), function(indices) {
    sample(indices, size = max(1, floor(length(indices) * validation_fraction)))
  }), use.names = FALSE)
  list(analysis = setdiff(seq_along(y), validation), validation = sort(validation))
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

split <- stratified_regression_split(train$SalePrice)
analysis_raw <- train[split$analysis, ]
validation_raw <- train[split$validation, ]
prep <- fit_preprocessor(analysis_raw)
analysis <- apply_preprocessor(analysis_raw, prep)
validation <- apply_preprocessor(validation_raw, prep)

formula <- as.formula(paste(
  "log(SalePrice) ~",
  paste(c(sprintf("`%s`", numeric_features), categorical_features), collapse = " + ")
))
model <- lm(formula, data = analysis)

predicted_log <- predict(model, newdata = validation)
predicted_price <- exp(predicted_log)
naive_price <- rep(exp(mean(log(analysis$SalePrice))), nrow(validation))

results <- data.frame(
  modele = c("Moyenne géométrique", "Régression linéaire log"),
  RMSE_log = c(
    rmse_log(validation$SalePrice, naive_price),
    rmse_log(validation$SalePrice, predicted_price)
  ),
  MAE_prix = c(
    mae(validation$SalePrice, naive_price),
    mae(validation$SalePrice, predicted_price)
  ),
  stringsAsFactors = FALSE
)

cat("=== SÉPARATION STRATIFIÉE PAR TRANCHES DE PRIX ===\n")
cat(sprintf("Apprentissage : %d maisons | médiane $%s\n",
            nrow(analysis), format(round(median(analysis$SalePrice)), big.mark = ",")))
cat(sprintf("Validation    : %d maisons | médiane $%s\n\n",
            nrow(validation), format(round(median(validation$SalePrice)), big.mark = ",")))

cat("=== PERFORMANCES (PLUS PETIT = MEILLEUR) ===\n")
print(transform(results, RMSE_log = round(RMSE_log, 4), MAE_prix = round(MAE_prix)), row.names = FALSE)

residuals_log <- log(validation$SalePrice) - predicted_log
cat("\n=== ERREURS DE LA RÉGRESSION ===\n")
cat(sprintf("Erreur médiane absolue : $%s\n", format(round(median(abs(validation$SalePrice - predicted_price))), big.mark = ",")))
cat(sprintf("90e percentile erreur absolue : $%s\n", format(round(quantile(abs(validation$SalePrice - predicted_price), 0.90)), big.mark = ",")))
cat(sprintf("Corrélation prix réel/prédit : %.3f\n", cor(validation$SalePrice, predicted_price)))

coefficients <- coef(summary(model))
coefficients <- coefficients[order(abs(coefficients[, "t value"]), decreasing = TRUE), , drop = FALSE]
cat("\n=== TERMES LES PLUS INFLUENTS (|t|) ===\n")
print(round(head(coefficients, 15), 4))

validation_predictions <- data.frame(
  Id = validation$Id,
  SalePrice = validation$SalePrice,
  Prediction = predicted_price,
  Error = predicted_price - validation$SalePrice,
  AbsoluteError = abs(predicted_price - validation$SalePrice),
  LogResidual = residuals_log
)
validation_predictions <- validation_predictions[order(validation_predictions$AbsoluteError, decreasing = TRUE), ]
write.csv(validation_predictions, file.path(output_dir, "validation_predictions.csv"), row.names = FALSE)
saveRDS(list(model = model, preprocessor = prep, split = split, metrics = results),
        file.path(output_dir, "baseline_linear_log.rds"))

cat("\nRésultats enregistrés dans outputs/.\n")
