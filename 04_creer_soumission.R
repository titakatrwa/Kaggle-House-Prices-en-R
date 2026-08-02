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

raw_dir <- file.path(project_dir, "data", "raw")
output_dir <- file.path(project_dir, "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read.csv(file.path(raw_dir, "train.csv"), na.strings = "NA", check.names = FALSE)
test <- read.csv(file.path(raw_dir, "test.csv"), na.strings = "NA", check.names = FALSE)
sample <- read.csv(file.path(raw_dir, "sample_submission.csv"), check.names = FALSE)

numeric_features <- c(
  "OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF",
  "1stFlrSF", "FullBath", "YearBuilt", "YearRemodAdd", "LotArea",
  "Fireplaces", "TotRmsAbvGrd"
)
categorical_features <- c(
  "Neighborhood", "ExterQual", "KitchenQual", "BsmtQual",
  "GarageFinish", "HouseStyle", "CentralAir"
)

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

prep <- fit_preprocessor(train)
train_clean <- apply_preprocessor(train, prep)
test_clean <- apply_preprocessor(test, prep)

formula <- as.formula(paste(
  "log(SalePrice) ~",
  paste(c(sprintf("`%s`", numeric_features), categorical_features), collapse = " + ")
))
final_model <- lm(formula, data = train_clean)
predicted_price <- exp(predict(final_model, newdata = test_clean))

submission <- data.frame(Id = test$Id, SalePrice = predicted_price)

# Contrôles du contrat de soumission Kaggle.
stopifnot(nrow(submission) == nrow(test))
stopifnot(nrow(submission) == 1459)
stopifnot(identical(submission$Id, sample$Id))
stopifnot(!anyNA(submission))
stopifnot(all(is.finite(submission$SalePrice)))
stopifnot(all(submission$SalePrice > 0))

submission_path <- file.path(output_dir, "submission_linear_log.csv")
write.csv(submission, submission_path, row.names = FALSE, quote = FALSE)
saveRDS(list(model = final_model, preprocessor = prep),
        file.path(output_dir, "final_linear_log_model.rds"))

cat("=== SOUMISSION HOUSE PRICES CRÉÉE ===\n")
cat(sprintf("Fichier : %s\n", submission_path))
cat(sprintf("Lignes  : %d\n", nrow(submission)))
cat(sprintf("Prix prédit minimum : $%s\n", format(round(min(submission$SalePrice)), big.mark = ",")))
cat(sprintf("Prix prédit médian  : $%s\n", format(round(median(submission$SalePrice)), big.mark = ",")))
cat(sprintf("Prix prédit maximum : $%s\n\n", format(round(max(submission$SalePrice)), big.mark = ",")))
print(head(submission, 10))
