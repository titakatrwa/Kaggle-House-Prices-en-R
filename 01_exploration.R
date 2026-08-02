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
train_path <- file.path(raw_dir, "train.csv")
test_path <- file.path(raw_dir, "test.csv")
sample_path <- file.path(raw_dir, "sample_submission.csv")

required <- c(train_path, test_path, sample_path)
missing_files <- required[!file.exists(required)]
if (length(missing_files) > 0) {
  stop(
    paste0(
      "Données House Prices absentes. Placez les fichiers suivants dans :\n",
      raw_dir, "\n\n- ", paste(basename(missing_files), collapse = "\n- ")
    ),
    call. = FALSE
  )
}

train <- read.csv(train_path, na.strings = c("NA"), check.names = FALSE)
test <- read.csv(test_path, na.strings = c("NA"), check.names = FALSE)
sample <- read.csv(sample_path, check.names = FALSE)

stopifnot("SalePrice" %in% names(train))
stopifnot(!"SalePrice" %in% names(test))
stopifnot(all(c("Id", "SalePrice") %in% names(sample)))
stopifnot(nrow(test) == nrow(sample))
stopifnot(identical(test$Id, sample$Id))

cat("=== DIMENSIONS ===\n")
cat(sprintf("Entraînement : %d lignes x %d colonnes\n", nrow(train), ncol(train)))
cat(sprintf("Test          : %d lignes x %d colonnes\n", nrow(test), ncol(test)))
cat(sprintf("Soumission    : %d lignes x %d colonnes\n\n", nrow(sample), ncol(sample)))

cat("=== TYPES DE VARIABLES ===\n")
types <- vapply(train, function(x) class(x)[1], character(1))
print(sort(table(types), decreasing = TRUE))

missing_summary <- data.frame(
  variable = names(train),
  type = types,
  manquants = vapply(train, function(x) sum(is.na(x)), integer(1)),
  pourcentage = round(vapply(train, function(x) mean(is.na(x)) * 100, numeric(1)), 1),
  row.names = NULL
)
missing_summary <- missing_summary[order(missing_summary$manquants, decreasing = TRUE), ]

cat("\n=== VARIABLES AVEC VALEURS MANQUANTES (TRAIN) ===\n")
print(missing_summary[missing_summary$manquants > 0, ], row.names = FALSE)

cat("\n=== QUALITÉ ET COHÉRENCE ===\n")
cat(sprintf("Id dupliqués dans train         : %d\n", sum(duplicated(train$Id))))
cat(sprintf("Lignes entièrement dupliquées   : %d\n", sum(duplicated(train))))
cat(sprintf("SalePrice manquants             : %d\n", sum(is.na(train$SalePrice))))
cat(sprintf("SalePrice nuls ou négatifs      : %d\n", sum(train$SalePrice <= 0, na.rm = TRUE)))
cat(sprintf("Colonnes communes train/test    : %d\n", length(intersect(names(train), names(test)))))

cat("\n=== CIBLE SALEPRICE ===\n")
target_stats <- c(
  minimum = min(train$SalePrice),
  q1 = unname(quantile(train$SalePrice, 0.25)),
  mediane = median(train$SalePrice),
  moyenne = mean(train$SalePrice),
  q3 = unname(quantile(train$SalePrice, 0.75)),
  maximum = max(train$SalePrice)
)
print(round(target_stats, 2))
cat(sprintf("Asymétrie empirique (SalePrice)      : %.3f\n", mean((train$SalePrice - mean(train$SalePrice))^3) / sd(train$SalePrice)^3))
cat(sprintf("Asymétrie empirique (log SalePrice)  : %.3f\n", mean((log(train$SalePrice) - mean(log(train$SalePrice)))^3) / sd(log(train$SalePrice))^3))

numeric_predictors <- names(train)[vapply(train, is.numeric, logical(1))]
numeric_predictors <- setdiff(numeric_predictors, c("Id", "SalePrice"))
correlations <- vapply(numeric_predictors, function(variable) {
  cor(train[[variable]], log(train$SalePrice), use = "complete.obs")
}, numeric(1))
correlations <- sort(correlations, decreasing = TRUE)

cat("\n=== PLUS FORTES CORRÉLATIONS AVEC LOG(SALEPRICE) ===\n")
print(round(head(correlations, 10), 3))

write.csv(missing_summary, file.path(project_dir, "outputs", "missing_values_train.csv"), row.names = FALSE)
cat("\nExploration initiale terminée. Le détail des valeurs manquantes est dans outputs/missing_values_train.csv.\n")
