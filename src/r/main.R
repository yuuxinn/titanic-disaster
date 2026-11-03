# Titanic survival prediction - R version.
#
# Loads train.csv / test.csv, cleans and engineers features, fits a
# logistic regression model (glm), and reports accuracy. Every step prints
# what it did so the whole pipeline can be followed from the terminal.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

set.seed(42)

banner <- function(text) {
  line <- strrep("=", 70)
  cat("\n", line, "\n", text, "\n", line, "\n", sep = "")
}

# Locate the data folder.
# Inside Docker the data is mounted at /app/data (next to this script).
# When running locally from the repo, the data lives in src/data
# (one level up from this script). DATA_DIR overrides both.
find_data_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_dir <- if (length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    getwd()
  }
  candidates <- c(Sys.getenv("DATA_DIR"),
                  file.path(script_dir, "data"),
                  file.path(script_dir, "..", "data"))
  for (path in candidates) {
    if (nzchar(path) && file.exists(file.path(path, "train.csv"))) {
      return(normalizePath(path))
    }
  }
  cat("ERROR: could not find train.csv.\n")
  cat("Put train.csv and test.csv in src/data/ (see README.md).\n")
  quit(status = 1)
}

# Pull the title (Mr, Mrs, Miss, ...) out of the Name column.
extract_title <- function(names) {
  titles <- trimws(sub(".*,\\s*([^.]+)\\..*", "\\1", names))
  titles[titles %in% c("Mlle", "Ms")] <- "Miss"
  titles[titles == "Mme"] <- "Mrs"
  titles[!titles %in% c("Mr", "Mrs", "Miss", "Master")] <- "Rare"
  titles
}

# Compute imputation values from the TRAINING set only.
learn_fill_values <- function(train) {
  age_by_title <- train %>%
    mutate(Title = extract_title(Name)) %>%
    group_by(Title) %>%
    summarise(med = median(Age, na.rm = TRUE), .groups = "drop")
  embarked_counts <- table(train$Embarked[!is.na(train$Embarked) & train$Embarked != ""])
  list(
    age_by_title = setNames(age_by_title$med, age_by_title$Title),
    age_overall  = median(train$Age, na.rm = TRUE),
    fare         = median(train$Fare, na.rm = TRUE),
    embarked     = names(embarked_counts)[which.max(embarked_counts)]
  )
}

# Clean the data and build model features. Prints every change.
prepare <- function(df, fill, name) {
  cat(sprintf("\n--- Preparing %s set ---\n", name))

  df$Title <- extract_title(df$Name)
  cat("[add] Title extracted from Name:", paste(names(table(df$Title)),
      table(df$Title), sep = "=", collapse = ", "), "\n")

  n_missing <- sum(is.na(df$Age))
  df$Age <- ifelse(is.na(df$Age), fill$age_by_title[df$Title], df$Age)
  df$Age[is.na(df$Age)] <- fill$age_overall
  cat(sprintf("[fill] Age: %d missing values filled with median age per Title\n", n_missing))

  n_missing <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fill$fare
  cat(sprintf("[fill] Fare: %d missing values filled with median %.2f\n", n_missing, fill$fare))

  missing_emb <- is.na(df$Embarked) | df$Embarked == ""
  df$Embarked[missing_emb] <- fill$embarked
  cat(sprintf("[fill] Embarked: %d missing values filled with mode '%s'\n",
              sum(missing_emb), fill$embarked))

  df <- df %>%
    mutate(
      HasCabin   = as.integer(!is.na(Cabin) & Cabin != ""),
      FamilySize = SibSp + Parch + 1,
      IsAlone    = as.integer(FamilySize == 1)
    )
  cat(sprintf("[add] HasCabin (1 if a cabin is listed): %d passengers have a cabin\n", sum(df$HasCabin)))
  cat(sprintf("[add] FamilySize = SibSp + Parch + 1 (mean %.2f)\n", mean(df$FamilySize)))
  cat(sprintf("[add] IsAlone: %d passengers travelling alone\n", sum(df$IsAlone)))

  # Fixed factor levels so train and test always encode the same way.
  df <- df %>%
    mutate(
      Pclass   = factor(Pclass, levels = c(3, 2, 1)),
      Embarked = factor(Embarked, levels = c("S", "C", "Q")),
      Title    = factor(Title, levels = c("Mr", "Mrs", "Miss", "Master", "Rare"))
    )
  cat("[encode] Pclass, Embarked, Title converted to factors (baseline: 3rd class, S, Mr)\n")
  cat("[drop] Columns not used as features: PassengerId, Name, Ticket, Cabin, SibSp, Parch, Sex\n")
  cat("       (Sex is left out because Title already encodes it - keeping both makes glm unstable)\n")
  df
}

main <- function() {
  data_dir <- find_data_dir()

  banner("STEP 1: Load training data")
  train <- read_csv(file.path(data_dir, "train.csv"), show_col_types = FALSE)
  cat("Loaded train.csv from", data_dir, "\n")
  cat(sprintf("Shape: %d rows x %d columns\n", nrow(train), ncol(train)))
  cat("\nFirst 5 rows:\n")
  print(as.data.frame(head(train, 5)))

  banner("STEP 2: Explore training data")
  cat("Missing values per column:\n")
  print(colSums(is.na(train)))
  cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
  cat("\nSurvival rate by Sex:\n")
  print(as.data.frame(train %>% group_by(Sex) %>%
          summarise(SurvivalRate = round(mean(Survived), 3))))
  cat("\nSurvival rate by Pclass:\n")
  print(as.data.frame(train %>% group_by(Pclass) %>%
          summarise(SurvivalRate = round(mean(Survived), 3))))

  banner("STEP 3: Clean data and engineer features")
  fill <- learn_fill_values(train)
  train_prep <- prepare(train, fill, "training")

  model_formula <- Survived ~ Age + Pclass + Fare + HasCabin +
    FamilySize + IsAlone + Embarked + Title
  cat("\nModel formula:", deparse(model_formula, width.cutoff = 200L), "\n")

  banner("STEP 4: Logistic regression - accuracy on training data")
  # Hold out 20% of train.csv so we also get an honest estimate on unseen data.
  val_idx <- sample(seq_len(nrow(train_prep)), size = round(0.2 * nrow(train_prep)))
  val_model <- glm(model_formula, data = train_prep[-val_idx, ], family = binomial)
  val_pred <- as.integer(predict(val_model, train_prep[val_idx, ], type = "response") > 0.5)
  cat(sprintf("Hold-out validation (80/20 split of train.csv): %.4f\n",
              mean(val_pred == train_prep$Survived[val_idx])))

  model <- glm(model_formula, data = train_prep, family = binomial)
  train_pred <- as.integer(predict(model, train_prep, type = "response") > 0.5)
  cat(sprintf("Model refit on all %d training rows.\n", nrow(train_prep)))
  cat(sprintf("Training accuracy: %.4f\n", mean(train_pred == train_prep$Survived)))
  cat("\nModel coefficients:\n")
  print(round(coef(summary(model)), 3))

  banner("STEP 5: Load test data and predict")
  test <- read_csv(file.path(data_dir, "test.csv"), show_col_types = FALSE)
  cat(sprintf("Loaded test.csv: %d rows x %d columns\n", nrow(test), ncol(test)))
  test_prep <- prepare(test, fill, "test")
  test_pred <- as.integer(predict(model, test_prep, type = "response") > 0.5)
  cat(sprintf("\nPredicted survivors: %d of %d (%.3f survival rate)\n",
              sum(test_pred), length(test_pred), mean(test_pred)))

  out_path <- file.path(data_dir, "predictions_r.csv")
  write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), out_path)
  cat("Predictions saved to", out_path, "\n")

  banner("STEP 6: Accuracy on test data")
  if ("Survived" %in% names(test)) {
    cat(sprintf("Test accuracy: %.4f\n", mean(test_pred == test$Survived)))
  } else {
    cat("test.csv has no 'Survived' column - Kaggle keeps the true labels private,",
        "so test accuracy cannot be computed locally.\n")
    cat("Upload predictions_r.csv to the Kaggle competition to get the official test accuracy.\n")
    cat("The hold-out validation accuracy in STEP 4 is the best local estimate of test performance.\n")
  }

  banner("Done")
}

main()
