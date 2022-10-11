library(tidyverse)
library(ranger)
library(xtable)
library(plyr)

library(furrr)
options(warn=-1)

RAW_DATA_DIRECTORY = "./data/validation/01-data_generation/python"
DATA_DIRECTORY = "./data/validation/02-model_training/r"

args = commandArgs(trailingOnly=TRUE)
num_seeds = strtoi(unlist(strsplit(args[1], '='))[2])

df <- read.csv(paste(RAW_DATA_DIRECTORY, "/seed_000.csv", sep = ""))

fit_models <- function(seed) {
  set.seed(seed)
  df$is_train <- rbinom(nrow(df), size = 1, .7)
  train <- df %>%
    filter(is_train == 1) %>%
    select(-is_train)
  test <- df %>%
    filter(is_train == 0) %>%
    select(-is_train)

  # Train propensity model
  pi_lr <- glm("treat_num ~.", data = select(train, Z, A, treat_num), family = "binomial")

  # Train regression models
  obs_lr <- glm("outcome ~.", data = select(train, Z, A, outcome), family = "binomial")
  count_lr <- glm("outcome ~.", data = select(filter(train, treat_num == 0), Z, A, outcome), family = "binomial")

  results <- do.call(rbind, Map(data.frame, propensity=dummy.coef(pi_lr), observational=dummy.coef(obs_lr), counterfactual=dummy.coef(count_lr)))

  write.csv(results, file=paste0(DATA_DIRECTORY, "/seed_", sprintf("%03d", seed), ".csv", sep = ""), row.names = TRUE)

}


plan(multisession, workers = 8)
asdf <- future_map(c(0:(num_seeds - 1)), fit_models)