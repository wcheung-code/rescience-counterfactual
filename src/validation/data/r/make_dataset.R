library(tidyverse)
library(ranger)
library(xtable)
library(plyr)

DATA_DIRECTORY = "./data/validation/01-data_generation/r"

sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

# number of samples
n <- 100000
# probability of group membership
p <- 0.5
# k specifies treatment assignment bias
k <- 1.6
# c controls treatment effect
c <- 0.1

args = commandArgs(trailingOnly=TRUE)
num_seeds = strtoi(unlist(strsplit(args[1], '='))[2])

for (seed in 0:(num_seeds - 1)) {
  set.seed(seed)
  X <- rnorm(n, 0)
  A <- rbinom(n, size = 1, p)
  s <- sigmoid(X - .5)
  y0 <- rbinom(n, 1, sigmoid(X - .5))
  y1 <- rbinom(n, 1, c * sigmoid(X - .5))
  is_train <- rbinom(n, size = 1, .7)
  T_in <- sigmoid(X - .5 + k * A)
  Treat <- rbinom(n, size = 1, T_in)
  y <- Treat * y1 + (1 - Treat) * y0
  df <- data.frame("X" = X, "treat_num" = Treat, "y" = y, "A" = A, "y1" = y1, "y0" = y0, "s" = s, "is_train" = is_train)
  df <- df %>%
    mutate(treat = as.factor(plyr::mapvalues(df$treat_num, c(1, 0), c("treat", "control")))) %>%
    mutate(outcome = factor(plyr::mapvalues(df$y, c(1, 0), c("harm", "ok")), levels = c("ok", "harm")))

  train <- df %>% 
    filter(is_train == 1) %>%
    select(-is_train)
  test <- df %>%
    filter(is_train == 0) %>%
    select(-is_train)
  write.csv(df, file=paste0(DATA_DIRECTORY, "/seed_", sprintf("%02d", seed), ".csv", sep = ""), row.names = FALSE)
}


