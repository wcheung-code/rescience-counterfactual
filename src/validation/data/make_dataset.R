library(tidyverse)
library(ranger)
library(xtable)
library(plyr)

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

print(df)