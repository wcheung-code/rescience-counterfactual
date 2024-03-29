# This is notebook for recreating the synthetic experiments in the paper Counterfactual Risk Assessments, Evaluation, and Fairness
# -----------------
library(tidyverse)
library(ranger)
library(xtable)
library(dplyr)

source("./github/doubly_robust_metrics.R")
source("./github/observational_metrics.R")
source("./github/synthetic_helper_functions.R")
source("./github/plot_params.r")

args = commandArgs(trailingOnly=TRUE)
seed = strtoi(unlist(strsplit(args[1], '='))[2])

# The post-processing scripts requires a python environment with numpy, pandas, sys, and cvxpy installed. 
# You can specify the path to this environment in "python_path"
# -----------------
# name of folder to save the plot figures in
fig_folder <- "./reproduction/"
# name of folder to save the data files in
data_folder <- "./data/"
# path of python environment
python_path <- "python3" # "/Users/amandacoston/anaconda/envs/py3/bin/python" 

# Sigmoid function
# -----------------
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

set.seed(seed)

# Generate data files
# -----------------
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

# Train propensity model
pi_lr <- glm("treat ~.", data = select(train, X, A, treat), family = "binomial")

train <- train %>%
  mutate(pi = unname(predict(pi_lr, train, type = "response")))

# Train regression models
obs_lr <- glm("outcome ~.", data = select(train, X, A, outcome), family = "binomial")
count_lr <- glm("outcome ~.", data = select(filter(train, treat == "control"), X, A, outcome), family = "binomial")

train <- train %>%
  mutate(mu_obs = unname(predict(obs_lr, train, type = "response"))) %>%
  mutate(mu_count = unname(predict(count_lr, train, type = "response")))
test <- test %>%
  mutate(pi = unname(predict(pi_lr, test, type = "response")))

test <- test %>%
  mutate(mu_obs = unname(predict(obs_lr, test, type = "response"))) %>%
  mutate(mu_count = unname(predict(count_lr, test, type = "response")))

# save data files
write_csv(select(train, -treat, -outcome), paste(data_folder, "train_c", c * 100, "_k", k * 100, ".csv", sep = ""))
write_csv(select(test, -treat, -outcome), paste(data_folder, "test_c", c * 100, "_k", k * 100, ".csv", sep = ""))
k_str <- toString(k * 100)
c_str <- toString(c * 100)

# call python script to do post-processing
system(paste(python_path, ' "./github/post_process.py" ', paste(data_folder, "train_c", c_str, "_k", k_str, ".csv", sep = ""), paste(data_folder, "test_c", c_str, "_k", k_str, ".csv", sep = ""), " ", paste(data_folder, "train_post_c", c_str, "_k", k_str, ".csv", sep = ""), " ", paste(data_folder, "test_post_c", c_str, "_k", k_str, ".csv", sep = ""), '"A" "mu_count" "y"'))

# read in post-processed results
train_post <- read.csv(paste(data_folder, "train_post_c", c_str, "_k", k_str, ".csv", sep = ""))
test_post <- read.csv(paste(data_folder, "test_post_c", c_str, "_k", k_str, ".csv", sep = ""))

roc_df <- process_post_synthetic(test_post, "A = 1", "A = 0", "mu_count", "eo_fair_pred", "y0", "y", "Original", "Post-Processed")

ggplot(filter(roc_df, Evaluation == "Counterfactual"), aes(x = FPR, y = TPR, color = Group, fill = Group)) + geom_line() + facet_grid(. ~ Method, switch = "y") + scale_colour_manual(values = group_colors_invert) + theme_bw(base_size = 15) + scale_x_continuous(labels = scale_format) + scale_y_continuous(labels = scale_format) + coord_fixed(ratio = 1) + theme(legend.position = "bottom", legend.key.width = unit(2, "cm")) + ylab("Recall")
ggsave(paste(fig_folder, "post_processed/", "fig_roc_seed_", sprintf("%03d", seed), ".png", sep = ""), width = 5, height = 4, dpi = 1000)

errors <- compute_rates_synthetic(0.5, filter(test_post, A == 1)$mu_count, y0_values = filter(test_post, A == 1)$y0, y_values = filter(test_post, A == 1)$y, group = "A=1", method = "Original") %>%
  bind_rows(compute_rates_synthetic(0.5, filter(test_post, A == 0)$mu_count, filter(test_post, A == 0)$y0, filter(test_post, A == 0)$y, group = "A=0", method = "Original")) %>%
  bind_rows(compute_rates_synthetic(0.5, filter(test_post, A == 1)$eo_fair_pred, y0_values = filter(test_post, A == 1)$y0, y_values = filter(test_post, A == 1)$y, group = "A=1", method = "Post-Proc.")) %>%
  bind_rows(compute_rates_synthetic(0.5, filter(test_post, A == 0)$eo_fair_pred, filter(test_post, A == 0)$y0, filter(test_post, A == 0)$y, group = "A=0", method = "Post-Proc."))

print(xtable(errors, caption = paste("Observational and counterfactual generalized FPR/FNR for the original and post-processed models for c =", toString(c), ", k = ", toString(k), sep = ""), label = paste("fig_synth_post_costs_c", c_str, "_k", k_str, sep = "")), floating = FALSE, latex.environments = NULL, file = paste(fig_folder, "post_processed/", "fig_table_seed_", sprintf("%03d", seed), ".tex", sep = ""), include.rownames = FALSE)

## mode:validation
#write.csv(filter(roc_df, Evaluation == "Counterfactual"), file = paste(data_folder, 'post_processed/', 'seed_', sprintf("%03d", seed), ".csv", sep = ""))


# DR estimate of counterfactual calibration curve
calib_dr <- compute_calib_df_dr(num_bins = 20, dat = test,obs_preds = quo(mu_obs),   count_preds = quo(mu_count), pi = quo(pi), Y = quo(y), treat = quo(treat_num), mu_true = quo(mu_count), attr = quo(A), attr_name = "A", attr_name_other = "not A") %>% filter(Group == "All")
calib_dr$Evaluation <- "Doubly-robust"

# True counterfactual calibration curve
calib_true <- compute_calib_df(20, test, quo(mu_count), quo(mu_obs), quo(y0), "All")
calib_true$Evaluation <- "True Counterfactual"

# Calibration curve on control
calib_control <- compute_calib_df(20, filter(test, treat_num == 0), quo(mu_count), quo(mu_obs), quo(y), "All")
calib_control$Evaluation <- "Control"

# Observational calibration curve
calib_obs <- compute_calib_df(20, test, quo(mu_count), quo(mu_obs), quo(y), "All")
calib_obs$Evaluation <- "Observational"

# Calibration curves combined
calib_comb <- rbind(calib_true, calib_control, calib_dr, calib_obs)
calib_comb$Evaluation <- factor(calib_comb$Evaluation, levels = c("Observational", "Control", "Doubly-robust", "True Counterfactual"))
calib_comb %>%
  mutate(Model = calib_comb$Method) %>%
  ggplot(aes(x = Average.score, y = Rate, color = Model, fill = Model)) + facet_grid(. ~ Evaluation, labeller = as_labeller(evaluation_names_synth)) + scale_fill_manual(values = method_colors) + scale_color_manual(values = method_colors) + geom_ribbon(aes(ymin = Low, ymax = High, alpha = 0.4), linetype = 0) + geom_abline(slope = 1, intercept = 0, linetype = 5) + geom_line() + ylab("Outcome rate") + xlab("Average risk score") + scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c(0, .25, .5, .75, 1), limits = c(0, 1)) + scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c(0, .25, .5, .75, 1), limits = c(0, 1)) + coord_fixed(ratio = 1) + theme_bw(base_size = 15) + theme(legend.position = "none") + scale_alpha(guide = "none")
ggsave(paste(fig_folder, "calibration/", "fig_seed_", sprintf("%03d", seed), ".png", sep = ""), width = 10, height = 3.6, dpi = 100)

## mode:validation
#write.csv(calib_comb, file = paste(data_folder, 'calibration/', 'seed_', sprintf("%03d", seed), ".csv", sep = ""))

t_arr <- seq(0, 1, 0.001)
ylim_begin <- 0

pr_dr <- compute_PR_df_dr(t_arr, test, "All", obs_preds = quo(mu_obs), count_preds = quo(mu_count), treat = quo(treat_num), pi = quo(pi), Y = quo(y), mu_true = quo(mu_count))

pr_dr <- pr_dr %>%
  drop_na(Precision, Recall) %>%
  mutate(Evaluation = "Doubly-robust")

pr_true <- compute_pr_df(test, quo(y0), quo(mu_count), quo(mu_obs), "All") %>%
  drop_na(Precision, Recall) %>%
  mutate(Evaluation = "True Counterfactual")

pr_control <- compute_pr_df(filter(test, treat_num == 0), quo(y), quo(mu_count), quo(mu_obs), "All") %>%
  drop_na(Precision, Recall) %>%
  mutate(Evaluation = "Control")

pr_obs <- compute_pr_df(test, quo(y), quo(mu_count), quo(mu_obs), "All") %>%
  drop_na(Precision, Recall) %>%
  mutate(Evaluation = "Observational")
pr_comb <- rbind(pr_true, select(pr_dr, -Precision.lower, -Precision.upper), pr_obs, pr_control)
pr_comb$Evaluation <- factor(pr_comb$Evaluation, levels = c("Observational", "Control", "Doubly-robust", "True Counterfactual"))
pr_comb$Model <- pr_comb$Method
ggplot(pr_comb, aes(x = Recall, y = Precision, color = Model, fill = Model)) + geom_line() + scale_color_manual(values = method_colors) + facet_grid(. ~ Evaluation, labeller = as_labeller(evaluation_names_synth)) + coord_fixed(ratio = 1) + scale_x_continuous(labels = scale_format) + scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c(0, .25, .5, .75, 1), limits = c(0, 1)) + theme_bw(base_size = 15) + theme(legend.position = "top", legend.key.width = unit(2, "cm"))
ggsave(paste(fig_folder, "precision_recall/", "fig_seed_", sprintf("%03d", seed), ".png", sep = ""), width = 10, height = 4, dpi = 1000)

## mode:validation
#write.csv(pr_comb, file = paste(data_folder, 'precision_recall/', 'seed_', sprintf("%03d", seed), ".csv", sep = ""))

t_arr <- seq(0, 1, .01)
#auc_A <- create_ROC_df(t_arr, filter(test, A == 1), "A", method1 = quo(mu_obs), method2 = quo(mu_count), method1_name = "Observational", method2_name = "Counterfactual", treat = quo(treat_num), pi = quo(pi), Y = quo(y), mu_true = quo(mu_count))
#auc_not_A <- create_ROC_df(t_arr, filter(test, A == 0), "not A", method1 = quo(mu_obs), method2 = quo(mu_count), method1_name = "Observational", method2_name = "Counterfactual", treat = quo(treat_num), pi = quo(pi), Y = quo(y), mu_true = quo(mu_count))
auc_df <- compute_ROC_df_dr(t_arr, test, "All", method1 = quo(mu_obs), method2 = quo(mu_count), method1_name = "Observational", method2_name = "Counterfactual", treat = quo(treat_num), pi = quo(pi), Y = quo(y), mu_true = quo(mu_count))
auc_df$Evaluation <- "Doubly-robust"

auc_true <- compute_roc_df(test, quo(y0), quo(mu_count), quo(mu_obs), "All") %>%
  drop_na(Recall, FPR) %>%
  mutate(Evaluation = "True Counterfactual")

auc_control <- compute_roc_df(filter(test, treat_num == 0), quo(y), quo(mu_count), quo(mu_obs), "All") %>%
  drop_na(Recall, FPR) %>%
  mutate(Evaluation = "Control")

auc_obs <- compute_roc_df(test, quo(y), quo(mu_count), quo(mu_obs), "All") %>%
  drop_na(Recall, FPR) %>%
  mutate(Evaluation = "Observational")

auc_all <- rbind(auc_obs, auc_control, auc_true, auc_df)
auc_all$Evaluation <- factor(auc_all$Evaluation, levels = c("Observational", "Control", "Doubly-robust", "True Counterfactual"))
auc_all$Model <- auc_all$Method

auc_all %>% ggplot(aes(x = FPR, y = Recall, color = Model, fill = Model)) + geom_line() + facet_grid(. ~ Evaluation, labeller = as_labeller(evaluation_names_synth)) + theme_bw(base_size = 15) + scale_color_manual(values = method_colors) + scale_x_continuous(labels = scale_format) + scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), labels = c(0, .25, .5, .75, 1), limits = c(0, 1)) + coord_fixed(ratio = 1) + theme(legend.position = "none")
ggsave(paste(fig_folder, "roc/", "fig_seed_", sprintf("%03d", seed), ".png", sep = ""), width = 10, height = 3.6, dpi = 1000)

## mode:validation
#write.csv(auc_all, file = paste(data_folder, 'roc/', 'seed_', sprintf("%03d", seed), ".csv", sep = ""))