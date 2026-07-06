#!/usr/bin/env Rscript

required <- c("readr", "ggplot2", "rstatix", "ggpubr", "ggprism", "broom")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) > 0) {
  stop(
    "Missing R packages: ",
    paste(missing, collapse = ", "),
    "\nRun: Rscript analysis/R/setup_renv.R",
    call. = FALSE
  )
}

library(readr)
library(ggplot2)
library(rstatix)
library(ggpubr)
library(ggprism)
library(broom)

input <- "data/processed/example_measurements.csv"
output <- "figures/final/example_plot_r.pdf"
dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
dir.create("figures/final", recursive = TRUE, showWarnings = FALSE)

if (!file.exists(input)) {
  set.seed(20260706)
  df <- data.frame(
    group = rep(c("Control", "Treatment"), each = 12),
    value = c(rnorm(12, 1.0, 0.16), rnorm(12, 1.42, 0.16))
  )
  readr::write_csv(df, input)
}

df <- readr::read_csv(input, show_col_types = FALSE)
stats <- rstatix::t_test(df, value ~ group) |>
  rstatix::add_significance()

summary_df <- aggregate(value ~ group, df, function(x) {
  n <- length(x)
  se <- stats::sd(x) / sqrt(n)
  ci <- stats::qt(0.975, df = n - 1) * se
  c(mean = mean(x), ymin = mean(x) - ci, ymax = mean(x) + ci)
})
summary_df <- data.frame(
  group = summary_df$group,
  mean = summary_df$value[, "mean"],
  ymin = summary_df$value[, "ymin"],
  ymax = summary_df$value[, "ymax"]
)

plot <- ggplot(df, aes(group, value, colour = group)) +
  geom_jitter(width = 0.12, height = 0, size = 1.8, alpha = 0.9) +
  geom_errorbar(
    data = summary_df,
    aes(x = group, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    colour = "black",
    width = 0.18,
    linewidth = 0.35
  ) +
  geom_point(
    data = summary_df,
    aes(x = group, y = mean),
    inherit.aes = FALSE,
    colour = "black",
    size = 1.8
  ) +
  ggprism::theme_prism(base_size = 8) +
  labs(x = NULL, y = "Normalised response") +
  theme(legend.position = "none")

ggsave(output, plot, width = 85, height = 65, units = "mm", device = cairo_pdf)
readr::write_csv(stats, "data/processed/example_stats.csv")
