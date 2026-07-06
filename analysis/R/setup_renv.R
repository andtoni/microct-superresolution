#!/usr/bin/env Rscript

repos <- c(CRAN = "https://cloud.r-project.org")
required <- c(
  "renv",
  "readr",
  "ggplot2",
  "rstatix",
  "ggpubr",
  "ggprism",
  "broom",
  "emmeans",
  "effectsize",
  "patchwork",
  "knitr",
  "rmarkdown"
)

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv", repos = repos)
}

if (!file.exists("renv.lock")) {
  renv::init(bare = TRUE, restart = FALSE)
} else {
  renv::restore(prompt = FALSE)
}

missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  renv::install(missing)
}
renv::snapshot(prompt = FALSE)
