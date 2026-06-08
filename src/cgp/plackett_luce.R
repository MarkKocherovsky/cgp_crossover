library(PlackettLuce)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tibble)

setwd('C:/Users/mk245/Documents/R')
getwd()
input_dir <- "./plackett_input/may_14_26/"
#metric <- "min_test_fitnesses"  # or "best_sizes"
metric <- "min_test_fitnesses"  # or "best_sizes"
cat(list.files(path=input_dir))
cat(input_dir)

files <- list.files(path=input_dir, pattern = paste0("_", metric, "\\.csv$"), full.names = TRUE)
cat("📂 Found", length(files), "CSV files.\n")

# <problem>_1d_<xover_type>_<selection_type>_cfg<cfg_num>_<metric>.csv
pattern <- paste0("^(.+)_1d_(.+)_(.+)_cfg(\\d+)_", metric, "\\.csv$")

parse_file <- function(path) {
  fname <- basename(path)
  m <- regexec(pattern, fname)
  mm <- regmatches(fname, m)[[1]]
  if (length(mm) == 0) return(NULL)
  
  tibble(
    file = path,
    fname = fname,
    problem = mm[2],
    xover = mm[3],
    selection = mm[4],
    cfg = as.integer(mm[5]),
    cfg_label = paste0("cfg", mm[5])
  )
}

meta <- bind_rows(lapply(files, parse_file))
if (nrow(meta) == 0) stop("No files matched expected filename format.")

# Your CSVs look like a single numeric column (no header).
read_metric_vec <- function(path) {
  v <- suppressWarnings(read.csv(path, header = FALSE)[[1]])
  as.numeric(v)
}

fit_group_cfgs <- function(df_group, debug = FALSE) {
  cols <- list()
  
  for (i in seq_len(nrow(df_group))) {
    cfg_name <- df_group$cfg_label[i]
    v <- suppressWarnings(read.csv(df_group$file[i], header = FALSE)[[1]])
    cols[[cfg_name]] <- as.numeric(v)
  }
  
  wide <- as.data.frame(cols)
  
  # trim to shortest length (align rows)
  nmin <- min(sapply(wide, length))
  wide <- wide[seq_len(nmin), , drop = FALSE]
  
  X <- as.matrix(sapply(wide, as.numeric))
  
  # keep rows with at least 2 non-NA entries
  keep <- rowSums(!is.na(X)) >= 2
  if (debug) {
    cat("   raw X dims:", nrow(X), "x", ncol(X),
        " | rows kept:", sum(keep),
        " | rows dropped:", sum(!keep), "\n")
    cat("   NA per col:", paste(colSums(is.na(X)), collapse = ", "), "\n")
  }
  X <- X[keep, , drop = FALSE]
  
  if (ncol(X) < 2 || nrow(X) < 1) return(NULL)
  
  # -----------------------------
  # Convert fitness values -> ranks per row (1 = best)
  # lower fitness is better, so rank ascending
  # -----------------------------
  rank_mat <- t(apply(X, 1, function(row) rank(row, ties.method = "min")))
  
  # as.rankings wants ranks/orderings, not raw scores
  R <- tryCatch(as.rankings(rank_mat), error = function(e) NULL)
  if (is.null(R) || all(is.na(R))) return(NULL)
  
  model <- tryCatch(PlackettLuce(R), error = function(e) NULL)
  if (is.null(model)) return(NULL)
  
  w <- itempar(model)
  prob <- as.numeric(w / sum(w))
  names(prob) <- names(w)
  
  as_tibble_row(prob)
}



# -----------------------------
# Run: group by problem + xover (and selection), compare cfgs
# -----------------------------
results <- meta %>%
  group_by(problem, xover, selection) %>%
  group_modify(~{
    cat("\n🔧 Group:", .y$problem, "|", .y$xover, "|", .y$selection,
        "| cfgs:", paste(sort(.x$cfg_label), collapse = ", "), "\n")
    cat("   files:", paste(basename(.x$file), collapse = " | "), "\n")
    
    res <- fit_group_cfgs(.x, debug = TRUE)   # <-- debug on
    if (is.null(res)) return(tibble())
    res
  }) %>%
  ungroup()


if (nrow(results) == 0) {
  stop("No groups produced a valid Plackett–Luce fit. (Often: <2 cfg files per group.)")
}

print(results)

write.csv(results,
          file = paste0("plackett_probs_by_cfg_", metric, ".csv"),
          row.names = FALSE)

# -----------------------------
# Plot: cfg probabilities per group
# -----------------------------
results_long <- results %>%
  pivot_longer(cols = starts_with("cfg"),
               names_to = "cfg",
               values_to = "prob")

ggplot(results_long, aes(x = cfg, y = prob, group = interaction(problem, xover, selection))) +
  geom_point() +
  facet_grid(problem ~ xover, scales = "free_y") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Configuration",
       y = "Plackett–Luce probability",
       title = paste("Config preference by problem and xover (", metric, ")", sep = "")) +
  theme(axis.title.y = element_blank(),
        strip.text.y.left = element_text(angle=0),
        strip.placement = "outside") +
  ylim(0, 1.0)

