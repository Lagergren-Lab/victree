library(reticulate)
library(ggplot2)
library(ggpubr)
library(ggforce) # for facet_wrap_paginate
library(tibble)
library(reshape2)
library(dplyr)
library(tidyr) # gather()


read_diagnostics <- function(dir_path, convert = TRUE) {
  np <- import("numpy", convert = convert)
  # read all diagnostics files
  out <- list(
    copy_num = np$load(file.path(dir_path, "copy.npy")),
    cell_assignment = np$load(file.path(dir_path, "cell_assignment.npy")),
    pi = np$load(file.path(dir_path, "pi.npy")),
    eps_a = np$load(file.path(dir_path, "eps_a.npy")),
    eps_b = np$load(file.path(dir_path, "eps_b.npy")),
    nu = np$load(file.path(dir_path, "nu.npy")),
    lambda = np$load(file.path(dir_path, "lambda.npy")),
    alpha = np$load(file.path(dir_path, "alpha.npy")),
    beta = np$load(file.path(dir_path, "beta.npy")),
    elbo = np$load(file.path(dir_path, "elbo.npy"))
  )
  return(out)
}

read_gt <- function(dir_path, convert = TRUE) {
  np <- import("numpy", convert = convert)
  # read all diagnostics files
  gt_out <- list(
    copy_num = np$load(file.path(dir_path, "copy.npy")),
    cell_assignment = as.integer(np$load(file.path(dir_path, "cell_assignment.npy"))),
    pi = np$load(file.path(dir_path, "pi.npy")),
    eps = np$load(file.path(dir_path, "eps.npy")),
    mu = np$load(file.path(dir_path, "mu.npy")),
    tau = np$load(file.path(dir_path, "tau.npy"))
  )
  return(gt_out)
}

plot_elbo <- function(diag_list) {
  elbo_df <- tibble(it = 1:length(diag_list$elbo), elbo = diag_list$elbo)
  p <- ggplot(elbo_df) +
    geom_line(aes(it, elbo)) +
    labs(title = "ELBO") +
    theme(aspect.ratio = .5)
  return(p)
}

plot_cell_assignment <- function(diag_list, gt = NULL, cell_sample_size = NA, nrow = 5, ncol = 5) {
  ca_long_df <- melt(diag_list$cell_assignment,
    value.name = "prob",
    varnames = c("iter", "cell", "clone")
  )
  p <- NULL
  if (is.null(gt)) {
    if (is.na(cell_sample_size)) {
      cell_sample_size <- N
    }

    ca_long_df <- ca_long_df %>%
      filter(cell %in% sample(1:N, cell_sample_size))
  } else {
    # TODO: fix gt clone names
    ca_long_df <- ca_long_df %>%
      mutate(cell = paste(as.character(cell - 1L), gt_list$cell_assignment[cell], sep = ":"))
  }
  ca_long_df <- ca_long_df %>%
    mutate(clone = factor(clone - 1L, levels = 0:(K - 1)))

  p <- ca_long_df %>%
    ggplot() +
    geom_line(aes(x = iter, y = prob, color = clone)) +
    facet_wrap_paginate(~cell, ncol = ncol, nrow = nrow, scales = "free")

  p_list <- list()
  for (i in 1:n_pages(p)) {
    p_list[[i]] <- p +
      facet_wrap_paginate(~cell, ncol = ncol, nrow = nrow, scales = "free", page = i)
  }

  return(p_list)
}

plot_copy <- function(diag_list, gt_list = NULL, nrow = 5) {
  li <- list()
  n_iter <- dim(diag_list$copy_num)[1]
  K <- dim(diag_list$copy_num)[2]

  for (k in 1:K) {
    steps <- seq(1, n_iter, length.out = nrow)
    copy_k <- apply(diag_list$copy_num[, k, , ], c(1, 2), which.max) %>%
      melt(value.name = "cn", varnames = c("iter", "site")) %>%
      filter(iter %in% steps) %>%
      mutate(cn = cn - 1)
    p <- ggplot(copy_k) +
      geom_line(aes(x = site, y = cn)) +
      labs(title = paste("CN clone", k-1)) +
      scale_y_continuous(breaks = 1:A - 1L) +
      facet_wrap(~iter, ncol = 1)

    li[[k]] <- p
  }

  # ground truth page
  if (!is.null(gt_list)) {
    copy_var <- apply(diag_list$copy_num[n_iter, , , ], c(1, 2), which.max) %>%
      melt(value.name = "vi", varnames = c("clone", "site")) %>%
      mutate(vi = vi - 1)
    copy_gt <- gt_list$copy_num %>%
      melt(value.name = "gt", varnames = c("clone", "site")) %>%
      mutate(gt = gt)
    p <- left_join(copy_gt, copy_var) %>%
      gather(key = "kind", value = "cn", gt, vi) %>%
      ggplot() +
      geom_line(aes(x = site, y = cn, color = kind)) +
      geom_line(aes(x = site, y = cn, color = kind, linetype = kind)) +
      scale_y_continuous(breaks = 1:A - 1L) +
      facet_wrap(~clone, ncol = 1)
    li[[length(li) + 1]] <- p
  }

  return(li)
}

# set up variables
copytree_path <- "/Users/zemp/phd/scilife/coPyTree"
diag_path <- file.path(copytree_path, "output", "diagnostics")
gt_path <- file.path(copytree_path, "datasets", "gt_simul_K4_A5_N100_M500")
# gt_path <- NA
pdf_path <- "./results.pdf"

pdf(pdf_path, onefile = TRUE, paper = "a4")

if (!is.na(gt_path)) {
  gt_list <- read_gt(gt_path)
}
diag_list <- read_diagnostics(diag_path)

n_iter <- dim(diag_list$copy_num)[1]
K <- dim(diag_list$copy_num)[2]
M <- dim(diag_list$copy_num)[3]
A <- dim(diag_list$copy_num)[4]
N <- dim(diag_list$cell_assignment)[2]

# elbo

plot_elbo(diag_list)

# qz

plot_cell_assignment(diag_list, gt = gt_list)

# qc
plot_copy(diag_list, gt_list)

graphics.off()
