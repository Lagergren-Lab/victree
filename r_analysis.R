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

plot_cell_assignment <- function(diag_list, gt = NULL, cell_sample_size = NA, nrow = 5, ncol = 5, gtvi_map = NULL) {
  ca_long_df <- melt(diag_list$cell_assignment,
    value.name = "prob",
    varnames = c("iter", "cell", "clone")
  ) %>%
    mutate(clone = clone - 1)

  # change label names depending on gt_vi map
  if (!is.null(gtvi_map)) {
    ca_long_df <- ca_long_df %>%
      left_join(gtvi_map, by = dplyr::join_by(clone == gt)) %>%
      select(-clone) %>%
      rename(clone = vi)
  }

  p <- NULL
  if (is.null(gt)) {
    if (is.na(cell_sample_size)) {
      cell_sample_size <- N
    }

    ca_long_df <- ca_long_df %>%
      filter(cell %in% sample(1:N, cell_sample_size))
  } else {
    ca_long_df <- ca_long_df %>%
      mutate(cell = paste(as.character(cell - 1L), gt_list$cell_assignment[cell], sep = ":"))
  }
  ca_long_df <- ca_long_df %>%
    mutate(clone = factor(clone, levels = 0:(K - 1)))

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

plot_copy <- function(diag_list, gt_list = NULL, nrow = 5, gtvi_map = NULL) {
  li <- list()
  n_iter <- dim(diag_list$copy_num)[1]
  K <- dim(diag_list$copy_num)[2]

  clones <- 1:K
  if (!is.null(gtvi_map)) {
    clones <- gtvi_map %>%
      arrange(gt) %>%
      mutate(vi = vi + 1) %>%
      pull(vi)
  }

  for (k in clones) {
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
    copy_var <- apply(diag_list$copy_num[n_iter, clones, , ], c(1, 2), which.max) %>%
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

plot_eps <- function(diag_list, gt_list = NULL, gtvi_map = NULL) {

  # get gt tree edges
  if (!is.null(gt_list)) {
    edges <- which(gt_list$eps > 0, arr.ind = TRUE)
  } else {
    # all edges
    edges <- expand.grid(u = 1:K, v = 1:K) %>%
      filter(v != 1, u != v)
  }
  n_iter <- dim(diag_list$copy_num)[1]
  K <- dim(diag_list$copy_num)[2]

  eps_df_a <- diag_list$eps_a %>%
    melt(value.name = "a", varnames = c("iter", "u", "v")) %>%
      filter(v != 1, u != v)
  eps_df <- diag_list$eps_b %>%
    melt(value.name = "b", varnames = c("iter", "u", "v")) %>%
      filter(v != 1, u != v) %>%
      left_join(eps_df_a, by = join_by(iter, u, v)) # %>%
  # TODO: continue


  plot_list <- list()
  idx <- 1L
  for (e in 1:(K-1)) {
    u <- edges[e, 1]
    v <- edges[e, 2]
    eps_df <- tibble(
      it = 1:n_iter,
      a = diag_list$eps_a[, u, v], 
      b = diag_list$eps_b[, u, v],
      mm = a / (a + b), # eps mean (beta dist)
      stdev = sqrt(a * b) / ((a + b) * sqrt(a + b + 1)))
    if (!is.null(gt_list)) {
      eps_df$gt <- rep(gt_list$eps[u, v], n_iter)
    }

    eps_ab_plot <- eps_df %>%
      select(a, b, it) %>%
      gather(key = "param", value = "value", a, b) %>%
      ggplot() +
      geom_line(aes(x = it, y = value)) +
      facet_wrap(~ param, nrow = 2)

    eps_mean_plot <- eps_df %>%
      select(it, mm, stdev, gt) %>%
      ggplot(aes(x = it)) +
        geom_line(aes(y = mm)) +
        geom_line(aes(y = gt), color = "red") +
        geom_ribbon(aes(ymin = mm-stdev, ymax = mm+stdev), alpha = 0.3) +
        labs(title = paste0("edge (", u, ",", v, ")"))
    plot_list[[idx]] <- eps_ab_plot
    plot_list[[idx + K-1]] <- eps_mean_plot
    idx <- idx + 1
  }

  ggarrange(plotlist = plot_list, nrow = K-1)

}

# GROUND TRUTH ONLY

plot_ari <- function(diag_list, gt_list) {
  require(aricode)

  ari <- c()
  for (it in 1:n_iter) {
    c2 <- apply(diag_list$cell_assignment[it,,], 1, which.max) - 1L
    ari <- append(ari, ARI(gt_list$cell_assignment, c2))
  }

  ari_df <- tibble(ari = ari, it = 1:n_iter)

  ari_plot <- ggplot(ari_df) +
    geom_line(aes(it, ari))

  # get matchings and likely mapping from gt to inferred clones

  matchings <- matrix(rep(0, K^2), nrow = K)
  final_ca <- apply(diag_list$cell_assignment[101,,], 1, which.max) - 1L
  for (n in 1:N) {
    a <- gt_list$cell_assignment[n] + 1
    b <- final_ca[n] + 1
    matchings[a, b] <- matchings[a, b] + 1
  }

  conf_mat <- matchings %>%
    melt(value.name = "count", varnames = c("gt", "vi")) %>%
    mutate(gt = gt - 1, vi = vi - 1) %>%  # switch to 0-based clone names
    group_by(gt) %>%                      # for each clone 
    mutate(prop = count / sum(count)) # find proportions of vi matches

  conf_mat_heatmap <- conf_mat %>%
    ggplot(aes(gt, vi)) +
      geom_tile(aes(fill = prop)) +
      geom_text(aes(label = prop)) +
      theme(legend.position = "none")

  gt_vi_map <- conf_mat %>%
    filter(prop == max(prop)) %>%         # get the max proportion
    select(gt, vi, prop)

    p <- ggarrange(ari_plot, conf_mat_heatmap)
  print(p)
  return(gt_vi_map)
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

gt_vi_map <- NULL
if (!is.null(gt_list)) {
  gt_vi_map <- plot_ari(diag_list, gt_list)
}

# qc
plot_copy(diag_list, gt_list, gtvi_map = gt_vi_map)

# eps
plot_eps(diag_list, gt_list, gtvi_map = gt_vi_map)

graphics.off()
