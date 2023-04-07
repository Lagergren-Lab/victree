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

plot_cell_assignment <- function(diag_list, gt = NULL, cell_sample_size = NA, nrow = 5, ncol = 5, remap_clones = FALSE) {
  ca_long_df <- melt(diag_list$cell_assignment,
    value.name = "prob",
    varnames = c("iter", "cell", "clone")
  ) %>%
    mutate(clone = clone - 1)

  p <- NULL
  if (!is.null(gt)) {
    if (remap_clones) {
    # change label names depending on gt_vi map
      clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = TRUE)
      ca_long_df <- ca_long_df %>%
        mutate(clone = recode(clone, !!!clone_map))
    }
    ca_long_df <- ca_long_df %>%
      mutate(cell = paste(as.character(cell - 1L), gt_list$cell_assignment[cell], sep = ":"))
  }
  ca_long_df <- ca_long_df %>%
    mutate(clone = factor(clone, levels = 0:(K - 1)))

  p <- ca_long_df %>%
    ggplot() +
    geom_line(aes(x = iter, y = prob, color = clone)) +
    labs(title = "Cell assignment to clones") +
    facet_wrap_paginate(~cell, ncol = ncol, nrow = nrow, scales = "free")

  p_list <- list()
  for (i in 1:n_pages(p)) {
    p_list[[i]] <- p +
      facet_wrap_paginate(~cell, ncol = ncol, nrow = nrow, scales = "free", page = i)
  }

  return(p_list)
}

plot_copy <- function(diag_list, gt_list = NULL, nrow = 5, remap_clones = FALSE) {
  li <- list()
  n_iter <- dim(diag_list$copy_num)[1]
  K <- dim(diag_list$copy_num)[2]

  clones <- 1:K
  if (remap_clones & !is.null(gt_list)) {
    clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = FALSE)
    # ca_long_df <- ca_long_df %>%
    #   mutate(clone = recode(clone, !!!clone_map))

    clones <- clone_map %>%
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
      labs(title = "Last iteration against ground truth") +
      facet_wrap(~clone, ncol = 1)
    li[[length(li) + 1]] <- p
  }

  return(li)
}

plot_eps <- function(diag_list, gt_list = NULL, remap_clones = FALSE, nrow = 5) {

  n_iter <- dim(diag_list$copy_num)[1]
  K <- dim(diag_list$copy_num)[2]

  eps_df_a <- diag_list$eps_a %>%
    melt(value.name = "a", varnames = c("it", "u", "v")) %>%
      filter(v != 1, u != v)
  eps_df <- diag_list$eps_b %>%
    melt(value.name = "b", varnames = c("it", "u", "v")) %>%
    filter(v != 1, u != v) %>%
    left_join(eps_df_a, by = join_by(it, u, v)) %>%
    mutate(u = u - 1, v = v - 1) %>%
    mutate(mm = a / (a + b), stdev = sqrt(a * b) / ((a + b) * sqrt(a + b + 1)))

  # if map is available, remap vi clone labels to match the gt ones
  if (remap_clones && !is.null(gt_list)) {
    clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = TRUE)
    eps_df <- eps_df %>%
      mutate(u = recode(u, !!!clone_map), v = recode(v, !!!clone_map))
  }

  # plot a/b params along iterations
  # two lines in same plot, two colors (one for a one for b)
  p <- eps_df %>%
    mutate(uv = paste(u, v, sep = ",")) %>%
    select(a, b, it, uv) %>%
    gather(key = "param", value = "value", a, b) %>% # to long format
    ggplot() +
      geom_line(aes(x = it, y = value, color = param)) +
      labs(title = "eps ~ Beta(a,b) vi params") +
      facet_wrap_paginate(~ uv, ncol = 3, nrow = nrow, scales = "free")

  p_list <- list()
  for (i in 1:n_pages(p)) {
    p_list[[i]] <- p +
    facet_wrap_paginate(~ uv, ncol = 3, nrow = nrow, scales = "free", page = i)
  }

  # get gt tree edges
  if (!is.null(gt_list)) {
    # split two dataframes
    # first with gt data for edges in tree
    vigt_df <- gt_list$eps %>%
      melt(value.name = "gt_eps", varnames = c("u", "v")) %>%
      filter(v != 1, u != v) %>%
      mutate(u = u - 1, v = v - 1) %>% # 0-based index
      left_join(eps_df, by = join_by(u, v))

    gt_df <- vigt_df %>%
      filter(gt_eps > 0)

    # second without gt edges
    eps_df <- vigt_df %>%
      filter(gt_eps == 0) %>% # only select edges not in tree
      select(it, u, v, mm, stdev)

    # plot the ground truth edges
    gt_plot <- gt_df %>%
      mutate(uv = paste(u, v, sep = ",")) %>%
      ggplot(aes(x = it)) +
        geom_line(aes(y = mm, color = "vi")) +
        geom_line(aes(y = gt_eps, color = "gt")) +
        geom_ribbon(aes(ymin = mm - stdev, ymax = mm + stdev), alpha = 0.3) +
        labs(title = "VI eps mean+stdev with ground truth") +
        facet_wrap_paginate(~ uv, ncol = 1, nrow = nrow, scales = "free")

    for (i in 1:n_pages(gt_plot)) {
      p <- gt_plot +
        facet_wrap_paginate(~ uv, ncol = 1, nrow = nrow, scales = "free", page = i)
      p_list[[length(p_list) + 1]] <- p
    }
  }

  # plot rest of edges, or all of them if no gt is av.
  eps_mean_plot <- eps_df %>%
    mutate(uv = paste(u, v, sep = ",")) %>%
    select(it, mm, stdev, uv) %>%
    ggplot(aes(x = it)) +
      geom_line(aes(y = mm)) +
      geom_ribbon(aes(ymin = mm-stdev, ymax = mm+stdev), alpha = 0.3) +
      labs(title = "VI eps mean+stdev") +
      facet_wrap_paginate(~ uv, ncol = 1, nrow = nrow, scales = "free")

  for (i in 1:n_pages(eps_mean_plot)) {
    p <- eps_mean_plot +
      facet_wrap_paginate(~ uv, ncol = 1, nrow = nrow, scales = "free", page = i)
    p_list[[length(p_list) + 1]] <- p
  }

  return(p_list)
}

plot_mutau <- function(diag_list, gt_list = NULL) {
  N <- dim(diag_list$nu)[2]

  nu_df <- melt(diag_list$nu, value.name = "nu", varnames = c("it", "cell"))
  lambda_df <- melt(diag_list$lambda, value.name = "lambda", varnames = c("it", "cell"))
  alpha_df <- melt(diag_list$alpha, value.name = "alpha", varnames = c("it", "cell"))
  beta_df <- melt(diag_list$beta, value.name = "beta", varnames = c("it", "cell"))

  mutau_df <- nu_df %>%
    left_join(lambda_df, by = join_by(it, cell)) %>%
    left_join(alpha_df, by = join_by(it, cell)) %>%
    left_join(beta_df, by = join_by(it, cell)) %>%
    mutate(tau_mean = alpha / beta, tau_sd = sqrt(alpha) / beta) %>%
    mutate(mu_mean = nu, mu_sd = 1 / sqrt(lambda * tau_mean)) %>%
    gather(key = "key", value = "value", tau_mean, mu_mean, tau_sd, mu_sd) %>%
    mutate(param = ifelse(grepl("tau", key), "tau", "mu")) %>%
    mutate(stat = ifelse(grepl("mean", key), "mean", "sd")) %>%
    select(-key) %>%
    spread(stat, value)

  # add gt data to dataframe
  if (!is.null(gt_list)) {
    gt_mudf <- tibble(cell = 1:N, gt = gt_list$mu, param = "mu")
    gt_df <- tibble(cell = 1:N, gt = gt_list$tau, param = "tau") %>%
      bind_rows(gt_mudf)
    mutau_df <- mutau_df %>%
      left_join(gt_df, by = join_by(cell, param))
  }

  p <- mutau_df %>%
    ggplot(aes(x = it)) +
      geom_line(aes(y = mean)) +
      geom_ribbon(aes(ymin = mean-sd, ymax = mean+sd), alpha = 0.3)

  # draw gt red line
  if (!is.null(gt_list)) {
    p <- p +
      geom_line(aes(y = gt, color = "gt"))
  }

  p <- p +
    labs(title = "VI mu, tau ~ NormalGamma() expectations for each cell") +
    facet_wrap_paginate(cell ~ param, nrow = 10, ncol = 2, scales = "free_y",
                        strip.position = "right", labeller = label_wrap_gen(multi_line=FALSE))

  p_list <- list()
  for (i in 1:n_pages(p)) {
    p_list[[i]] <- p +
      facet_wrap_paginate(cell ~ param, nrow = 10, ncol = 2, scales = "free_y",
                          strip.position = "right", labeller = label_wrap_gen(multi_line=FALSE), page = i)
  }

  return(p_list)
}


# GROUND TRUTH ONLY

get_confusion_mat <- function(diag_list, gt_list) {
  K <- dim(diag_list$cell_assignment)[3]
  N <- dim(diag_list$cell_assignment)[2]
  n_iter <- dim(diag_list$cell_assignment)[1]

  prop_df <- diag_list$cell_assignment[n_iter,,] %>%
    melt(value.name = "prob", varnames = c("cell", "vi")) %>%
    mutate(vi = vi - 1) %>% # switch to 0-based clone names
    left_join(tibble(gt = gt_list$cell_assignment, cell = 1:N), by = join_by(cell)) %>%
    group_by(vi, gt) %>%
    summarise(prev = sum(prob)) %>%
    mutate(sum_prev = sum(prev)) %>%
    mutate(prop = ifelse(sum_prev > 0, prev / sum_prev, 0)) %>%
    ungroup() %>%
    select(vi, gt, prop)

  return(prop_df)
}

get_clone_map <- function(diag_list, gt_list, as_named_vector = FALSE) {
  clone_map <- get_confusion_mat(diag_list, gt_list) %>%
    group_by(vi) %>%
    filter(prop == max(prop)) %>%
    ungroup()

  if (as_named_vector) {
    named_vec <- clone_map$gt
    names(named_vec) <- clone_map$vi
    clone_map <- named_vec
  }

  return(clone_map)
}

plot_ari_heatmap <- function(diag_list, gt_list) {
  require(aricode)
  n_iter <- dim(diag_list$cell_assignment)[1]
  N <- dim(diag_list$cell_assignment)[2]
  K <- dim(diag_list$cell_assignment)[3]

  ari <- c()
  for (it in 1:n_iter) {
    c2 <- apply(diag_list$cell_assignment[it,,], 1, which.max) - 1L
    ari <- append(ari, ARI(gt_list$cell_assignment, c2))
  }

  ari_df <- tibble(ari = ari, it = 1:n_iter)

  ari_plot <- ggplot(ari_df) +
    geom_line(aes(it, ari)) +
    labs(title = "Adjusted Rand Index")

  conf_mat <- get_confusion_mat(diag_list, gt_list)

  conf_mat_heatmap <- conf_mat %>%
    ggplot(aes(x = gt, y = vi)) +
      geom_tile(aes(fill = prop)) +
      geom_text(aes(label = round(prop, 3))) +
      theme(legend.position = "none", panel.background = element_blank()) +
      labs(title = "Cell proportion in gt/vi labels")

  gt_vi_map <- get_clone_map(diag_list, gt_list, as_named_vector = T)

  info_text <- ggparagraph(paste("In all following plots, vi clone labels are re-mapped",
                                 "so to match the gt labels according to the table below (heatmap)",
                                 "e.g clone", 2, "->", gt_vi_map[2], ", clone",
                                 3, "->", gt_vi_map[3], "etc."))

  p <- ggarrange(info_text, ari_plot, conf_mat_heatmap, ncol = 1)
  return(p)
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

if (!is.null(gt_list)) {
  plot_ari_heatmap(diag_list, gt_list)
}

# qz
plot_cell_assignment(diag_list, gt = gt_list, remap_clones = TRUE)

# qc
plot_copy(diag_list, gt_list, remap_clones = TRUE)

# eps
plot_eps(diag_list, gt_list, remap_clones = TRUE)

# mutau
plot_mutau(diag_list, gt_list)

graphics.off()
