library(reticulate)
library(ggplot2)
library(ggpubr)
library(ggforce) # for facet_wrap_paginate
library(tibble)
library(reshape2)
library(dplyr)
library(tidyr) # gather()
library(rhdf5)

suppressPackageStartupMessages(library("argparse"))

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
    elbo = np$load(file.path(dir_path, "elbo.npy")),
    trees = np$load(file.path(dir_path, "tree_samples.npy")),
    tree_weights = np$load(file.path(dir_path, "tree_weights.npy")),
    tree_mat = np$load(file.path(dir_path, "tree_matrix.npy"))
  )
  return(out)
}

read_h5_checkpoint <- function(h5_file_path) {
  h5 <- H5Fopen(h5_file_path, "H5F_ACC_RDONLY")
  group_names <- h5ls(h5, recursive = FALSE)$name
  
  # initialize an empty list
  h5_list <- vector("list", length(group_names))
  
  # iterate through group names
  for (i in seq_along(group_names)) {
    group_name <- group_names[[i]]
    
    # open the current group
    group <- H5Gopen(h5, group_name)
    # get a list of dataset names in the current group
    dataset_names <- h5ls(group, recursive = FALSE)$name
    # initialize an empty list for the datasets
    dataset_list <- vector("list", length(dataset_names))
    # iterate through dataset names
    for (j in seq_along(dataset_names)) {
      dataset_name <- dataset_names[[j]]
      
      # read the current dataset
      dataset <- h5read(group, dataset_name)
      # transpose the array because h5read reads the data
      # in inverse order for efficiency reasons
      dataset_list[[j]] <- aperm(dataset, length(dim(dataset)):1)
    }
    
    # assign the dataset list to the h5 list with the current group name as a key
    h5_list[[i]] <- setNames(dataset_list, dataset_names)
    
    H5Gclose(group)
  }
  H5Fclose(h5)
  return(setNames(h5_list, group_names))
}

read_h5_gt <- function(h5_file_path) {
  h5 <- H5Fopen(h5_file_path, "H5F_ACC_RDONLY")
  gt_group <- H5Gopen(h5, "gt")
  # select the ground truth group of simulated data
  dataset_names<- h5ls(gt_group, recursive = FALSE)$name
  
  h5_list <- vector("list", length(dataset_names))
  
  for (i in seq_along(dataset_names)) {
    dataset_name <- dataset_names[[i]]
    
    dataset <- h5read(gt_group, dataset_name)
    h5_list[[i]] <- aperm(dataset, length(dim(dataset)):1)
  }
  H5Gclose(gt_group)
  H5Fclose(h5)
  return(setNames(h5_list, dataset_names))
}

read_h5_obs <- function(h5_file_path) {
  h5 <- H5Fopen(h5_file_path, "H5F_ACC_RDONLY")
  gt_group <- H5Gopen(h5, "layers")
  obs_arr <- h5read(gt_group, "copy")
  H5Gclose(gt_group)
  H5Fclose(h5)
  return(obs_arr)
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
  elbo_df <- tibble(it = 1:length(diag_list$VarTreeJointDist$elbo), elbo = diag_list$VarTreeJointDist$elbo)
  p <- ggplot(elbo_df) +
    geom_line(aes(it, elbo)) +
    labs(title = "ELBO") +
    theme(aspect.ratio = .5)
  return(p)
}

plot_cell_assignment <- function(diag_list, gt = NULL, nrow = 5, ncol = 5, remap_clones = FALSE, cpages = 0) {

  K <- dim(diag_list$qZ$pi)[3]
  N <- dim(diag_list$qZ$pi)[2]

  sampled_cells <- sample(1:N, nrow * ncol)
  ca_long_df <- melt(diag_list$qZ$pi[, sampled_cells, ],
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
  if (cpages < 1) {
    cpages <- n_pages(p)
  } else {
    cpages <- min(n_pages(p), cpages)
  }

  for (i in 1:cpages) {
    p_list[[i]] <- p +
      facet_wrap_paginate(~cell, ncol = ncol, nrow = nrow, scales = "free", page = i)
  }

  return(p_list)
}

plot_copy <- function(diag_list, gt_list = NULL, nrow = 5, remap_clones = FALSE) {
  li <- list()
  n_iter <- dim(diag_list$qC$single_filtering_probs)[1]
  K <- dim(diag_list$qC$single_filtering_probs)[2]
  A <- dim(diag_list$qC$single_filtering_probs)[4]

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
    copy_k <- apply(diag_list$qC$single_filtering_probs[, k, , ], c(1, 2), which.max) %>%
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
    copy_var <- apply(diag_list$qC$single_filtering_probs[n_iter, clones, , ], c(1, 2), which.max) %>%
      melt(value.name = "vi", varnames = c("clone", "site")) %>%
      mutate(clone = clone - 1, vi = vi - 1)
    copy_gt <- gt_list$copy %>%
      melt(value.name = "gt", varnames = c("clone", "site")) %>%
      mutate(clone = clone - 1)
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

  n_iter <- dim(diag_list$qC$single_filtering_probs)[1]
  K <- dim(diag_list$qC$single_filtering_probs)[2]

  eps_df_a <- diag_list$qEpsilonMulti$alpha %>%
    melt(value.name = "a", varnames = c("it", "u", "v")) %>%
      filter(v != 1, u != v)
  eps_df <- diag_list$qEpsilonMulti$beta %>%
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

plot_mutau <- function(diag_list, gt_list = NULL, cpages = 0) {
  N <- dim(diag_list$qMuTau$nu)[2]

  nu_df <- melt(diag_list$qMuTau$nu, value.name = "nu", varnames = c("it", "cell"))
  lambda_df <- melt(diag_list$qMuTau$lmbda, value.name = "lambda", varnames = c("it", "cell"))
  alpha_df <- melt(diag_list$qMuTau$alpha, value.name = "alpha", varnames = c("it", "cell"))
  beta_df <- melt(diag_list$qMuTau$beta, value.name = "beta", varnames = c("it", "cell"))

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

  if (cpages < 1) {
    cpages <- n_pages(p)
  } else {
    cpages <- min(n_pages(p), cpages)
  }

  for (i in 1:cpages) {
    p_list[[i]] <- p +
      facet_wrap_paginate(cell ~ param, nrow = 10, ncol = 2, scales = "free_y",
                          strip.position = "right", labeller = label_wrap_gen(multi_line=FALSE), page = i)
  }

  return(p_list)
}


# GROUND TRUTH ONLY

get_confusion_mat <- function(diag_list, gt_list) {
  K <- dim(diag_list$qZ$pi)[3]
  N <- dim(diag_list$qZ$pi)[2]
  n_iter <- dim(diag_list$qZ$pi)[1]

  prop_df <- diag_list$qZ$pi[n_iter,,] %>%
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
  n_iter <- dim(diag_list$qZ$pi)[1]
  N <- dim(diag_list$qZ$pi)[2]
  K <- dim(diag_list$qZ$pi)[3]

  ari <- c()
  for (it in 1:n_iter) {
    c2 <- apply(diag_list$qZ$pi[it,,], 1, which.max) - 1L
    ari <- append(ari, ARI(as.numeric(gt_list$cell_assignment), c2))
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

  if (remap_clones) {
    info_text <- ggparagraph(paste("In all following plots, vi clone labels are re-mapped",
                                   "so to match the gt labels according to the table below (heatmap)",
                                   "e.g clone", 2, "->", gt_vi_map[2], ", clone",
                                   3, "->", gt_vi_map[3], "etc."))
  } else {
    info_text <- ggparagraph(paste("Note: clones in ground truth might have different labels",
                                   "than those in VI results. Plots with ground truth comparison should be",
                                   "viewed under such consideration."))
  }

  p <- ggarrange(info_text, ari_plot, conf_mat_heatmap, ncol = 1)
  return(p)
}

plot_cell_prop <- function(diag_list) {
  n_iter <- dim(diag_list$qC$single_filtering_probs)[1]
  K <- dim(diag_list$qZ$pi)[3]
  p <- apply(diag_list$qZ$pi[n_iter,,], 1, which.max) %>%
    as_tibble_col(column_name = "clone") %>%
    mutate(clone = clone - 1) %>%
    mutate(clone = factor(clone, levels = 0:(K - 1))) %>%
    ggplot(aes(y = clone)) +
      geom_bar(aes(fill = clone))
  return(p)
}

plot_trees <- function(diag_list, nsamples = 10, gt_list = NULL, remap_clones = FALSE) {
  n_iter <- dim(diag_list$qT$trees_sample_newick)[1]
  K <- dim(diag_list$qZ$pi)[3]
  nsamples <- min(nsamples, length(unique(diag_list$qT$trees_sample_newick)))
  trees_df <- tibble(newick = diag_list$qT$trees_sample_newick[n_iter, ],
                     weight = diag_list$qT$trees_sample_weights[n_iter, ])

  str_clone_map <- as.character(1:K)
  names(str_clone_map) <- str_clone_map # identity
  if (remap_clones) {
    str_clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = TRUE)
    str_clone_map <- setNames(as.character(str_clone_map),
                              as.character(names(str_clone_map)))
  }
  # plot tree samples frequencies
  p <- trees_df %>%
    mutate(newick = recode(newick, !!!str_clone_map)) %>%  # TODO: check that it works
    group_by(newick) %>%
    summarise(weight = sum(weight)) %>%
    arrange(desc(weight)) %>%
    slice(1:nsamples) %>%
    ggplot(aes(x = weight, y = reorder(newick, weight))) +
      geom_col() +
      labs(title = "Tree samples at last iteration", x = "sum(weight)", y = "newick")

  return(p)
}

plot_tree_matrix <- function(diag_list) {

  long_mat <- diag_list$qT$weight_matrix %>%
    melt(value.name = "weight", varnames = c("iter", "u", "v"))
  
  p <- long_mat %>%
    mutate(u = u - 1, v = v - 1) %>% # 0-based index
    filter(iter == max(iter)) %>%
    select(-iter) %>%
    ggplot(aes(x = u, y = v)) +
      geom_raster(aes(fill = weight)) +
      geom_text(aes(label = round(weight, 2))) +
      labs(title = "Graph weight matrix at last it")

  return(p)
}


# plot data and copy number
plot_cn_obs <- function(diag_list, obs, gt_list = NULL, nrow = 5,
                        remap_clones = FALSE) {
  # obs is array M x N (sites x cells)
  li <- list()
  n_iter <- dim(diag_list$qC$single_filtering_probs)[1]
  K <- dim(diag_list$qC$single_filtering_probs)[2]
  A <- dim(diag_list$qC$single_filtering_probs)[4]

  # get clone map from vi to gt
  clones <- 1:K
  names(clones) <- clones
  if (remap_clones & !is.null(gt_list)) {
    clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = TRUE)
  }
  
  # get max prob cell assignment
  cell_assignment <- apply(diag_list$qZ$pi[n_iter,,], 1, which.max) %>%
    as_tibble_col(column_name = "clone") %>%
    mutate(cell = 1:nrow(.)) %>%
    mutate(clone = clone - 1) %>%
    mutate(clone = recode(clone, !!!clone_map))
  
  # match cells with cell assignment to clones
  obs_df <- obs %>%
    melt(value.name = "read", varnames = c("site", "cell")) %>%
    left_join(cell_assignment, by = join_by(cell))
  # print(head(obs_df))
  
  clones <- 1:K
  if (remap_clones & !is.null(gt_list)) {
    clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = FALSE)
    clones <- clone_map %>%
      arrange(gt) %>%
      mutate(vi = vi + 1) %>%
      pull(vi)
  }
  # get max prob copy number values
  copy_var <- apply(diag_list$qC$single_filtering_probs[n_iter, clones, , ],
                    c(1, 2), which.max) %>%
    melt(value.name = "vi", varnames = c("clone", "site")) %>%
    mutate(clone = clone - 1, vi = vi - 2)
  
  if (!is.null(gt_list)) {
    copy_gt <- gt_list$copy %>%
      melt(value.name = "gt", varnames = c("clone", "site")) %>%
      mutate(clone = clone - 1)
    copy_df <- left_join(copy_gt, copy_var) %>%
      gather(key = "type", value = "cn", gt, vi)
  } else {
    copy_df <- copy_var %>%
      mutate(type = "vi")
  }
  
  p <- copy_df %>%
    ggplot() +
#    geom_line(aes(x = site, y = cn, color = type)) +
    geom_line(aes(x = site, y = cn, color = type, linetype = type)) +
    geom_point(data = obs_df, mapping = aes(x = site, y = read), alpha = 0.008) +
    scale_y_continuous(breaks = seq(0, A, by = 2)) +
    labs(title = "Copy number and GC-corrected data") +
    facet_wrap(~clone, ncol = 1)
  
  return(p)
}
