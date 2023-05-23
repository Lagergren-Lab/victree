#!/usr/bin/env Rscript

source("./diagnostics_lib.r")

map_cells_to_clones <- function(cell_reads, cell_assignment) {
  cell_reads_mapped <- cell_reads %>%
    left_join()

  obs_df <- obs %>%
    melt(value.name = "read", varnames = c("site", "cell")) %>%
    left_join(cell_assignment, by = join_by(cell))
}

plot_simul_copy <- function(simul, obs, cells_per_clone=10, nbins=100) {
  N <- dim(obs)[2] # rhdf reads dims in opposite order!
  M <- dim(obs)[1]
  binsize <- M %/% nbins # integer division
  loc <- binsize %/% 2

  cell_clone_map <- tibble(clone = simul$cell_assignment, cell = 1:N)

  # extract a smaller sample of the cells
  cells <- 1:N
  if (N > cells_per_clone) {
  cells <- cell_clone_map %>%
    group_by(clone) %>%
    sample_n(cells_per_clone) %>%
    ungroup() %>%
    pull(cell)
  }

  cell_reads_df <- obs %>%
    melt(value.name = "read", varnames = c("site", "cell")) %>%
    filter(cell %in% cells) %>%
    left_join(cell_clone_map, by = join_by(cell)) %>% # map to clones
    # mutate(clone = simul$cell_assignment[cell]) %>%
    mutate(bin = (site %/% binsize) * binsize + loc)#  %>% # change the site to mid of bin
    # group_by(clone, bin) %>% # for each bin (in each clone separately)
    # summarize(read = median(read)) %>% # average the read to the cells belonging to that clone
    # ungroup()
    
  copy_simul <- simul$copy %>%
    melt(value.name = "cn", varnames = c("clone", "site")) %>%
    mutate(clone = clone - 1L)

  p <- copy_simul %>%
    ggplot() +
    geom_point(data = cell_reads_df, mapping = aes(bin, read), color = 2, alpha = 0.1) +
    geom_line(aes(x = site, y = cn), color = 1) +
    # scale_y_continuous(breaks = 1:A - 1L) +
    labs(title = "cn profile") +
    facet_wrap(~clone, ncol = 1)

  return(p)
}

# TODO: add cell proportions

# plot_simul_cell_prop <- function(diag_list) {
#   n_iter <- dim(diag_list$qC$single_filtering_probs)[1]
#   K <- dim(diag_list$qZ$pi)[3]
#   p <- apply(diag_list$qZ$pi[n_iter,,], 1, which.max) %>%
#     as_tibble_col(column_name = "clone") %>%
#     mutate(clone = clone - 1) %>%
#     mutate(clone = factor(clone, levels = 0:(K - 1))) %>%
#     ggplot(aes(y = clone)) +
#       geom_bar(aes(fill = clone))
#   return(p)
# }

parser <- ArgumentParser(description = "draw data stats in pdf")
parser$add_argument("simul_h5_file", nargs=1, type = "character", help="simulated h5 file with gt group")


args <- parser$parse_args()
gt <- read_h5_gt(args$simul_h5_file)
obs <- read_h5_obs(args$simul_h5_file)

pdf_path <- file.path(paste0(args$simul_h5_file, ".pdf"))
pdf(pdf_path, onefile = TRUE)
plot_simul_copy(gt, obs)

graphics.off()

