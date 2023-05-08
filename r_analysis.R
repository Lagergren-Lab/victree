#!/usr/bin/env Rscript

source("~/phd/scilife/coPyTree/diagnostics_lib.r")
# arguments parsing
parser <- ArgumentParser(description = "draw diagnostics plots to pdf")
parser$add_argument("diag_dir", nargs=1, type = "character", help="directory with diagnostics files")
parser$add_argument("-gt", "--gt-dir", default = NULL, type = "character", help="directory with ground truth files")
parser$add_argument("-o", "--out-dir", type = "character", help = "directory where to save results.pdf", default = NULL)
parser$add_argument("-m", "--remap-clones", action = "store_true", default = FALSE,
                    help = "remap clones to most likely matches with ground truth")

args <- parser$parse_args()

# set up variables

# if (dir.exists(args$diag_dir)) {
#   diag_dir <- args$diag_dir
#   # diag_dir <- file.path(copytree_path, "output", "diagnostics")
# } else {
#   stop(paste0("Specified dir (", args$diag_dir, ") does not exist"))
# }

diag_list <- read_h5_checkpoint(args$diag_dir)
# diag_list <- read_diagnostics(diag_dir)
obs <- read_h5_obs(args$gt_dir) 

remap_clones <- FALSE
gt_list <- NULL
if (!is.null(args$gt_dir) && file.exists(args$gt_dir)) {
  # gt_dir <- file.path(copytree_path, "datasets", "gt_simul_K4_A5_N100_M500")
  # gt_list <- read_gt(args$gt_dir)
  gt_list <- read_h5_gt(args$gt_dir)
  gtK <- dim(gt_list$copy)[1]
  diagK <- dim(diag_list$qC$single_filtering_probs)[2]
  if (gtK == diagK) {
    remap_clones <- args$remap_clones
  } else if (args$remap_clones) {
      warning(paste("Clones cannot be remapped: ground truth number of clones differ",
              "from inferred ones (", gtK, "!=", diagK, ")"))
  }
}

pdf_dir <- file.path(dirname(args$diag_dir), "results")
if (!is.null(args$out_dir)) {
  pdf_dir <- args$out_dir
}
if (!dir.exists(pdf_dir)) {
  dir.create(pdf_dir)
}
pdf_path <- file.path(pdf_dir, "./plots.pdf")

pdf(pdf_path, onefile = TRUE, paper = "a4")


# elbo
ggarrange(plot_elbo(diag_list), plot_cell_prop(diag_list), ncol = 1)

# trees
plot_trees(diag_list, nsamples = 10, gt_list = gt_list, remap_clones = remap_clones)

plot_tree_matrix(diag_list)

# cell assignments
if (!is.null(gt_list)) {
  plot_ari_heatmap(diag_list, gt_list)
}

# if map is not 1-1, do not change labels
if (remap_clones) {
  clone_map <- get_clone_map(diag_list, gt_list, as_named_vector = TRUE)
  if (!all(sort(names(clone_map)) == sort(clone_map))) {
    warning(paste("Clones cannot be remapped: one or more gt clones",
                  "have not been reconstructed.", names(clone_map),
                  "->", as.numeric(clone_map)))
    remap_clones <- FALSE
  }
}

# qz
plot_cell_assignment(diag_list, gt = gt_list, remap_clones = remap_clones)

# qc
plot_copy(diag_list, gt_list, remap_clones = remap_clones)

plot_cn_obs(diag_list, obs, gt_list, remap_clones = T)

# eps
plot_eps(diag_list, gt_list, remap_clones = remap_clones)

# mutau
plot_mutau(diag_list, gt_list)


graphics.off()
