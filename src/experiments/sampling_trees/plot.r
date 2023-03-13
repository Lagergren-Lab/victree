library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
# TODO: add file read from command line

df <- read_csv("./sampling_trees_updated_qt.csv",
  col_types = cols(
    ...1 = col_double(),
    n_nodes = col_factor(),
    sample_size = col_double(),
    kl = col_double(),
    `time(s)` = col_double()
  ),
  col_select = -`...1`
)
p <- df %>%
  gather(key = "att", value = "meas", -c(n_nodes, sample_size)) %>%
  ggplot(aes(x = sample_size, y = meas)) +
  geom_line(aes(color = n_nodes)) +
  facet_grid(att ~ ., scales = "free_y")

ggsave("./sampling_updqt_plot_time_kl.png", plot = p)
