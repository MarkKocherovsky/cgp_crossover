install.packages("PlackettLuce")
library(PlackettLuce)

ranking <- read.csv("../output/graphs/min_test_fitnesses_rankings.csv")
R <- as.rankings(as.matrix(ranking))
model <- PlackettLuce(R)
w <- itempar(model)
p_best <- w / sum(w)
print(p_best)
