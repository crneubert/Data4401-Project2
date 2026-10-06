# Offline syntax check of the original baseline; no model fitting or downloads.
lines <- readLines("modeling/logistic_regression_baseline.qmd", warn = FALSE)
chunks <- list()
inside <- FALSE
for (line in lines) {
  if (grepl("^```\\{r", line)) {
    inside <- TRUE
    name <- sub("^```\\{r ([^,}]+).*", "\\1", line)
    chunks[[name]] <- character()
    next
  }
  if (inside && grepl("^```$", line)) {
    inside <- FALSE
    next
  }
  if (inside) chunks[[name]] <- c(chunks[[name]], line)
}
invisible(lapply(chunks, function(code) parse(text = code)))
cat("All R chunks parse successfully.\n")
