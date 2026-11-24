# Within each system, compare the 30 independent replicate means, while
# allowing serially correlated frame-level residuals within each replicate.
# This is a FIXED-replicate GLS model, not a random-intercept mixed model.
# Run after the companion Python export script:
#   Rscript run_within_system_hbond_GLS.R

library(nlme)

outdir <- "GLS_within_system"
dat <- read.csv(file.path(outdir, "frame_level_mean_hbonds.csv"))
stopifnot(nrow(dat) == 4800, all(is.finite(dat$mean_hbond)))

models_out <- list()
tests_out <- list()
variance_out <- list()

for (system_name in c("GCU", "CGU")) {
  d <- subset(dat, system == system_name)
  d$replicate <- factor(d$replicate)
  d <- d[order(d$replicate, d$frame), ]
  stopifnot(nlevels(d$replicate) == 30, nrow(d) == 2400)

  # Compare alternative residual correlation structures using identical data
  # and identical fixed effects (so REML AIC comparisons are meaningful).
  fit <- function(corr = NULL) {
    opts <- list(model = mean_hbond ~ replicate, data = d, method = "REML",
                 control = glsControl(maxIter = 100, msMaxIter = 100))
    if (!is.null(corr)) opts$correlation <- corr
    tryCatch(do.call(gls, opts), error = function(e) {
      message("Model failed: ", conditionMessage(e))
      NULL
    })
  }
  all_fits <- list(
    independent = fit(),
    AR1 = fit(corAR1(form = ~ frame | replicate)),
    AR2 = fit(corARMA(form = ~ frame | replicate, p = 2, q = 0)),
    ARMA11 = fit(corARMA(form = ~ frame | replicate, p = 1, q = 1))
  )

  fits <- Filter(Negate(is.null), all_fits)
  if (!length(fits)) stop("No covariance model converged for ", system_name)
  comparison <- data.frame(
    system = system_name,
    correlation = names(fits),
    AIC = vapply(fits, AIC, numeric(1)),
    BIC = vapply(fits, BIC, numeric(1))
  )
  comparison <- comparison[order(comparison$AIC), ]
  models_out[[system_name]] <- comparison
  best_name <- comparison$correlation[1]
  best <- fits[[best_name]]

  cat("\n", system_name, "model comparison:\n", sep = "")
  print(comparison, row.names = FALSE)
  cat("\nOmnibus replicate test, lowest-AIC covariance model:", best_name, "\n")
  wald <- as.data.frame(anova(best, type = "marginal"))
  print(wald)
  wald$term <- rownames(wald)
  wald$system <- system_name
  wald$correlation <- best_name
  tests_out[[system_name]] <- wald[wald$term == "replicate", , drop = FALSE]

  means <- tapply(d$mean_hbond, d$replicate, mean)
  variances <- tapply(d$mean_hbond, d$replicate, var)
  within <- mean(variances)
  between <- var(means)
  variance_out[[system_name]] <- data.frame(
    system = system_name,
    mean_within_replicate_variance = within,
    between_replicate_mean_variance = between,
    between_over_within = between / within
  )

  # Correlation diagnostics. nlme::plot.ACF uses lattice graphics, which
  # does not auto-print in Rscript and does not respect base par(mfrow).
  # Draw the underlying ACF estimates with base R instead.
  acf_before <- ACF(fits[["independent"]], resType = "response",
                    form = ~ frame | replicate, maxLag = 20)
  acf_after <- ACF(best, resType = "normalized",
                   form = ~ frame | replicate, maxLag = 20)
  plot_acf_base <- function(x, title) {
    keep <- is.finite(x$lag) & is.finite(x$ACF) & x$lag > 0
    if (!any(keep)) stop("No finite ACF values available for ", title)
    plot(x$lag[keep], x$ACF[keep], type = "h", lwd = 2,
         xlab = "Lag (frames)", ylab = "Residual ACF",
         ylim = c(-1, 1), main = title)
    abline(h = 0, col = "gray50", lty = 2)
  }
  png(file.path(outdir, paste0(system_name, "_ACF_before_after.png")),
      width = 1200, height = 500, res = 130)
  par(mfrow = c(1, 2), mar = c(4, 4, 4, 1))
  plot_acf_base(acf_before, paste(system_name, "before correlation adjustment"))
  plot_acf_base(acf_after, paste(system_name, "normalized residuals:", best_name))
  dev.off()

  # Plot all 30 time series, to check drift/nonstationarity.
  traces <- sapply(split(d$mean_hbond, d$replicate), identity)
  png(file.path(outdir, paste0(system_name, "_replicate_traces.png")),
      width = 1200, height = 650, res = 130)
  matplot(1:80, traces, type = "l", lty = 1,
          col = adjustcolor("gray45", alpha.f = 0.4),
          xlab = "Retained frame", ylab = "Mean H-bond count per feature",
          main = paste(system_name, "30 replicate time series"))
  lines(1:80, rowMeans(traces), col = "black", lwd = 3)
  dev.off()
}

write.csv(do.call(rbind, models_out),
          file.path(outdir, "01_covariance_model_comparison.csv"), row.names = FALSE)
write.csv(do.call(rbind, tests_out),
          file.path(outdir, "02_omnibus_GLS_Wald_F.csv"), row.names = FALSE)
write.csv(do.call(rbind, variance_out),
          file.path(outdir, "03_within_between_variance.csv"), row.names = FALSE)
cat("\nSaved results to", normalizePath(outdir), "\n")
cat("Inspect both ACF plots and replicate traces before trusting any p-value.\n")
