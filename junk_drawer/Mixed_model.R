# MDSA-tools: GCU versus CGU hydrogen-bond variance components
# Random intercept for each independently initialized MD replicate + AR(1)
# correlation between successive frames of the SAME replicate.
#
# Run: Rscript fit_hbond_mixed_model_AR1.R
# Requirements: install.packages(c("nlme", "reticulate"))
# Reticulate needs a Python installation with numpy.
#
# This is a SCALAR analysis of mean H-bond count across retained residue-pair
# features per frame. It does not quantify full-vector conformational diversity.

###########################
# User settings
###########################

# Set this to your folder containing the two .npy arrays if autodetection fails.
DATA_DIR <- ""

# Optional: set explicitly if reticulate cannot find the Python with NumPy.
# Example: "C:/Users/luisp/AppData/Local/Programs/Python/Python312/python.exe"
PYTHON_PATH <- ""

# TRUE: use first 80 post-equilibration frames of EVERY replicate (matches
# your original ANOVA). FALSE: use 80 frames for replicates 1-20 and 160 frames
# for replicates 21-30. Both options account for within-replicate correlation.
FIRST_80_ONLY <- TRUE

OUTPUT_DIR <- file.path("ANOVA_results", "mixed_model_AR1")

###########################
# Packages and file paths
###########################

for (pkg in c("nlme", "reticulate")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Install required R package: install.packages('", pkg, "')")
  }
}

home <- path.expand("~")
candidate_dirs <- unique(c(
  DATA_DIR,
  file.path(home, "Projects", "MDSA_tools_revision", "Trajectories_and_topologies"),
  file.path(home, "Projects", "MDSA_tools_revisions", "Trajectories_and_topologies"),
  "C:/Users/luisp/Projects/MDSA_tools_revision/Trajectories_and_topologies",
  "C:/Users/luisp/Projects/MDSA_tools_revisions/Trajectories_and_topologies",
  "C:/Users/Luis/Projects/MDSA_tools_revision/Trajectories_and_topologies",
  "C:/Users/Luis/Projects/MDSA_tools_revisions/Trajectories_and_topologies",
  file.path(getwd(), "Trajectories_and_topologies"),
  file.path(getwd(), "..", "Trajectories_and_topologies"),
  file.path(getwd(), "..", "..", "Trajectories_and_topologies"),
  getwd()
))
candidate_dirs <- candidate_dirs[nzchar(candidate_dirs)]

filename_pairs <- list(
  c("redone_CCU_GCU_Trajectory_array.npy",
    "redone_CCU_CGU_Trajectory_array.npy"),
  c("redone_unrestrained_CCU_GCU_Trajectory_array.npy",
    "redone_unrestrained_CCU_CGU_Trajectory_array.npy")
)

files_found <- NULL
for (dir in candidate_dirs) {
  for (pair in filename_pairs) {
    paths <- file.path(dir, pair)
    if (all(file.exists(paths))) {
      files_found <- normalizePath(paths, winslash = "/", mustWork = TRUE)
      break
    }
  }
  if (!is.null(files_found)) break
}

if (is.null(files_found)) {
  stop(
    "Could not find both .npy files. Set DATA_DIR near the top of this script.\n",
    "Expected either redone_CCU_{GCU,CGU}_Trajectory_array.npy or\n",
    "redone_unrestrained_CCU_{GCU,CGU}_Trajectory_array.npy."
  )
}

if (nzchar(PYTHON_PATH)) {
  reticulate::use_python(PYTHON_PATH, required = TRUE)
}

if (!reticulate::py_module_available("numpy")) {
  stop("NumPy unavailable in reticulate's Python. Check reticulate::py_config().")
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
cat("Reading arrays:\n", paste(files_found, collapse = "\n "), "\n")
cat("First 80 frames only:", FIRST_80_ONLY, "\n")

###########################
# Build one scalar H-bond summary per frame
###########################
# This uses NumPy for efficient reading of the existing .npy files. All
# statistical fitting and the output summaries below happen in R.
#
# The source arrays have frame x (residues+header) x (residues+header).
# Their top row and left column store residue labels. We exclude those headers,
# take the upper triangle (excluding diagonal), and filter interaction columns
# that were zero in EVERY frame of BOTH systems, as in your original workflow.
#
# The all-zero filter is built from ALL 3200 frames per system, EVEN when the
# statistical model is restricted to each replicate's first 80 frames.
# This reproduces the preprocessing order of your original ANOVA script.

reticulate::py_run_string("\nimport numpy as np\n\ndef build_hbond_frame_data(path_gcu, path_cgu, first_80_only=True):\n    lengths = [80] * 20 + [160] * 10\n    expected = sum(lengths)\n    arrays = [np.load(path_gcu, mmap_mode='r'),\n              np.load(path_cgu, mmap_mode='r')]\n\n    for system, arr in zip(('GCU', 'CGU'), arrays):\n        if arr.ndim != 3 or arr.shape[0] != expected:\n            raise ValueError(f'{system}: expected {expected} frames with 3 dimensions, got {arr.shape}')\n        if arr.shape[1] != arr.shape[2]:\n            raise ValueError(f'{system}: expected a square adjacency matrix for every frame')\n\n    if arrays[0].shape[1:] != arrays[1].shape[1:]:\n        raise ValueError('GCU and CGU use different matrix dimensions')\n    if not np.array_equal(arrays[0][0, 1:, 0], arrays[1][0, 1:, 0]):\n        raise ValueError('GCU and CGU have different residue-row labels/order')\n    if not np.array_equal(arrays[0][0, 0, 1:], arrays[1][0, 0, 1:]):\n        raise ValueError('GCU and CGU have different residue-column labels/order')\n\n    nres = arrays[0].shape[1] - 1\n    triu_r, triu_c = np.triu_indices(nres, k=1)\n    n_features = len(triu_r)\n    retained = np.zeros(n_features, dtype=bool)\n    rows = {'system': [], 'replicate_id': [], 'replicate': [],\n            'frame': [], 'hbond_total': []}\n\n    for system, arr in zip(('GCU', 'CGU'), arrays):\n        start = 0\n        for rep_number, rep_length in enumerate(lengths, start=1):\n            stop = start + rep_length\n            # Only one replicate at a time is materialized (small memory use).\n            block = arr[start:stop, 1:, 1:][:, triu_r, triu_c]\n            retained |= np.any(block != 0, axis=0)\n\n            n_take = 80 if first_80_only else rep_length\n            totals = np.sum(block[:n_take], axis=1, dtype=np.float64)\n            rows['system'].extend([system] * n_take)\n            rows['replicate_id'].extend([f'{system}_{rep_number:02d}'] * n_take)\n            rows['replicate'].extend([rep_number] * n_take)\n            rows['frame'].extend(range(1, n_take + 1))\n            rows['hbond_total'].extend(totals.tolist())\n            start = stop\n        assert start == expected\n\n    n_retained = int(retained.sum())\n    if n_retained == 0:\n        raise ValueError('No nonzero residue-pair interaction features found')\n    # Each frame uses the same denominator (same retained feature set).\n    rows['hbond_mean'] = [v / n_retained for v in rows['hbond_total']]\n\n    return {'data': rows, 'n_residues': nres,\n            'n_features': n_features, 'n_retained': n_retained}\n")

processed <- reticulate::py$build_hbond_frame_data(
  files_found[1], files_found[2], FIRST_80_ONLY
)

df <- as.data.frame(processed$data, stringsAsFactors = FALSE)
df$system <- factor(df$system, levels = c("GCU", "CGU"))
df$replicate_id <- factor(df$replicate_id)
df$replicate <- as.integer(df$replicate)
df$frame <- as.integer(df$frame)
df$hbond_mean <- as.numeric(df$hbond_mean)

df <- df[order(df$replicate_id, df$frame), ]
row.names(df) <- NULL

stopifnot(nlevels(df$replicate_id) == 60L)
stopifnot(all(table(df$replicate_id) %in% c(80L, 160L)))
stopifnot(nrow(df) == if (FIRST_80_ONLY) 4800L else 6400L)

cat("Residues:", processed$n_residues, "\n")
cat("Possible residue-pair features:", processed$n_features, "\n")
cat("Retained nonzero features:", processed$n_retained, "\n")
cat("Observations:", nrow(df), "; independent replicates: 60\n")

write.csv(df, file.path(OUTPUT_DIR, "frame_hbond_means.csv"), row.names = FALSE)

# Reproduce the level of the prior ANOVA: each replicate contributes one mean.
by_rep <- do.call(rbind, lapply(split(df, df$replicate_id), function(d) {
  data.frame(
    system = as.character(d$system[1]),
    replicate_id = as.character(d$replicate_id[1]),
    n_frames = nrow(d),
    mean_hbond = mean(d$hbond_mean),
    sd_hbond_within_rep = sd(d$hbond_mean)
  )
}))
row.names(by_rep) <- NULL
write.csv(by_rep, file.path(OUTPUT_DIR, "replicate_summary.csv"), row.names = FALSE)
cat("\nMean of the 30 replicate means in each system:\n")
print(aggregate(mean_hbond ~ system, data = by_rep, FUN = mean))
cat("\nOriginal ANOVA means (sanity check): GCU approximately 0.3596, CGU approximately 0.3709\n")

###########################
# Mixed-effects model: random replicate intercept + AR(1) residuals
###########################

control <- nlme::lmeControl(
  opt = "optim",
  maxIter = 200,
  msMaxIter = 200,
  niterEM = 50
)

fit_ar1 <- function(data, formula) {
  nlme::lme(
    fixed = formula,
    random = ~ 1 | replicate_id,
    correlation = nlme::corAR1(form = ~ frame | replicate_id),
    data = data,
    method = "REML",
    control = control
  )
}

# Pooled model: system fixed effect; replicate-within-system random effect.
pooled <- fit_ar1(df, hbond_mean ~ system)

# Each system separately: explicitly characterize within-system diversity.
gcu_model <- fit_ar1(subset(df, system == "GCU"), hbond_mean ~ 1)
cgu_model <- fit_ar1(subset(df, system == "CGU"), hbond_mean ~ 1)

###########################
# Variance components and residual dependence
###########################

extract_components <- function(model, label) {
  vc <- nlme::VarCorr(model)
  between_rep_variance <- as.numeric(vc["(Intercept)", "Variance"])
  within_rep_variance <- as.numeric(vc["Residual", "Variance"])
  rho <- as.numeric(nlme::coef(model$modelStruct$corStruct,
                              unconstrained = FALSE))[1]

  data.frame(
    analysis = label,
    between_replicate_var = between_rep_variance,
    within_replicate_residual_var = within_rep_variance,
    AR1_rho = rho,
    between_rep_share = between_rep_variance /
      (between_rep_variance + within_rep_variance),
    within_rep_share = within_rep_variance /
      (between_rep_variance + within_rep_variance)
  )
}

components <- rbind(
  extract_components(pooled, "Pooled (system fixed effect)"),
  extract_components(gcu_model, "GCU only"),
  extract_components(cgu_model, "CGU only")
)
write.csv(components, file.path(OUTPUT_DIR, "variance_components.csv"), row.names = FALSE)

cat("\n=== VARIANCE COMPONENTS ===\n")
print(components, row.names = FALSE, digits = 5)
cat("\nAR1_rho is the residual correlation at ONE frame lag (0.5 ns if sampled at 2 frames/ns).\n")
cat("Shares describe variance in the scalar per-frame H-bond mean, not full\n")
cat("conformational diversity; AR(1) affects temporal dependence separately.\n")

# Add the model-estimated difference between CGU and GCU in mean H-bonding.
fixed_effects <- summary(pooled)$tTable
write.csv(as.data.frame(fixed_effects),
          file.path(OUTPUT_DIR, "pooled_fixed_effects.csv"))

capture.output(
  cat("=== POOLED SYSTEM + REPLICATE + AR1 MODEL ===\n"),
  print(summary(pooled)),
  cat("\n=== GCU ONLY ===\n"),
  print(summary(gcu_model)),
  cat("\n=== CGU ONLY ===\n"),
  print(summary(cgu_model)),
  cat("\n=== VARIANCE COMPONENTS ===\n"),
  print(components),
  file = file.path(OUTPUT_DIR, "model_summaries.txt")
)

saveRDS(list(pooled = pooled, GCU = gcu_model, CGU = cgu_model),
        file.path(OUTPUT_DIR, "fitted_models.rds"))

###########################
# Simple diagnostic plots
###########################

png(file.path(OUTPUT_DIR, "pooled_residuals_vs_fitted.png"),
    width = 1000, height = 800, res = 130)
plot(fitted(pooled), resid(pooled, type = "normalized"),
     xlab = "Fitted mean H-bond value",
     ylab = "Normalized residual",
     main = "Mixed model: normalized residuals vs fitted")
abline(h = 0, lty = 2)
dev.off()

# The fitted AR(1) should reduce, though not necessarily eliminate, residual ACF.
acf_result <- nlme::ACF(pooled, resType = "normalized", maxLag = 20)
write.csv(as.data.frame(acf_result),
          file.path(OUTPUT_DIR, "normalized_residual_ACF.csv"), row.names = FALSE)
png(file.path(OUTPUT_DIR, "normalized_residual_ACF.png"),
    width = 1000, height = 800, res = 130)
plot(acf_result, main = "Normalized residual autocorrelation after AR(1)")
dev.off()

png(file.path(OUTPUT_DIR, "replicate_means_by_system.png"),
    width = 1000, height = 800, res = 130)
boxplot(mean_hbond ~ system, data = by_rep,
        xlab = "System", ylab = "Mean H-bond value per replicate",
        main = "30 independently initialized replicate means per system")
stripchart(mean_hbond ~ system, data = by_rep, vertical = TRUE,
           method = "jitter", add = TRUE, pch = 16)
dev.off()

cat("\nSaved model results to:", normalizePath(OUTPUT_DIR, winslash = "/"), "\n")
