# Generate EMC2 reference values for the RDM and LBA likelihoods in src/{rdm,lba}_jax.py.
#
#   Rscript tests/reference/generate_emc2_reference.R
#
# Run from the repository root. Writes four CSV fixtures next to this script; tests/
# test_emc2_reference.py reads them, so the pytest suite itself needs no R. Regenerating must
# produce byte-identical files -- every random draw is seeded, and every number is printed with
# %.17g so the round trip through text is exact in float64.
#
# EMC2 is the reference implementation for both models (Stevenson, Donzallaz, Heathcote et al.).
# The two entry points differ in a way that is easy to get wrong:
#
#   RDM: EMC2:::dRDM(rt, pars) / pRDM(rt, pars) take the *raw* rt and a matrix with named
#        columns v, B, A, t0, s. They rescale A, B, v <- (A, B, v) / s internally and then use a
#        unit-noise Wald with threshold B + A. Our model has no start-point variability, so
#        A = 0 exactly (EMC2's dWald has an exact A = 0 branch; A = 1e-8 differs in the tail).
#        Our mu = b/v, lam = (b/s)^2 is algebraically the same distribution.
#
#   LBA: EMC2:::dlba(t, A, b, v, sv, posdrift) / plba(...) take the *decision* time
#        t = rt - t0, and the threshold b itself rather than our gap B (b = A + B). Their `sv`
#        is our per-accumulator drift SD `s`; posdrift = TRUE matches our Phi(v/s) renormalizer.
#
# These two are Rcpp functions that do NOT recycle scalar arguments: every argument must be a
# vector as long as `t`, or they silently return Inf/NaN. Everything below is built with rep().
#
# EMC2's own race assembly (EMC2:::log_likelihood_race) is log(dfun(winner)) +
# log(1 - pfun(loser)), floored per trial at min_ll = log(1e-10) -- structurally identical to
# rdm_race_logpdf / lba_race_logpdf, and that 1e-10 is the same constant as their min_p.

suppressMessages(library(EMC2))

set.seed(20260817)

dRDM <- EMC2:::dRDM
pRDM <- EMC2:::pRDM
dlba <- EMC2:::dlba
plba <- EMC2:::plba
rWald <- EMC2:::rWald

MIN_LL <- log(1e-10)
OUT_DIR <- "tests/reference"
STAMP <- sprintf(
  "# EMC2 %s / %s / generated %s by tests/reference/generate_emc2_reference.R",
  as.character(packageVersion("EMC2")), R.version.string, Sys.Date()
)

#' Write a data frame at full float64 precision, behind a `#` provenance line.
write_fixture <- function(df, filename, extra_header = NULL) {
  formatted <- as.data.frame(
    lapply(df, function(col) if (is.numeric(col)) sprintf("%.17g", col) else as.character(col)),
    stringsAsFactors = FALSE
  )
  path <- file.path(OUT_DIR, filename)
  writeLines(c(STAMP, extra_header), path)
  # append = TRUE always warns about writing a header; that is exactly what is wanted here.
  suppressWarnings(
    write.table(formatted, path, append = TRUE, sep = ",", row.names = FALSE, quote = FALSE)
  )
  cat("wrote", path, "-", nrow(df), "rows\n")
}

#' Draw `n` values from a normal truncated below at 0 (the prior form used in src/priors.py).
rtnorm0 <- function(n, mean, sd) truncnorm::rtruncnorm(n, a = 0, mean = mean, sd = sd)


# ---------------------------------------------------------------------------
# RDM pointwise reference
#
# Prior rows follow conf/simulator/prior_simulator/rdm_simple.yaml; the winner/loser noise
# assignment is swapped at random so both orientations of the s_true / s_false = 1 race in
# rdm_jax._rdm_simple_log_likelihood are covered. Edge rows probe the regions the prior does
# not reach: decision times just above t0, the deep survival tail, and extreme s / b / v.
# ---------------------------------------------------------------------------

rdm_prior_rows <- function(n) {
  v_intercept <- rtnorm0(n, 1.0, 0.5)
  v_slope <- rtnorm0(n, 1.5, 0.5)
  s_true <- rgamma(n, shape = 12, scale = 0.1)
  b <- rgamma(n, shape = 8, scale = 0.15)
  t0 <- rtnorm0(n, 0.3, 0.2)

  true_wins <- runif(n) < 0.5
  data.frame(
    rt = t0 + exp(runif(n, log(0.05), log(5.0))),
    v_win = ifelse(true_wins, v_intercept + v_slope, v_intercept),
    v_lose = ifelse(true_wins, v_intercept, v_intercept + v_slope),
    s_win = ifelse(true_wins, s_true, 1.0),
    s_lose = ifelse(true_wins, 1.0, s_true),
    b = b, t0 = t0
  )
}

rdm_edge_rows <- function() {
  base <- list(v_win = 2.5, v_lose = 1.0, s_win = 1.2, s_lose = 1.0, b = 1.2, t0 = 0.3)
  grid <- expand.grid(
    dt = c(1e-6, 1e-4, 1e-2, 0.1, 1.0, 10, 100, 1000),
    s_win = c(0.1, 1.0, 5.0),
    b = c(0.05, 1.2, 5.0),
    v_win = c(0.01, 2.5, 10.0)
  )
  data.frame(
    rt = base$t0 + grid$dt,
    v_win = grid$v_win, v_lose = base$v_lose,
    s_win = grid$s_win, s_lose = base$s_lose,
    b = grid$b, t0 = base$t0
  )
}

rdm_reference <- function(df) {
  n <- nrow(df)
  pars_win <- cbind(v = df$v_win, B = df$b, A = rep(0, n), t0 = df$t0, s = df$s_win)
  pars_lose <- cbind(v = df$v_lose, B = df$b, A = rep(0, n), t0 = df$t0, s = df$s_lose)
  df$log_pdf <- log(dRDM(df$rt, pars_win))
  df$log_sf <- log(1 - pRDM(df$rt, pars_lose))
  df$log_race <- df$log_pdf + df$log_sf
  df
}

rdm_points <- rdm_reference(rbind(rdm_prior_rows(150), rdm_edge_rows()))
write_fixture(rdm_points, "emc2_rdm_pointwise.csv")


# ---------------------------------------------------------------------------
# LBA pointwise reference
#
# Same structure, priors from conf/simulator/prior_simulator/lba_simple.yaml (note the drift
# intercept is centred at 2, not the RDM's 1, and there is an extra start-point range A). The
# edge grid deliberately straddles the small-decision-time region where the closed-form density
# is a difference of nearly equal terms and underflows in float64 -- that is what pins
# lba_logpdf's floor against EMC2's min_ll.
# ---------------------------------------------------------------------------

lba_prior_rows <- function(n) {
  v_intercept <- rtnorm0(n, 2.0, 0.5)
  v_slope <- rtnorm0(n, 1.5, 0.5)
  s_true <- rgamma(n, shape = 12, scale = 0.1)
  sp_max <- rgamma(n, shape = 6, scale = 0.1)
  sp_gap <- rgamma(n, shape = 8, scale = 0.15)
  t0 <- rtnorm0(n, 0.3, 0.2)

  true_wins <- runif(n) < 0.5
  data.frame(
    rt = t0 + exp(runif(n, log(0.05), log(5.0))),
    v_win = ifelse(true_wins, v_intercept + v_slope, v_intercept),
    v_lose = ifelse(true_wins, v_intercept, v_intercept + v_slope),
    sv_win = ifelse(true_wins, s_true, 1.0),
    sv_lose = ifelse(true_wins, 1.0, s_true),
    A = sp_max, B = sp_gap, t0 = t0
  )
}

lba_edge_rows <- function() {
  base <- list(v_win = 3.5, v_lose = 2.0, sv_win = 1.2, sv_lose = 1.0, t0 = 0.3)
  grid <- expand.grid(
    dt = c(1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5, 1.0, 10, 100, 1000),
    A = c(0.05, 0.6, 2.0),
    B = c(1e-4, 0.1, 1.2, 5.0),
    sv_win = c(0.1, 1.2, 5.0)
  )
  data.frame(
    rt = base$t0 + grid$dt,
    v_win = base$v_win, v_lose = base$v_lose,
    sv_win = grid$sv_win, sv_lose = base$sv_lose,
    A = grid$A, B = grid$B, t0 = base$t0
  )
}

lba_reference <- function(df) {
  n <- nrow(df)
  dt <- df$rt - df$t0
  b <- df$A + df$B
  df$log_pdf <- log(dlba(dt, df$A, b, df$v_win, df$sv_win, TRUE))
  df$log_sf <- log(1 - plba(dt, df$A, b, df$v_lose, df$sv_lose, TRUE))
  df$log_race <- df$log_pdf + df$log_sf
  df
}

lba_points <- lba_reference(rbind(lba_prior_rows(150), lba_edge_rows()))
write_fixture(lba_points, "emc2_lba_pointwise.csv")


# ---------------------------------------------------------------------------
# Whole-dataset references
#
# Per-accumulator agreement does not catch a swapped winner/loser role or a mis-assigned
# s_true / s_false, because those are decided in _rdm_simple_log_likelihood /
# _lba_simple_log_likelihood rather than in the density functions. So simulate one dataset per
# model here, at the centre of the prior, and store both the data and EMC2's per-trial and
# total log-likelihood; the Python side then reproduces the total from (rt, choice) alone.
#
# `choice` is 0 for the false accumulator and 1 for the true one, matching data_x[:, 1].
# ---------------------------------------------------------------------------

N_TRIALS <- 300

#' Assemble a simulated race into the (rt, choice) contract and score it with EMC2.
race_dataset <- function(fpt_false, fpt_true, t0, score_fn) {
  choice <- as.integer(fpt_true < fpt_false)
  rt <- pmin(fpt_false, fpt_true) + t0
  df <- data.frame(rt = rt, choice = choice)
  df$log_race <- score_fn(df)
  df
}

# --- RDM. rWald(n, B, v, A) is unit-noise, so feed it B/s and v/s exactly as dRDM does. ---
rdm_truth <- list(v_intercept = 1.0, v_slope = 1.5, s_true = 1.2, s_false = 1.0, b = 1.2, t0 = 0.3)

rdm_data <- with(rdm_truth, {
  score <- function(df) {
    n <- nrow(df)
    win_true <- df$choice == 1
    v_win <- ifelse(win_true, v_intercept + v_slope, v_intercept)
    v_lose <- ifelse(win_true, v_intercept, v_intercept + v_slope)
    s_win <- ifelse(win_true, s_true, s_false)
    s_lose <- ifelse(win_true, s_false, s_true)
    pars_win <- cbind(v = v_win, B = rep(b, n), A = rep(0, n), t0 = rep(t0, n), s = s_win)
    pars_lose <- cbind(v = v_lose, B = rep(b, n), A = rep(0, n), t0 = rep(t0, n), s = s_lose)
    log(dRDM(df$rt, pars_win)) + log(1 - pRDM(df$rt, pars_lose))
  }
  race_dataset(
    rWald(N_TRIALS, B = rep(b / s_false, N_TRIALS), v = rep(v_intercept / s_false, N_TRIALS),
          A = rep(0, N_TRIALS)),
    rWald(N_TRIALS, B = rep(b / s_true, N_TRIALS), v = rep((v_intercept + v_slope) / s_true, N_TRIALS),
          A = rep(0, N_TRIALS)),
    t0, score
  )
})

write_fixture(
  rdm_data, "emc2_rdm_dataset.csv",
  c(
    sprintf("# v_intercept=%.17g v_slope=%.17g s_true=%.17g s_false=%.17g b=%.17g t0=%.17g",
            rdm_truth$v_intercept, rdm_truth$v_slope, rdm_truth$s_true, rdm_truth$s_false,
            rdm_truth$b, rdm_truth$t0),
    sprintf("# total_log_likelihood=%.17g", sum(pmax(MIN_LL, rdm_data$log_race)))
  )
)

# --- LBA. First passage is the deterministic (b - k) / d, with k ~ U(0, A), d ~ N(v, sv)[0,). ---
lba_truth <- list(v_intercept = 2.0, v_slope = 1.5, s_true = 1.2, s_false = 1.0,
                  A = 0.6, B = 1.2, t0 = 0.3)

lba_data <- with(lba_truth, {
  b <- A + B
  score <- function(df) {
    n <- nrow(df)
    win_true <- df$choice == 1
    v_win <- ifelse(win_true, v_intercept + v_slope, v_intercept)
    v_lose <- ifelse(win_true, v_intercept, v_intercept + v_slope)
    sv_win <- ifelse(win_true, s_true, s_false)
    sv_lose <- ifelse(win_true, s_false, s_true)
    dt <- df$rt - t0
    log(dlba(dt, rep(A, n), rep(b, n), v_win, sv_win, TRUE)) +
      log(1 - plba(dt, rep(A, n), rep(b, n), v_lose, sv_lose, TRUE))
  }
  race_dataset(
    (b - runif(N_TRIALS, 0, A)) / rtnorm0(N_TRIALS, v_intercept, s_false),
    (b - runif(N_TRIALS, 0, A)) / rtnorm0(N_TRIALS, v_intercept + v_slope, s_true),
    t0, score
  )
})

write_fixture(
  lba_data, "emc2_lba_dataset.csv",
  c(
    sprintf("# v_intercept=%.17g v_slope=%.17g s_true=%.17g s_false=%.17g A=%.17g B=%.17g t0=%.17g",
            lba_truth$v_intercept, lba_truth$v_slope, lba_truth$s_true, lba_truth$s_false,
            lba_truth$A, lba_truth$B, lba_truth$t0),
    sprintf("# total_log_likelihood=%.17g", sum(pmax(MIN_LL, lba_data$log_race)))
  )
)
