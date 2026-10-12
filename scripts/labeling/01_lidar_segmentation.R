#!/usr/bin/env Rscript
# Copyright (C) 2023-2026 Yunzhe Zhu
# SPDX-License-Identifier: GPL-3.0-only
# Part of fuel-mapping-CA (https://github.com/YunzheZhu/fuel-mapping-CA).
#
# 01_lidar_segmentation.R
# -----------------------
# Step 1-2 of the semi-automatic labelling workflow (paper Sec. 2.4, Fig. 2):
# LiDAR object segmentation and initial object labels.
#
#   point cloud -> noise removal -> height normalisation -> 0.6 m CHM
#   -> adaptive local-maximum object tops -> Dalponte region growing
#   -> convex-hull object polygons -> planar mask -> zq95 + planar fraction
#   -> initial class (Clss_95): 0 other, 1 tree, 2 shrub, 3 building
#
# IMPORTANT - provenance. This is a cleaned *reference re-implementation* of
# the lidR workflow used for the paper. The original 2023-24 script that wrote
# the paper's `*_crowns.shp` files was not preserved verbatim. This version was
# reconstructed from later copies of the same workflow and from 2024 R session
# histories, with parameters aligned to the paper text. Polygons and classes
# can therefore differ slightly from the labels used in the paper. All initial
# labels were manually checked and edited against NAIP imagery afterwards
# (Fig. 2 step 3), so the output of this script is a starting point, not a
# final label.
#
# Method references (as cited in the paper):
#   Zhu, Dixon & Jin (2026) Remote Sens. Environ. 333, 115091, Sec. 2.4.
#   Dixon et al. (2023) Remote Sens. Environ. 298, 113842 (LiDAR-based object
#     delineation used to automate labelling).
#   Roussel et al. (2020) lidR. Remote Sens. Environ. 251, 112061.
#   Dalponte & Coomes (2016) Methods Ecol. Evol. 7, 1236-1245;
#     Coomes et al. (2017) Remote Sens. Environ. 194, 77-88.
#   Limberger & Oliveira (2015) Pattern Recognit. 48, 2043-2053 (planar regions).
#
# Requirements: R >= 4.1, lidR (>= 4.0), terra, sf, exactextractr, data.table.
# Tested with R 4.6.0, lidR 4.3.2, terra 1.9, sf 1.1, exactextractr 0.10.
#
# Usage
#   Rscript 01_lidar_segmentation.R --input <file.laz | folder> --outdir <dir>
#           [--threads 4] [--format gpkg|shp] [--overwrite]
#           [--<parameter> <value> ...]   (see DEFAULTS below)
# or, from R:  source("01_lidar_segmentation.R"); segment_tile("tile.laz", "out/")
#
# Input: one LAZ/LAS per tile (e.g. from scripts/preprocessing/download_3dep_lidar.py),
#   already in a projected metric CRS (EPSG:6414 in this project) and with the
#   ground points classified (ASPRS class 2), as delivered by USGS 3DEP.
#
# Outputs, per input <stem>.laz, written to --outdir:
#   <stem>_chm.tif     0.6 m canopy height model (3x3 median smoothed), metres
#   <stem>_dtm.tif     0.6 m IDW terrain model (3x3 median smoothed), metres
#   <stem>_pMask.tif   0.6 m planar mask (1 = planar pixel, 3x3 median smoothed)
#   <stem>_ttops.gpkg  object tops (points)
#   <stem>_crowns.gpkg object polygons with attributes:
#       treeID   object id
#       Z        top height (max normalised height, m)
#       zq95     95th percentile of normalised point heights (m)
#       npoints  number of points in the object
#       area     polygon area (m2)
#       planar   mean planar-mask value inside the polygon (0-1)
#       Clss_95  initial class: 0 other (low object), 1 tree, 2 shrub, 3 building
#   (with --format shp the polygons are written as <stem>_crowns.shp instead)

suppressPackageStartupMessages({
  library(lidR)
  library(terra)
  library(sf)
  library(exactextractr)
  library(data.table)
})

# ---------------------------------------------------------------------------
# Parameters. Values follow the paper text where it gives one; the remainder
# (marked "code") follow the reconstructed 2024-25 scripts.
# ---------------------------------------------------------------------------
DEFAULTS <- list(
  res = 0.6,                # CHM / DTM / planar-mask resolution, m (paper)
  drop_z_below = NA_real_,  # drop points below this absolute elevation, m; NA keeps all. The 2023-24
                            #   code used 0, which removes the ground in areas below sea level
  min_z_agl = NA_real_,     # optional: after height normalisation, drop points below this height
                            #   above ground, m (e.g. -0.5 for remaining low noise); NA keeps all
  sor_k = 10,               # noise: k nearest neighbours (paper)
  sor_m = 3,                # noise: drop points > m SD from neighbour mean (paper)
  idw_k = 10,               # IDW neighbours for the DTM (code)
  normalize = "knnidw",     # "knnidw": point-wise IDW normalisation (code, all
                            #   preserved versions); "dtm": subtract the smoothed
                            #   0.6 m DTM raster (wording of the paper)
  hmin = 0.5,               # minimum height of an object top, m (code)
  th_tree = 0.5,            # Dalponte: minimum CHM height of a crown pixel, m (code)
  th_seed = 0.4,            # Dalponte: seed pixel > 40 % of top height (paper)
  th_cr = 0.5,              # Dalponte: pixel > 50 % of region mean height (paper)
  max_cr = 40,              # Dalponte: maximum object diameter, pixels (code)
  planar_k = 25,            # shp_plane(): neighbours used to fit planes (code)
  planar_pixel_frac = 0.95, # pixel is planar if > 95 % of its points are (code)
  planar_frac = 0.5,        # object is a building if planar fraction >= 0.5 (paper)
  tree_height = 5,          # zq95 >= 5 m -> tree, otherwise shrub (paper)
  min_height = 1,           # zq95 < 1 m -> other (paper: objects < 1 m are "Other")
  min_area = 0.36,          # drop polygons smaller than one 0.6 m pixel, m2 (code)
  hull = "first",           # convex hull of "first" returns (paper) or "all" points
  format = "gpkg",          # polygon output format: "gpkg" or "shp"
  threads = 2,              # lidR threads
  overwrite = FALSE
)

#' Adaptive local-maximum window size (m) as a function of height (m).
#' 2 m for objects below 2 m, rising smoothly to about 10 m for objects >= 30 m,
#' so that small shrubs and large trees/buildings can both be detected.
window_size <- function(h) {
  ws <- 2 + 8.9 * (1 - exp(-0.08 * (h - 2)))
  ws[h < 2] <- 2
  ws[h > 30] <- 10
  ws
}

#' Initial object class from height and planar fraction (paper Sec. 2.4).
#' Objects lower than min_height are other (0), planar or not. Taller planar
#' objects are buildings (3); the rest are trees (1) if zq95 >= tree_height,
#' else shrubs (2).
classify_objects <- function(zq95, planar, tree_height = 5, min_height = 1,
                             planar_frac = 0.5) {
  cls <- ifelse(zq95 >= tree_height, 1L, 2L)
  cls[!is.na(planar) & planar >= planar_frac] <- 3L
  cls[zq95 < min_height] <- 0L
  cls
}

median3 <- function(r) terra::focal(r, w = 3, fun = "median", na.rm = TRUE)

write_raster <- function(r, path) {
  terra::writeRaster(r, path, overwrite = TRUE, gdal = c("COMPRESS=DEFLATE"))
}

write_vector <- function(x, path) {
  st_write(x, path, delete_dsn = file.exists(path), quiet = TRUE)
}

#' Segment one LAZ/LAS tile and write CHM, DTM, planar mask, tops and polygons.
#'
#' @param laz    path to the input point cloud
#' @param outdir output folder
#' @param p      named list of parameters (see DEFAULTS)
#' @return the object polygons (sf), invisibly; NULL if nothing was found
segment_tile <- function(laz, outdir, p = DEFAULTS) {
  p <- modifyList(DEFAULTS, p)
  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  stem <- tools::file_path_sans_ext(basename(laz))
  out <- function(suffix) file.path(outdir, paste0(stem, suffix))
  crowns_path <- out(if (p$format == "shp") "_crowns.shp" else "_crowns.gpkg")
  if (file.exists(crowns_path) && !isTRUE(p$overwrite)) {
    message("exists, skipped: ", crowns_path)
    return(invisible(NULL))
  }

  # 1) Read and remove noise: statistical outlier removal (k = 10, 3 SD).
  las <- if (is.na(p$drop_z_below)) readLAS(laz) else
    readLAS(laz, filter = sprintf("-drop_z_below %g", p$drop_z_below))
  if (is.empty(las)) stop("empty point cloud: ", laz)
  las <- classify_noise(las, sor(k = p$sor_k, m = p$sor_m, quantile = FALSE))
  las <- filter_poi(las, Classification != LASNOISE)

  # 2) Terrain: IDW DTM from ground points, 3x3 median; height normalisation.
  dtm <- median3(rasterize_terrain(las, res = p$res, algorithm = knnidw(k = p$idw_k)))
  las <- if (p$normalize == "dtm") normalize_height(las, dtm) else normalize_height(las, knnidw())
  if (!is.na(p$min_z_agl)) las <- filter_poi(las, Z >= p$min_z_agl)

  # 3) Canopy height model at 0.6 m, 3x3 median.
  chm <- median3(rasterize_canopy(las, res = p$res, algorithm = p2r(subcircle = 0.2)))

  # 4) Planar mask: plane fitting per point, then the planar share per pixel.
  las <- segment_shapes(las, shp_plane(k = p$planar_k), "planar")
  planar_share <- pixel_metrics(las, ~mean(planar), res = p$res)
  pmask <- median3((planar_share > p$planar_pixel_frac) * 1)

  write_raster(chm, out("_chm.tif"))
  write_raster(dtm, out("_dtm.tif"))
  write_raster(pmask, out("_pMask.tif"))

  # 5) Object tops (adaptive local maxima) and region growing on the CHM.
  ttops <- locate_trees(las, lmf(ws = window_size, hmin = p$hmin))
  if (nrow(ttops) == 0) {
    message("no objects found: ", laz)
    return(invisible(NULL))
  }
  write_vector(ttops, out("_ttops.gpkg"))
  algo <- dalponte2016(chm, ttops, th_tree = p$th_tree, th_seed = p$th_seed,
                       th_cr = p$th_cr, max_cr = p$max_cr)
  las <- segment_trees(las, algo)
  objs <- filter_poi(las, !is.na(treeID))

  # 6) Object polygons: convex hull of the object's (first-return) points.
  hull_pts <- if (p$hull == "first") filter_first(objs) else objs
  crowns <- crown_metrics(hull_pts, func = ~list(nhull = length(Z)), geom = "convex")
  crowns <- crowns[st_is(crowns, "POLYGON"), ]
  crowns$area <- as.numeric(st_area(crowns))
  crowns <- crowns[crowns$area >= p$min_area, ]
  if (nrow(crowns) == 0) {
    message("no polygons left: ", laz)
    return(invisible(NULL))
  }

  # 7) Object metrics from all points, planar fraction, initial class.
  mets <- objs@data[, list(Z = max(Z), zq95 = as.numeric(quantile(Z, 0.95)),
                           npoints = .N), by = treeID]
  idx <- match(crowns$treeID, mets$treeID)
  crowns$Z <- mets$Z[idx]
  crowns$zq95 <- mets$zq95[idx]
  crowns$npoints <- mets$npoints[idx]
  crowns$planar <- exact_extract(pmask, crowns, "mean", progress = FALSE)
  crowns$planar[is.nan(crowns$planar)] <- NA
  crowns$Clss_95 <- classify_objects(crowns$zq95, crowns$planar, p$tree_height,
                                     p$min_height, p$planar_frac)
  crowns <- crowns[, c("treeID", "Z", "zq95", "npoints", "area", "planar", "Clss_95")]

  write_vector(crowns, crowns_path)
  message(sprintf("%s: %d objects (tree %d, shrub %d, building %d, other %d)",
                  stem, nrow(crowns), sum(crowns$Clss_95 == 1), sum(crowns$Clss_95 == 2),
                  sum(crowns$Clss_95 == 3), sum(crowns$Clss_95 == 0)))
  invisible(crowns)
}

# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    has_value <- i < length(args) && !startsWith(args[i + 1], "--")
    out[[key]] <- if (has_value) args[i + 1] else TRUE
    i <- i + if (has_value) 2 else 1
  }
  out
}

main <- function() {
  a <- parse_args(commandArgs(trailingOnly = TRUE))
  if (is.null(a$input) || is.null(a$outdir) || isTRUE(a$help)) {
    cat("Usage: Rscript 01_lidar_segmentation.R --input <file.laz|folder> --outdir <dir>",
        "[--threads N] [--format gpkg|shp] [--overwrite] [--<parameter> <value>]\n",
        "Parameters and defaults:\n")
    str(DEFAULTS)
    quit(status = if (isTRUE(a$help)) 0 else 1)
  }
  p <- DEFAULTS
  for (key in setdiff(names(a), c("input", "outdir"))) {
    if (!key %in% names(DEFAULTS)) stop("unknown option --", key)
    default <- DEFAULTS[[key]]
    p[[key]] <- if (is.logical(default)) isTRUE(a[[key]]) || identical(a[[key]], "TRUE")
                else if (is.numeric(default)) as.numeric(a[[key]]) else a[[key]]
  }
  set_lidr_threads(p$threads)

  files <- if (dir.exists(a$input)) {
    list.files(a$input, pattern = "\\.la[sz]$", full.names = TRUE, ignore.case = TRUE)
  } else a$input
  for (f in files) {
    message("processing ", f)
    tryCatch(segment_tile(f, a$outdir, p),
             error = function(e) message("FAILED ", f, ": ", conditionMessage(e)))
  }
}

if (sys.nframe() == 0L) main()
