# Installs the R packages needed by main.R.
#
# The rocker/r-ver image pins its CRAN mirror to a dated Posit Package
# Manager snapshot, so the same package versions are installed every time
# the image is built (dplyr 1.1.4 and readr 2.1.5 for rocker/r-ver:4.4.1).

packages <- c("dplyr", "readr")

install.packages(packages)

# install.packages() only warns on failure, so stop the Docker build
# explicitly if anything did not install.
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste("Package failed to install:", pkg))
  }
  cat(sprintf("Installed %s %s\n", pkg, as.character(packageVersion(pkg))))
}
