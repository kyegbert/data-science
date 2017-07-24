## Dockerfile: R version 3.1.3 in Debian

### Description

This Dockerfile replicates the R environment I used for Coursera Data Science Specialization courses.  The Docker conainer allows me to re-run my old projects in an isolated environment while keeping the R version on my laptop up-to-date. The Dockerfile installs a specific major and minor version of R (3.1.3) and specific versions of R packages. RStudio is not installed. R is built from source as the specific minor version of R (3.1.3) is not available via the Debian package repository. This Dockerfile is originally based on the Rocker r-base Dockerfile (https://github.com/rocker-org/rocker/blob/master/r-base/Dockerfile) with heavy modifications. This Dockerfile was tested most recently using Docker Community Edition version 10.06.0-mac19.

### Usage

When launching the container, share a host directory with the container for access to R scripts, such as: 

`docker run --name r-coursera -it -p 3838:3838 --user docker -v $(pwd)/<host-path-to-shared-volume>:/<container-path-to-shared-volume> <image_id>`.

R scripts are run from the command line (bash) and not via RStudio.

R scripts that do not include calls to graphic devices, such as the png() function, can be called from the command line like so: 

`Rscript --no-save --no-restore --verbose /<container-path-to-rscript>/<rscript.R>`

R scripts that include functions that call graphics devices or output plots need to be prefaced with xvfb-run like so: 

`xvfb-run Rscript --no-save --no-restore --verbose /<container-path-to-rscript>/<rscript.R>`

R scripts that utilize Pandoc to output PDFs, HTML, etc. need to be run like so: 

`xvfb-run Rscript -e 'library(rmarkdown); rmarkdown::render("/<container-path-to-rscript>/<rscript.Rmd>")'`

Run Shiny apps as below. Pointing the `shiny.host` to a non-loopback IP allows the app to be accessed outside the container.

`xvfb-run Rscript -e 'library(methods); shiny::runApp("/<container-path-to-project-directory>/", port=3838, launch.browser=FALSE, host = getOption("shiny.host", "0.0.0.0"))'`

### License

This Dockerfile is licensed under the GPL 2 or later, the same license as the Rocker r-base Dockerfile my Dockerfile is derived from (https://github.com/rocker-org/rocker/blob/master/r-base/Dockerfile).
