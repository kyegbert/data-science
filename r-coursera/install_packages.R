# Script installs specific legacy packages from CRAN mirror for R 3.1.3.
# Package installation errors are written to install_packages.Rout on Docker 
# host.

# Enable ssl downloads
options(download.file.method = "wget")

mirror <- "https://cran.r-project.org"

# Install RCurl and dependencies first to retry package URL 404 errors in 
# get_cran_package()
pkgs <- c("/src/contrib/bitops_1.0-6.tar.gz",
    "/src/contrib/Archive/RCurl/RCurl_1.95-4.5.tar.gz"
)

pkgs <- paste0(mirror, pkgs)

lapply(pkgs, install.packages, repos=NULL, type="source")

library(RCurl)

# Packages that have dependencies are listed at the end of this vector so 
# dependencies are installed first.  
# Not using dependencies=TRUE attribute because we're installing specific 
# versions of packages. 
pkgs <- c("/src/contrib/Archive/acepack/acepack_1.3-3.3.tar.gz",
          "/src/contrib/brew_1.0-6.tar.gz",
          "/src/contrib/caTools_1.17.1.tar.gz",
          "/src/contrib/Archive/chron/chron_2.3-45.tar.gz",
          "/src/contrib/Archive/colorspace/colorspace_1.2-4.tar.gz",
          "/src/contrib/Archive/CORElearn/CORElearn_0.9.45.tar.gz",
          "/src/contrib/Archive/corrplot/corrplot_0.73.tar.gz",
          "/src/contrib/Archive/DBI/DBI_0.3.1.tar.gz",
          "/src/contrib/dichromat_2.0-0.tar.gz",
          "/src/contrib/Archive/digest/digest_0.6.8.tar.gz",
          "/src/contrib/Archive/e1071/e1071_1.6-4.tar.gz",
          "/src/contrib/Archive/ElemStatLearn/ElemStatLearn_2012.04-0.tar.gz",
          "/src/contrib/Archive/formatR/formatR_1.0.tar.gz",
          "/src/contrib/Archive/Formula/Formula_1.2-0.tar.gz",
          "/src/contrib/fracdiff_1.4-2.tar.gz",
          "/src/contrib/Archive/gbm/gbm_2.1.tar.gz",
          "/src/contrib/Archive/gridExtra/gridExtra_0.9.1.tar.gz",
          "/src/contrib/Archive/gtable/gtable_0.1.2.tar.gz",
          "/src/contrib/Archive/gtools/gtools_3.4.1.tar.gz",
          "/src/contrib/hflights_0.1.tar.gz",
          "/src/contrib/Archive/highr/highr_0.4.tar.gz",
          "/src/contrib/Archive/HistData/HistData_0.7-5.tar.gz",
          "/src/contrib/Archive/htmltools/htmltools_0.2.6.tar.gz",
          "/src/contrib/Archive/iterators/iterators_1.0.7.tar.gz",
          "/src/contrib/jpeg_0.1-8.tar.gz",
          "/src/contrib/Archive/jsonlite/jsonlite_0.9.14.tar.gz",
          "/src/contrib/Archive/kernlab/kernlab_0.9-20.tar.gz",
          "/src/contrib/labeling_0.3.tar.gz",
          "/src/contrib/Archive/Lahman/Lahman_3.0-1.tar.gz",
          "/src/contrib/lars_1.2.tar.gz",
          "/src/contrib/Archive/lazyeval/lazyeval_0.1.10.tar.gz",
          "/src/contrib/magrittr_1.5.tar.gz",
          "/src/contrib/manipulate_1.0.1.tar.gz",
          "/src/contrib/Archive/memoise/memoise_0.2.1.tar.gz",
          "/src/contrib/Archive/mime/mime_0.2.tar.gz",
          "/src/contrib/Archive/munsell/munsell_0.4.tar.gz",
          "/src/contrib/nloptr_1.0.4.tar.gz",
          "/src/contrib/openintro_1.4.tar.gz",
          "/src/contrib/Archive/pgmm/pgmm_1.1.tar.gz",
          "/src/contrib/profileModel_0.5-9.tar.gz",
          "/src/contrib/Archive/proto/proto_0.3-10.tar.gz",
          "/src/contrib/quadprog_1.5-5.tar.gz",
          "/src/contrib/Archive/R6/R6_2.0.1.tar.gz",
          "/src/contrib/Archive/randomForest/randomForest_4.6-10.tar.gz",
          "/src/contrib/Archive/rattle/rattle_3.4.1.tar.gz",
          "/src/contrib/rbenchmark_1.0.0.tar.gz",
          "/src/contrib/RColorBrewer_1.1-2.tar.gz",
          "/src/contrib/Archive/Rcpp/Rcpp_0.11.4.tar.gz",
          "/src/contrib/Archive/rJava/rJava_0.9-6.tar.gz",
          "/src/contrib/RJSONIO_1.3-0.tar.gz",
          "/src/contrib/Archive/RMySQL/RMySQL_0.10.1.tar.gz",
          "/src/contrib/Archive/rpart.plot/rpart.plot_1.5.2.tar.gz",
          "/src/contrib/Archive/RSQLite/RSQLite_1.0.0.tar.gz",
          "/src/contrib/Archive/rstudioapi/rstudioapi_0.2.tar.gz",
          "/src/contrib/Archive/SparseM/SparseM_1.6.tar.gz",
          "/src/contrib/Archive/stringi/stringi_0.4-1.tar.gz",
          "/src/contrib/Archive/stringr/stringr_0.6.2.tar.gz",
          "/src/contrib/Archive/testthat/testthat_0.9.1.tar.gz",
          "/src/contrib/timeDate_3012.100.tar.gz",
          "/src/contrib/whisker_0.3-2.tar.gz",
          "/src/contrib/Archive/XML/XML_3.98-1.1.tar.gz",
          "/src/contrib/Archive/xtable/xtable_1.7-4.tar.gz",
          "/src/contrib/Archive/yaml/yaml_2.1.13.tar.gz",
          "/src/contrib/Archive/zoo/zoo_1.7-11.tar.gz",
          "/src/contrib/Archive/RcppEigen/RcppEigen_0.3.2.1.2.tar.gz",
          "/src/contrib/Archive/httr/httr_0.6.1.tar.gz",
          "/src/contrib/Archive/swirl/swirl_2.2.21.tar.gz",
          "/src/contrib/Archive/BH/BH_1.55.0-3.tar.gz",
          "/src/contrib/Archive/RcppArmadillo/RcppArmadillo_0.4.650.1.1.tar.gz",
          "/src/contrib/xlsxjars_0.6.1.tar.gz",
          "/src/contrib/xlsx_0.5.7.tar.gz",
          "/src/contrib/brglm_0.5-9.tar.gz",
          "/src/contrib/elasticnet_1.1.tar.gz",
          "/src/contrib/Archive/evaluate/evaluate_0.5.5.tar.gz",
          "/src/contrib/Archive/foreach/foreach_1.4.2.tar.gz",
          "/src/contrib/Archive/gdata/gdata_2.13.3.tar.gz",
          "/src/contrib/Archive/googleVis/googleVis_0.5.8.tar.gz",
          "/src/contrib/Archive/gplots/gplots_2.16.0.tar.gz",
          "/src/contrib/gsubfn_0.6-6.tar.gz",
          "/src/contrib/Archive/httpuv/httpuv_1.3.2.tar.gz",
          "/src/contrib/Archive/latticeExtra/latticeExtra_0.6-26.tar.gz",
          "/src/contrib/minqa_1.2.4.tar.gz",
          "/src/contrib/Archive/plyr/plyr_1.8.1.tar.gz",
          "/src/contrib/Archive/pROC/pROC_1.7.3.tar.gz",
          "/src/contrib/Archive/quantreg/quantreg_5.11.tar.gz",
          "/src/contrib/Archive/reshape/reshape_0.8.5.tar.gz",
          "/src/contrib/Archive/reshape2/reshape2_1.4.1.tar.gz",
          "/src/contrib/Archive/ROCR/ROCR_1.0-5.tar.gz",
          "/src/contrib/Archive/roxygen2/roxygen2_4.1.1.tar.gz",
          "/src/contrib/Archive/scales/scales_0.2.4.tar.gz",
          "/src/contrib/Archive/shiny/shiny_0.11.1.tar.gz",
          "/src/contrib/sqldf_0.4-10.tar.gz",
          "/src/contrib/Archive/tseries/tseries_0.10-34.tar.gz",
          "/src/contrib/xts_0.9-7.tar.gz",
          "/src/contrib/AppliedPredictiveModeling_1.1-6.tar.gz",
          "/src/contrib/Archive/lme4/lme4_1.1-7.tar.gz",
          "/src/contrib/BradleyTerry2_1.0-6.tar.gz",
          "/src/contrib/Archive/pbkrtest/pbkrtest_0.4-2.tar.gz",
          "/src/contrib/Archive/car/car_2.0-25.tar.gz",
          "/src/contrib/Archive/ggplot2/ggplot2_1.0.0.tar.gz",
          "/src/contrib/Archive/caret/caret_6.0-41.tar.gz",
          "/src/contrib/Archive/data.table/data.table_1.9.2.tar.gz",
          "/src/contrib/Archive/devtools/devtools_1.7.0.tar.gz",
          "/src/contrib/Archive/forecast/forecast_5.9.tar.gz",
          "/src/contrib/Archive/GGally/GGally_0.5.0.tar.gz",
          "/src/contrib/Archive/Hmisc/Hmisc_3.14-6.tar.gz",
          "/src/contrib/Archive/lubridate/lubridate_1.3.3.tar.gz",
          "/src/contrib/Archive/microbenchmark/microbenchmark_1.4-2.tar.gz",
          "/src/contrib/Archive/TTR/TTR_0.22-0.tar.gz",
          "/src/contrib/Archive/quantmod/quantmod_0.4-3.tar.gz",
          "/src/contrib/Archive/UsingR/UsingR_2.0-4.tar.gz",
          "/src/contrib/assertthat_0.1.tar.gz",
          "/src/contrib/Archive/dplyr/dplyr_0.4.1.tar.gz",
          "/src/contrib/Archive/tidyr/tidyr_0.2.0.tar.gz",
          "/src/contrib/Archive/markdown/markdown_0.7.4.tar.gz",
          "/src/contrib/Archive/knitr/knitr_1.9.tar.gz",
          "/src/contrib/Archive/rmarkdown/rmarkdown_0.5.1.tar.gz"
    )

# Retry packages that might have been moved to archive
get_cran_package <- function(pkg) {
    pkg <- paste0(mirror, pkg)
    if (!(url.exists(pkg))) {
        if (!(grepl(pkg, "Archive"))) {
            pkg_name_version <- tail(strsplit(pkg, split="/")[[1]],1)
            pkg_name <- regmatches(pkg_name_version, 
                regexpr("([a-zA-Z\\d\\.]+)(?=_)", pkg_name_version, perl=TRUE))
            pkg <- paste0(mirror, 
                "/src/contrib/Archive/",
                pkg_name,
                "/",
                pkg_name_version)
        }
    }
    install.packages(pkg, repos=NULL, type="source")
}

lapply(pkgs, get_cran_package)

# Install legacy non-CRAN package
# SSL not available for this url
install.packages("http://cran.wustl.edu/src/contrib/Archive/Defaults/Defaults_1.1-1.tar.gz", 
    repos=NULL, type="source")
