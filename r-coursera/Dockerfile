FROM debian:stable

MAINTAINER "Kim Egbert" @kyegbert

ADD install_packages.R /tmp/install_packages.R

# Set a default non-root user inside container to run container processes
# Add user to 'staff' group, granting them write privileges to 
# /usr/local/lib/R/site.library.
# User should also have and own a home directory (for linked volumes to work 
# properly). 
RUN useradd docker \
    && mkdir /home/docker \
    && chown docker:docker /home/docker \
    && addgroup docker staff

# Install required packages to build R from source
# Suppress prompts as installation is unattended
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        default-libmysqlclient-dev \
        g++ \
        gcc \
        gfortran \
        libjpeg-dev \
        libbz2-1.0 \
        libcairo2-dev \
        libcurl4-openssl-dev \
        libpcre3 \
        libreadline-dev \
        liblzma5 \
        libxml2-dev \
        lmodern \
        nano \
        readline-common \
        locales \
        openjdk-8-jdk \
        pandoc \
        pandoc-citeproc \
        r-cran-rjava \
        tcl-dev \
        texinfo \
        texlive-base \
        texlive-binaries \
        texlive-fonts-extra \
        texlive-fonts-recommended \
        texlive-generic-recommended \
        texlive-humanities \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-pictures \
        tk-dev \
        wget \
        xorg \
        xorg-dev \
        xvfb \
        xauth \
        xfonts-base \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Configure default locale
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && locale-gen en_US.utf8 \
    && /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

# Build and install R 3.1.3
RUN cd usr/local/src && mkdir R && cd R \
    && VERSION=3.1.3 \
    && wget https://cran.r-project.org/src/base/R-3/R-$VERSION.tar.gz \
    && tar -xzvf R-$VERSION.tar.gz \
    && cd R-$VERSION \
    && ./configure \
    && make \
    && make install

# Install specific versions of R packages
# Send output and errors to log file on host
RUN mkdir /tmp/r_log \
    && mkdir /tmp/shared_volume
RUN /bin/bash -c "Rscript --no-save --no-restore --verbose /tmp/install_packages.R |& tee -a /tmp/r_log/install_packages.Rout"

# Run R scripts via command line using Rscript or launch R console for 
# interactive session.