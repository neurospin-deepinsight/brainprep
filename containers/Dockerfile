# Use Ubuntu 20.04 LTS
FROM ubuntu:focal-20210416


# Install dependencies and prereqs
RUN apt-get update \
    && apt-get -y install \
        wget \
        nano \
        unzip \
        libxext6 \
        libxt6 \
        moreutils \
        locales \
        curl \
        bzip2 \
        python3.8 \
        python3-pip \
        python3.8-dev \
        git \
        tcsh \
        locales \
    && apt-get clean

RUN locale-gen "en_US.UTF-8" \
    && dpkg-reconfigure locales \
    && echo "LANGUAGE='en_US.UTF-8'" >> /etc/default/locale \
    && echo "LC_ALL='en_US.UTF-8'" >> /etc/default/locale

ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LANGUAGE="en_US.UTF-8" \
    LC_ALL=C


<<<<<<< HEAD
# Installing brainprep  https://github.com/neurospin-deepinsight/brainprep.git
RUN pip install --no-cache-dir git+https://github.com/LoicDorval/brainprep.git
=======
# Installing brainprep
RUN pip install --no-cache-dir git+https://github.com/LoicDorval/brainprep.git@qc_LD
>>>>>>> 26d6f5efd1cb67266ca8efa5f872966948d647fa


# FSL 5.0.11 (neurodocker build)
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           bc \
           dc \
           file \
           libfontconfig1 \
           libfreetype6 \
           libgl1-mesa-dev \
           libgl1-mesa-dri \
           libglu1-mesa-dev \
           libgomp1 \
           libice6 \
           libxcursor1 \
           libxft2 \
           libxinerama1 \
           libxrandr2 \
           libxrender1 \
           libxt6 \
           sudo \
           wget \
    && apt-get clean

RUN rm -rf /var/lib/apt/lists/* \
    && echo "Downloading FSL ..." \
    && mkdir -p /opt/fsl-5.0.11 \
    && curl -fsSL --retry 5 https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.11-centos6_64.tar.gz \
    | tar -xz -C /opt/fsl-5.0.11 --strip-components 1

RUN echo "Installing FSL conda environment ..." \
    && bash /opt/fsl-5.0.11/etc/fslconf/fslpython_install.sh -f /opt/fsl-5.0.11

ENV FSLDIR="/opt/fsl-5.0.11" \
    PATH="/opt/fsl-5.0.11/bin:$PATH" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLTCLSH="/opt/fsl-5.0.11/bin/fsltclsh" \
    FSLWISH="/opt/fsl-5.0.11/bin/fslwish" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q" \
    POSSUMDIR="/opt/fsl-5.0.11" \
    LD_LIBRARY_PATH="/opt/fsl-5.0.11:$LD_LIBRARY_PATH"


# Installing ANTs 2.3.4 (NeuroDocker build)
ENV ANTSPATH="/usr/lib/ants" \
    PATH="/usr/lib/ants:$PATH"
WORKDIR $ANTSPATH
RUN curl -sSL "https://dl.dropbox.com/s/gwf51ykkk5bifyj/ants-Linux-centos6_x86_64-v2.3.4.tar.gz" \
    | tar -xzC $ANTSPATH --strip-components 1


# Install Matlab MCR at /opt/mcr
ENV MATLAB_VERSION="R2017b" \
    MCR_VERSION="v93"
RUN mkdir /tmp/mcr_install \
    && mkdir /opt/mcr \
    && wget --progress=bar:force -P /tmp/mcr_install https://ssd.mathworks.com/supportfiles/downloads/R2017b/deployment_files/R2017b/installers/glnxa64/MCR_R2017b_glnxa64_installer.zip \
    && unzip -q /tmp/mcr_install/MCR_R2017b_glnxa64_installer.zip -d /tmp/mcr_install \
    && /tmp/mcr_install/install -destinationFolder /opt/mcr -agreeToLicense yes -mode silent
ENV MCRROOT="/opt/mcr/${MCR_VERSION}" \
    LD_LIBRARY_PATH="$MCRROOT/runtime/glnxa64:$MCRROOT/bin/glnxa64:$MCRROOT/sys/os/glnxa64:$LD_LIBRARY_PATH"


# Install CAT12.7 r1743 Standalone at /opt/spm
ENV SPM_VERSION="12" \
    SPM_REVISION="r7771" \
    MCR_INHIBIT_CTF_LOCK=1 \
    SPM_HTML_BROWSER=0
RUN wget --progress=bar:force -P /tmp http://www.neuro.uni-jena.de/cat12/CAT12.7_r1743_${MATLAB_VERSION}_MCR_Linux.zip \
    && unzip -q /tmp/CAT12.7_r1743_${MATLAB_VERSION}_MCR_Linux.zip -d /opt \
    && mv /opt/CAT12.7_r1743_${MATLAB_VERSION}_MCR_Linux /opt/spm \
    && /opt/spm/run_spm12.sh ${MCRROOT} quit \
    && chmod +x /opt/spm/spm12 /opt/spm/*.sh \
    && chmod -R o+r /opt/spm \
    && rm -rf /tmp/*
ENV PATH="${PATH}:/opt/spm/standalone" \
    SPMROOT="/opt/spm" \
    CAT_VERSION="12.7" \
    CAT_REVISION="r1733"


# Installing FreeSurfer v6.0.1
RUN curl -sSL https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/6.0.1/freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.1.tar.gz \
    | tar zxv --no-same-owner -C /opt \
    --exclude="freesurfer/diffusion" \
    --exclude="freesurfer/docs" \
    --exclude="freesurfer/fsfast" \
    --exclude="freesurfer/lib/cuda" \
    --exclude="freesurfer/lib/qt" \
    --exclude="freesurfer/matlab" \
    --exclude="freesurfer/mni/share/man" \
    --exclude="freesurfer/subjects/fsaverage_sym" \
    --exclude="freesurfer/subjects/fsaverage3" \
    --exclude="freesurfer/subjects/fsaverage4" \
    --exclude="freesurfer/subjects/cvs_avg35" \
    --exclude="freesurfer/subjects/cvs_avg35_inMNI152" \
    --exclude="freesurfer/subjects/bert" \
    --exclude="freesurfer/subjects/lh.EC_average" \
    --exclude="freesurfer/subjects/rh.EC_average" \
    --exclude="freesurfer/subjects/sample-*.mgz" \
    --exclude="freesurfer/subjects/V1_average" \
    --exclude="freesurfer/trctrain"

ENV FSL_DIR="/opt/fsl-5.0.11" \
    OS="Linux" \
    FS_OVERRIDE=0 \
    FIX_VERTEX_AREA="" \
    FSF_OUTPUT_FORMAT="nii.gz" \
    FREESURFER_HOME="/opt/freesurfer"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MINC_LIB_DIR="$FREESURFER_HOME/mni/lib" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data"
ENV PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    MNI_PERL5LIB="$MINC_LIB_DIR/perl5/5.8.5" \
    PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH"


# Fic error undefined symbol: FT_Done_MM_Var
RUN rm /lib/x86_64-linux-gnu/libfontconfig.so.1

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users brainprep
WORKDIR /home/brainprep
ENV HOME="/home/brainprep"


# Define entrypoint
# ENTRYPOINT ["python3"]


# Define labels
MAINTAINER Antoine Grigis <antoine.grigis@cea.fr>
LABEL description="FSL 5.0.11, ANTS 2.3.4, FreeSurfer v6.0.1, MATLAB Compiler Runtime R2017b v9.3 and CAT12.7 r1743 standalone docker image for 'brainprep' 0.0.0"
LABEL maintainer="antoine.grigis@cea.fr"
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="brainprep" \
      org.label-schema.description="brainprep - tools for brain MRI deep learning pre-processing." \
      org.label-schema.url="https://github.com/neurospin-deepinsight/brainprep" \
      org.label-schema.schema-version="1.0"
