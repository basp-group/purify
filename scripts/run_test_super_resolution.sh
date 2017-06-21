#!/bin/sh
#TO KEEP

#LOCATION OF PURIFY INSTALLATION
PURIFYPATH="/Users/Arwina/GitHub/purify-wproj/"

#LOCATION OF THE BUILD FOLDER
BUILDDIR=$PURIFYPATH"build/"

#LOCATION OF THE PROGRAM 
CODEDIR="cpp/example/"

#LOCATION OF THE OUTPUTS
NEWRESDIR=$BUILDDIR"outputs/"

#GO TO THE BUILD DIR TO RUN THE PROGRAM
cd $BUILDDIR 
#echo "You are here : "
#pwd
echo " "
########################################

#NAME OF PROGRAM TO RUN
CODENAMER="super_resolution_synthetic_data" #reconstruct using sdmm
echo " "
echo "INFO: IMAGES ARE .fits and UVW COV ARE .vis"

#IMAGE NAME WITHOUT .fits located in data/images/
IMAGENAME="M31.64.fits"

#NAME OF UV COVERAGE FILE WITHOUT ".vis" located in data/coverages
VISIBILITIES="ASKAP_1Ghz_dec-30_ra60h0_2h_2400dt.uvw"
#uvw_random_128pix_M816"

#ENERGY LEVEL FOR G MATRIX - file
ENERGYG=0.9999999

#ENERGY LEVEL FOR C MATRIX
ENERGYC=0.9999999

#INPUT SNR
ISNR=40

#Resolution
RES=1 

#W-FRACtion
WFRAC=-1

#TEST NBER
RUN=1

#Obs. freq
FREQ=1e9

#########################################
#GET ARGS

while getopts :m:v:g:c:w:t:s:r:f:h opt; do
  case $opt in
    h) echo "Help"
       echo "-v :vis file name"
       echo "-m :sky model name"
       echo "-g :energy level for the G mat. ;DEFAULT=$ENERGYG"
       echo "-c :energy level for the Chirpmatrix; DEFAULT=$ENERGYC"
       echo "-s :input SNR; DEFAULT=$ISNR"
       echo "-t :index of run; DEFAULT=$RUN"
       echo "-w :w  fraction; DEFAULT =$WFRAC"
       echo "-r :uv - scale; DEFAULT =$RES"
       echo "-f :uv - scale; DEFAULT =$FREQ"
       exit 0
       ;;
    m) 
       IMAGENAME=$OPTARG
       ;;
    v) 
       VISIBILITIES=$OPTARG
       ;;
    g) 
       ENERGYG=$OPTARG
       ;;
    c) 
       ENERGYC=$OPTARG
       ;;
    w) 
       WFRAC=$OPTARG     
       ;;

    t) 
       RUN=$OPTARG
      ;;
    s) 
       ISNR=$OPTARG
       ;; 
    r) 
       RES=$OPTARG
       ;;
    f)
       FREQ=$OPTARG
       ;; 
    \?)
      echo "oups .. "
      echo "Seems a wrong arg have been passed" 
      echo "Check help using -h"
      exit 1
    ;;
  esac
done

#############################################
#COMMANDS

OUTPUTDIR="SR_"$VISIBILITIES"."$IMAGENAME".W"$WFRAC".RES"$RES".ISNR"$ISNR"/"
mkdir -p $NEWRESDIR$OUTPUTDIR
VIS=$VISIBILITIES
VISIBILITIES=$VISIBILITIES".vis"

echo " "
cd $BUILDDIR

echo "/* --------- RECONSTRUCT  IMAGE #SPARSITY LEVEL OF THE G MATRIX (w random) KB KERNELS------------*/"

./$CODEDIR$CODENAMER $IMAGENAME $PURIFYPATH"data/coverages/"$VISIBILITIES $ENERGYG $ENERGYC $OUTPUTDIR $ISNR $WFRAC $RUN $RES $FREQ| tee "-a" $NEWRESDIR"SR_"$VIS"."$IMAGENAME".W"$WFRAC".RES"$RES".ISNR"$ISNR"/Rand_RECONSTRUCT"$RUN".log"

echo ""
echo "purify is done."
