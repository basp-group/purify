#!/bin/sh
#TO KEEP

#LOCATION OF PURIFY INSTALLATION
PURIFYPATH="/Users/Arwina/GitHub/purify-cpp-wprojection-omp/"

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
CODENAME="w_correction_example_padmm_upsampling" # generate data
echo " "
echo "INFO: IMAGES ARE .fits and UVW COV ARE .vis"

#IMAGE NAME WITHOUT .fits located in data/images/
IMAGENAME="M31.64"

#NAME OF UV COVERAGE FILE WITHOUT ".vis" located in data/coverages
VISIBILITIES="ASKAP_1Ghz_dec-30_ra60h0_2h_2400dt.uvw"
#uvw_random_128pix_M816"

#ENERGY LEVEL FOR G MATRIX
ENERGYG=0.9999999999

#ENERGY LEVEL FOR C MATRIX
ENERGYC=0.9999999999

#INPUT SNR
ISNR=30

#FOV
FOV=2 #NOT USED AT THE MOMENT

#W-FRACtion
WFRAC=-1  # -1: keeping the original w values otherwise WFRAC in [0 1]

#TEST NBER
RUN=1

#########################################
#GET ARGS

while getopts :m:v:g:c:w:t:s:h opt; do
  case $opt in
    h) echo "Help"
       echo "-v :vis file name; located in data/coverages/"
       echo "-m :sky model name; located in data/images/"
       echo "-g :energy level for the G matrix; DEFAULT=$ENERGYG"
       echo "-c :energy level for the Chirpmatrix; DEFAULT=$ENERGYC"
       echo "-s :input SNR; DEFAULT=$ISNR"
       echo "-t :index of run; DEFAULT=$RUN"
       echo "-w :WFRAC = -1 if keeping the original values otherwise  WFRAC in [0 1]; DEFAULT =$WFRAC"
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
    \?)
      echo "Seems a wrong arg have been passed"
      echo "Check help using -h"
      exit 1
    ;;
  esac
done

#############################################
#COMMANDS

OUTPUTDIR="TEST_"$VISIBILITIES"."$IMAGENAME".WFRAC"$WFRAC".ISNR"$ISNR"/"
mkdir -p $NEWRESDIR$OUTPUTDIR
VIS=$VISIBILITIES
VISIBILITIES=$VISIBILITIES".vis"

echo " "
cd $BUILDDIR
echo "--------------------------------------------------"
echo "Input Parameters "
echo "--------------------------------------------------"
echo "image:  "$IMAGENAME 
echo "vis:  "$VISIBILITIES 
echo "energyG:  "$ENERGYG 
echo "energyC:  "$ENERGYC
echo "fov:  "$FOV
echo "isnr:  "$ISNR
echo "wfrac:  "$WFRAC
echo "RUN NBER:  "$RUN
echo "Output folder:  "$OUTPUTDIR
echo " "

echo "/* --------- STARTING TEST ------------*/"

./$CODEDIR$CODENAME $IMAGENAME $PURIFYPATH"data/coverages/"$VISIBILITIES $ENERGYG $ENERGYC $OUTPUTDIR $ISNR $WFRAC $RUN $FOV | tee "-a" $NEWRESDIR"TEST_"$VIS"."$IMAGENAME".WFRAC"$WFRAC".ISNR"$ISNR"/TEST_RUN.EC"$ENERGYC".WFRAC"$WFRAC".EG"$ENERGYG".T"$RUN".log"

echo "----"
echo "purify is done."

