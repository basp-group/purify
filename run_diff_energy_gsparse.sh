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
CODENAMER="w_correction_padmm_reconstruction_diff_GS_all" #reconstruct using sdmm
echo " "
echo "INFO: IMAGES ARE .fits and UVW COV ARE .vis"

#IMAGE NAME WITHOUT .fits located in data/images/
IMAGENAME="M31.64"

#NAME OF UV COVERAGE FILE WITHOUT ".vis" located in data/coverages
VISIBILITIES="ASKAP_1Ghz_dec-30_ra60h0_2h_2400dt.uvw"
#uvw_random_128pix_M816"

#ENERGY LEVEL FOR G MATRIX - file
ENERGYG="gList.txt"

#ENERGY LEVEL FOR C MATRIX
ENERGYC=0.99999999

#INPUT SNR
ISNR=60

#FOV
FOV=2 #NOT USED AT THE MOMENT

#W-FRACtion
WFRAC=-1

#TEST NBER
RUN=1

#########################################
#GET ARGS

while getopts :m:v:g:c:w:t:s:h opt; do
  case $opt in
    h) echo "Help"
       echo "-v :vis file name"
       echo "-m :sky model name"
       echo "-g :energy level - file name DEFAULT=$ENERGYG"
       echo "-c :energy level for the Chirpmatrix; DEFAULT=$ENERGYC"
       echo "-s :input SNR; DEFAULT=$ISNR"
       echo "-t :index of run; DEFAULT=$RUN"
       echo "-w :w  fraction; DEFAULT =$WFRAC"
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

OUTPUTDIR="RTEST_DIFF_G_"$VISIBILITIES"."$IMAGENAME".W"$WFRAC".ISNR"$ISNR"/"
mkdir -p $NEWRESDIR$OUTPUTDIR
VIS=$VISIBILITIES
VISIBILITIES=$VISIBILITIES".vis"

echo " "
cd $BUILDDIR
#echo "/*-----------Input Parameters--------------------*/"
#echo "image:  "$IMAGENAME 
#echo "vis:  "$VISIBILITIES 
#echo "energyG:  "$ENERGYG 
#echo "energyC:  "$ENERGYC
#echo "fov:  "$FOV
#echo "isnr:  "$ISNR
#echo "wfrac:  "$WFRAC
#echo "RUN NBER:  "$RUN
#echo "Output folder:  "$OUTPUTDIR
#echo " "


#
echo "/* --------- RECONSTRUCT  IMAGE #SPARSITY LEVEL OF THE G MATRIX ------------*/"

./$CODEDIR$CODENAMER $IMAGENAME $PURIFYPATH"data/coverages/"$VISIBILITIES $PURIFYPATH$ENERGYG $ENERGYC $OUTPUTDIR $ISNR $WFRAC $RUN $FOV | tee "-a" $NEWRESDIR"RTEST_DIFF_G_"$VIS"."$IMAGENAME".W"$WFRAC".ISNR"$ISNR"/RECONSTRUCT.log"

echo ""
echo "purify is done."
