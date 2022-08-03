#!/bin/sh
# Copyright (c) 2016, David Stutz
# Contact: david.stutz@rwth-aachen.de, davidstutz.de
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Supposed to be run from within examples/.

# ============================================= bsd =================================================
SUPERPIXELS=("96" "216" "384" "600" "864" "1176" "1536" "1944")
IMG_PATH=/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/data_preprocessing/test
GT_PATH=/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/data_preprocessing/test/map_csv

for SUPERPIXEL in "${SUPERPIXELS[@]}"
do
   echo $SUPERPIXEL
       ../bin/eval_summary_cli /home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/output/test_4/bsd/test_multiscale_enforce_connect/SPixelNet_nSpixel_${SUPERPIXEL}/map_csv  $IMG_PATH $GT_PATH
done
# ============================================= bsd =================================================


# ============================================= nyu =================================================
SUPERPIXELS=("192" "432" "768" "1200" "1728" "2352" "3072" "3888")
IMG_PATH=/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/nyu_test_set/img
GT_PATH=/home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/nyu_test_set/label_csv

for SUPERPIXEL in "${SUPERPIXELS[@]}"
do
   echo $SUPERPIXEL
       ../bin/eval_summary_cli /home/DISCOVER_summer2022/yanzj/workspace/code/Cerberus/output/test_4/nyu/test_multiscale_enforce_connect/SPixelNet_nSpixel_${SUPERPIXEL}/map_csv  $IMG_PATH $GT_PATH
done
# ============================================= bsd =================================================