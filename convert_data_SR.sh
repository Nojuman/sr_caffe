#!/usr/bin/env sh

EXAMPLE=examples/SuperResolution/SR_data
DATA=examples/SuperResolution
BUILD=build/examples/SuperResolution
TRAIN_DATA_ROOT=examples/SuperResolution/SR_prepare_data/train/
TEST_DATA_ROOT=examples/SuperResolution/SR_prepare_data/test/
BACKEND="lmdb"


echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/SR_trainLR_${BACKEND}
rm -rf $EXAMPLE/SR_trainSR_${BACKEND}
rm -rf $EXAMPLE/SR_testLR_${BACKEND}
rm -rf $EXAMPLE/SR_testSR_${BACKEND}

GLOG_logtostderr=1 \
  $BUILD/convert_SR.bin \
  --resize_height=33 \
  --resize_width=33 \
  --gray \
  $TRAIN_DATA_ROOT \
  $DATA/SR_prepare_data/train/LRFile.txt \
  $EXAMPLE/SR_trainLR_${BACKEND} --backend=${BACKEND}

GLOG_logtostderr=1 \
  $BUILD/convert_SR.bin \
  --resize_height=33 \
  --resize_width=33 \
  --gray \
  $TEST_DATA_ROOT \
  $DATA/SR_prepare_data/test/LRFile.txt \
  $EXAMPLE/SR_testLR_${BACKEND} --backend=${BACKEND}

GLOG_logtostderr=1 \
  $BUILD/convert_SR.bin \
  --resize_height=21 \
  --resize_width=21 \
  --gray \
  $TRAIN_DATA_ROOT \
  $DATA/SR_prepare_data/train/SRFile.txt \
  $EXAMPLE/SR_trainSR_${BACKEND} --backend=${BACKEND}

GLOG_logtostderr=1 \
  $BUILD/convert_SR.bin \
  --resize_height=21 \
  --resize_width=21 \
  --gray \
  $TEST_DATA_ROOT \
  $DATA/SR_prepare_data/test/SRFile.txt \
  $EXAMPLE/SR_testSR_${BACKEND} --backend=${BACKEND}

echo "Done"




