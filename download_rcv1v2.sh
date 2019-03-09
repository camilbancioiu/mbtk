#!/usr/bin/env bash

TOKENS_TRAIN=lyrl2004_tokens_train.dat
TOKENS_TEST_0=lyrl2004_tokens_test_pt0.dat
TOKENS_TEST_1=lyrl2004_tokens_test_pt1.dat
TOKENS_TEST_2=lyrl2004_tokens_test_pt2.dat
TOKENS_TEST_3=lyrl2004_tokens_test_pt3.dat

TOKENS_ALL=lyrl2004_tokens_all.dat

INDUSTRY_HIERARCHY=oa5.rcv1.industries.hier.txt
ALL_DOCUMENT_IDS=oa7.rcv1v2-ids.txt

TOPIC_ASSIGNMENTS_NO_EXT=oa8.rcv1-v2.topics.qrels
TOPIC_ASSIGNMENTS=oa8.rcv1-v2.topics.qrels.txt

INDUSTRY_ASSIGNMENTS_NO_EXT=oa9.rcv1-v2.industries.qrels
INDUSTRY_ASSIGNMENTS=oa9.rcv1-v2.industries.qrels.txt

mkdir -p dataset_rcv1v2
cd dataset_rcv1v2
mkdir -p token
cd token

if [ ! -f ${TOKENS_TRAIN} ]; then
  wget -O ${TOKENS_TRAIN}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/${TOKENS_TRAIN}.gz
  gzip -d ${TOKENS_TRAIN}.gz
fi

if [ ! -f ${TOKENS_TEST_0} ]; then
  wget -O ${TOKENS_TEST_0}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/${TOKENS_TEST_0}.gz
  gzip -d ${TOKENS_TEST_0}.gz
fi

if [ ! -f ${TOKENS_TEST_1} ]; then
  wget -O ${TOKENS_TEST_1}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/${TOKENS_TEST_1}.gz
  gzip -d ${TOKENS_TEST_1}.gz
fi

if [ ! -f ${TOKENS_TEST_2} ]; then
  wget -O ${TOKENS_TEST_2}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/${TOKENS_TEST_2}.gz
  gzip -d ${TOKENS_TEST_2}.gz
fi

if [ ! -f ${TOKENS_TEST_3} ]; then
  wget -O ${TOKENS_TEST_3}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/${TOKENS_TEST_3}.gz
  gzip -d ${TOKENS_TEST_3}.gz
fi

if [ ! -f ${TOKENS_ALL} ]; then
  cat ${TOKENS_TRAIN} ${TOKENS_TEST_0} ${TOKENS_TEST_1} ${TOKENS_TEST_2} ${TOKENS_TEST_3} > ${TOKENS_ALL}
fi

cd ..

if [ ! -f ${INDUSTRY_HIERARCHY} ]; then
  wget -O ${INDUSTRY_HIERARCHY} http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a05-industry-hierarchy/rcv1.industries.hier
fi

if [ ! -f ${ALL_DOCUMENT_IDS} ]; then
  wget -O ${ALL_DOCUMENT_IDS}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz
  gzip -d ${ALL_DOCUMENT_IDS}.gz
fi

if [ ! -f ${TOPIC_ASSIGNMENTS} ]; then
  wget -O ${TOPIC_ASSIGNMENTS_NO_EXT}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
  gzip -d ${TOPIC_ASSIGNMENTS_NO_EXT}.gz
  mv ${TOPIC_ASSIGNMENTS_NO_EXT} ${TOPIC_ASSIGNMENTS}
fi

if [ ! -f ${INDUSTRY_ASSIGNMENTS} ]; then
  wget -O ${INDUSTRY_ASSIGNMENTS_NO_EXT}.gz http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a09-industry-qrels/rcv1-v2.industries.qrels.gz
  gzip -d ${INDUSTRY_ASSIGNMENTS_NO_EXT}.gz
  mv ${INDUSTRY_ASSIGNMENTS_NO_EXT} ${INDUSTRY_ASSIGNMENTS}
fi
