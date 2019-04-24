#!/bin/bash
echo "Unzipping lubm1 input graphs"
unzip -qn data/lubm1_intact/graphs_with_descriptions.zip -d data/lubm1_intact/
echo "Unzipping lubm1 inference graphs"
unzip -qn data/lubm1_intact/jena_inference_with_descriptions.zip -d data/lubm1_intact/
cd code
python prepare_lubm_data.py