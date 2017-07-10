#!/bin/bash
# My first script

python retrain.py \
  --bottleneck_dir=bottlenecks \
  --how_many_training_steps=50 \
  --model_dir=inception \
  --summaries_dir=training_summaries/basic \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --image_dir=data

# change parameters above to set: 
#	bottleneck save directory; 
#	number of iterations; 
#	directory of the pretrained model; 
#	directory to save training logs; 
#	output finetuned model; 
#	label order after training; 
#	directory of training and validation data
