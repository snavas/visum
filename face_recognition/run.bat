python retrain.py --bottleneck_dir=bottlenecks --how_many_training_steps=50 --model_dir=inception --summaries_dir=training_summaries/basic --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=data
python run_test.py
pause
