while true; do
	python .\\models-master\\research\\object_detection\\model_main.py --pipeline_config_path=.\\models\\model\\pipeline.config --model_dir=.\\models\\model --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr
done