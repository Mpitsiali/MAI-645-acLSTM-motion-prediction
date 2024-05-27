# Quantitative evaluation and Qualitative evaluation
Running the code_src/pytorch_test_{representation_name}_synthesize_motion.py files will generate predicted and the ground truth motions, where it compares the respective outputs with MSE and outputs the score
May require commenting out code to test trainings with different losses

# Training
Running code_src/gpu_pytorch_train_{representation_name}_aclstm.py files will start training. May require commenting out some paths or the loss function to start training with specific settings
