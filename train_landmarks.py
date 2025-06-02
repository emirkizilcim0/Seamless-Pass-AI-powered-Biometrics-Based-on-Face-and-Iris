import dlib

options = dlib.shape_predictor_training_options()
options.be_verbose   = True
options.oversampling_amount = 50       # from 300 to 20 (migth cause overfitting)
options.tree_depth = 5                   # slightly smaller trees
options.cascade_depth = 10                
options.num_trees_per_cascade_level = 250
options.nu = 0.1                     
options.num_threads = 0                 # auto detect all CPU cores

# Training
dlib.train_shape_predictor("training.xml", "my_shape_predictor_68.dat", options=options)
print("training is done")

# Testing
error = dlib.test_shape_predictor("testing.xml", "my_shape_predictor_68.dat")
print(f"the error value is: {error}")