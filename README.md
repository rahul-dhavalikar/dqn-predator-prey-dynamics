# SIMULATION OF PREDATOR PREY DYNAMICS USING DEEP REINFORCEMENT LEARNING
### CS 275: Artificial Life for Computer Graphics and Vision

#### Group Members:
1. Akshay Sharma
2. Anoosha Sagar
3. Maithili Bhide
4. Rahul Dhavalikar

#### Contents of this Repository:
1. **_index.html_**: web page of the project for quick understanding and navigation of the project
2. **_report.pdf_**: report of the project
3. **_videos_**: this folder contains all the videos of our simulation
4. **_img_**: this folder contains all the images used in the webpage and report
5. **_code_**: code of the project
6. **_code/ddqn.py_**: file containing the DDQN class
7. **_code/ddqn_run.py_**: main driver program of the project
8. **_code/multiagent/scenarios_**: this folder contains all the different scenarios used in our project
9. **_code/run_**: this folder contains all the shell scripts required for training and testing our project

#### How to Train a Scenario
To train a scenario, run any of the training shell scripts in **_code/run_**.
For example, to train a simple 1v1 scenario, run the following script
```
./simple_1v1_train.sh
```

#### How to Test Out a Scenario
To test a scenario after training, run the corresponding testing shell script in **_code/run_**.
For example, to test the simple 1v1 scenario after successfully training it, run the following script
```
./simple_1v1_test.sh
```
By default, the testing shell scripts contain the path to our pretrained models **_(e.g., ./save/simple_1v1_final)_**. While testing your own models, this path should be changed to point to your model.
