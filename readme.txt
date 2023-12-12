-------------------------------------------------------

Code for Dual-Level Curriculum Meta-Learning under Noise with Theoretical Guarantees

-------------------------------------------------------

This package contains source code for submission:Dual-Level Curriculum Meta-Learning under Noise with Theoretical Guarantees


The required packages are listed in the file: requirements.txt, please intall through the following command: 

pip install -r requirements.txt

The code is stored in the folder code. 

The synthestic noisy labels with different rates are stores in the folder data_tensor under folders Asymmetric_label and 
Symmetric_label for asymmetric noisy label and symmetric noisy label, respectively.

To download the CIFAR-100N real-world noisy datasets, please use the links: http://noisylabels.com

To run the algorithm, change the path to the current directory in the command window, and run the : train_dcml.py

To test the model, run the : test.py