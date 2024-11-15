# Classification of pseudo random noise

This is the code for my bachelor thesis on "Classification of analog generated pseudo-random noise". Here I build a library to analyse noise using different methods.

With this programm, we create all necessary data to classify and analyse random noise.

To run the program efficiently, put all the noise data into the 'data' folder.

Before running the program, please run:

> pip install -r requirements.txt

to install all required libraries.

Afterwards, you can run

> python main.py

The program will collect the data under the folder 'results'.

For the data format please check that it is a CSV file with the following structure:

Time (ms);CH1 (V)
float;float