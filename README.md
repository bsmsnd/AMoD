# AMoD
Autonomous Mobility On-Demand Using Deep Learning

Getting Started
----------------

This AMoD Model should be run with [AMoD Simulator](https://github.com/idsc-frazzoli/amod). It was part of NIPS 2018 Challenges and was developed by the Institute of Dynamic Systems and Control at ETH Zürich in collaboration with the Institute for Transport Planning and Systems at ETH Zürich. We give many thanks to their effort. 

Once AMoDeus™ has been configured and AidoHost is launched properly, launch `AidoGuest.py` to begin dispatching. 

This model is based on Pytorch, and have enabled GPU acceleration if available.

Using Ideal Simulator
--------------------

We have implemented a ideal(mini) simulator to allow evaluation of the model in a simple way. Simply run the simulator.py and it will call all necessary modules. Visualization is also enabled in which pentagrams represent requests and circles represent the vehicles. Color will also indicate weather the request has been accepted, or whether the vehicle is available for pickups.

Note that the simulator is only for validation, and should not be used for large-scale learning becasuse it has 2 assumptions that are not valid in real traffic: it allows vehicles to move in any direction, and requests are generated randomly. 
 
Run Different Models
--------------------

To run different models, 
- Change this line `from DispatchingLogic import DispatchingLogic` in the `simulator.py`. The name of the model should correspond to the file name as listed in the folder.
- Change hyperparameters in `constant.py` and 'simulator.py'.
- To run Amodeus simulator, please refer to the tutorial under the folder "Amodeus tutorial".

