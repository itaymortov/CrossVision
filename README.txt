Blocked Crosswalk Detection Project
This project is designed to detect when a crosswalk is blocked by an object, and alert the driver. We are using YOLOv8 models to detect the crosswalk and the objects that are blocking it as well as traffic lights.

We recommend watching the guidance video on how the project works and how to install and run it. 
Youtube link: https://youtu.be/EqXZXgc48fU

Installation

- To install, download the ZIP folder with our files to your computer (choose a location of your convenience).
- Check if you have python 3.9 installed on your computer (anaconda is recommended) if not, install it ,if you have it, you can skip this step.
- In the TERMINAL write the command 'pip install -r requirements.txt' (you need to go into the folder where the project is). If you use PyCharm you can open a new project in the folder that contains the project's files and write the above command in the PyCharm TERMINAL.
- After that you can simply run the project in PyCharm (to run from the TERMINAL you can simply write the line 'python main.py' (if you want to change the source of the project you can do it inside the code in the line 218 -  detector = CrossVisionDetection('insert source here')
- Note: while running, if you want to exit you can press the 'q' key and the software will stop.

Customization
You can customize the program by changing the thresholds for crosswalk and object overlap, or by changing the alert mechanism (i.e sound or light). You can also modify the YOLOv8 models used for object detection, or use a different model altogether.
