# A Software Architecture for Multimodal Semantic Perception Fusion

## Objective of the Project
The project proposes an implementation of architecture to fuse geometric features computed from point clouds and Convolution Neural Network (CNN) classifications, based on images.

## The System’s Architecture

### Overall Architecture
Architecture fuses geometric features computed from N-perception modules.
This implementation uses two perception modules, point clouds (PITT) and Convolution Neural Network (CNN - TensorFlow).
Between a Perception module and Feature selector module, there is an Adapter to implement a standard message for each perception module.
Then, Feature Selector receives all the information acquired by the sensors and produces two outputs. An intersection of data that identifies any common features between the objects from different perception modules and an union data containing all information from the sensors.
Correlation Table Manager takes union data and __........__
Reasoner takes in input the output of Correlation Table Manager and generates an index of correlation for objects recognized from different perception modules __..........__
Finally the Feature Selector join the data from Reasoner and Feature Selector module by searching for objects' IDs and assorting all features coming from different perceptive modules. Then, it returns an output message for each object recognized comprehensive with all information collected by the various sensors.


<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/Schermata%202019-11-14%20alle%2014.55.55.png">
</p>

#### Messages
##### Feature.msg

```
string types
string name
string[] value
```
##### Obj.msg

```
feature[] obj
```
##### Adapter.msg

```
uint32 id_mod
obj[] adap
```
##### CommonFeature.msg

```
adapter[] common
```
##### FeatureMatcher.msg

```
commonFeature[] matcher
```
##### Corr.msg

```
string first_percepted_object
string second_percepted_object
float32 correlation
```
##### CorrelationTable.msg

```
corr[] table
```
##### Record.msg

```
string[] rec
float32 corr
```
##### OutputReasoner.msg

```
record[] lines
```
##### MatcherObj.msg

```
obj[] sameObj
float32 correlation
```
##### Matcher_out.msg

```
feature[] obj
float32 correlation
```
#### Messages Rappresentation

<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/Schermata%202019-11-14%20alle%2016.39.25.png">
</p>

### Description of the Modules
#### PITT Perception module
It describes all the modules within the architecture, i.e, (i) the inputs, (ii) the internal working, and (iii) the outputs.
* __Input__: 
* __Output__:
* __Publisher__: [__O1__]

#### Tensorflow Perception module
It describes all the modules within the architecture, i.e, (i) the inputs, (ii) the internal working, and (iii) the outputs.
* __Input__: 
* __Output__:
* __Publisher__: [__O2__]

#### Adapter module
It's a module between a perception module and Feature selector module and makes a standard message for each perception module.  We provide an adapter for the Pitt module and another for the Tensorflow module. 
To add another perception module, it needs to implement a different adapter.
* __Input__: a type of message of a perception module
* __Output__: an adapter.msg
* __Publisher__: /outputAdapterPitt | /outputAdapterTensor

#### Features Selector module
It receives all the information acquired by the sensors, storing them inside a buffer.
It produces two outputs: the first one transmits the information saved in the buffer to the "Feature Matcher" and the other output transmits to the "Correlation Table Manager" module.
For the second output, the module identifies any common features between the objects recognized by the sensors (for example, shape or color).
* __Input__: an adapter.msg
* __Output__: a selectorMatcher.msg
* __Publisher__: /featureScheduler/pubIntersection [__F__]| /featureScheduler/pubUnion [__R__]

#### Features Matcher module
In this module, the data received from the Reasoner and the Feature Selector are joined.
From the Reasoner, this module receives the objects' IDs detected by different perception modules plus the degree of correlation (a number that indicates the reliability of the analysis obtained).
From Feature Selector, Feature Matcher receives all data of objects detected.
The Features Matcher finds the information by searching for objects' IDs and assorting all features coming from different perceptive modules. Then, it returns an output message for each object recognized comprehensive with all information collected by the various sensors.
* __Input__: a selectorMatcher.msg | outputReasoner.msg
* __Output__: a matcher_out.msg
* __Publisher__: /featureMatcher/dataPub [__P__]

#### Correlation Table Manager.
Correlation Table Manager receives messages coming from the "Feature Selector" node and computes correlations between objects perceived by different perception modules with features in common. Indeed, it builds binomial(N,2) correlation tables (where N is the number of perceptive modules with at least one feature in common) and appropriately sends these data to the "Reasoner" module
* __Input__: selectorMatcher.msg
* __Output__: correlationTable
* __Publisher__: /correlationTables [__T__]

#### Reasoner module
It describes all the modules within the architecture, i.e, (i) the inputs, (ii) the internal working, and (iii) the outputs.
* __Input__: 
* __Output__:
* __Publisher__: [__U__]

# Implementation

## correlation_functions & tableMatcher


## Contents:

### 1. [Structure of the scripts](#desc1)
##### 1.1 [Correlation functions](#desc1.1)
##### 1.2 [Table matcher](#desc1.2)

### 2. [Steps to add a feature](#desc2) 
##### 2.1 [Correlation functions](#desc2.1)	
##### 2.2 [Table matcher](#desc2.2)

### 3. [Parameters setup](#desc3)

### 4. [Test phase](#desc4)
#### 4.1 [Input](#desc4.1)
#### 4.2 [Processing](#desc4.2)
#### 4.3 [Outpu](#desc4.3)
#### 4.4 [Exception management](#desc4.4)

### 5. [Commands to launch the package](#desc5)

<a name="desc1"></a>
### 1. Structure of the scripts

The executables implementing the Correlation Table Manager are essentially two: correlation_functions.py and tableMatcher.py. 

<a name="desc1.1"></a>
#### 1.1. Correlation function

correlation_functions.py file contains all the methods useful to compute distances between features.
The euclidean distance function is defined at the beginning of the file and it is used to compute the distance coefficient for the following methods:

- get_dist_2d: 		Function implementing distance between 2D coordinates

- get_dist_3d:		Function implementig distance between 3D coordinates

- get_colour_name:	Function converting colour name into RGB values. Then computing distances 			between RGB channels.

- get_RGB:		Function computing distances between RGB channels.

- get_HSV:		Function computing distances between HSV channels.



The above functions are characterized by a similar structure: 

1. - Two “for” cycles. 
- Two indices (index l and index k) are necessary to consider an object of one perceptive system with an object of another perceptive system. 
- The two generic objects mentioned share necessarily at least two features (‘id’ and ‘feature_in_common’). 
- The index j defined in tableMatcher.py allows to select the right feature in common.

2. Definition of delta list() composed of all the differences between components
3. Call the distance function. In this case euclidean_distance. 
4. Updating of the coefficients table by summing up the distances multiplied by a normalizing factor in the position [l][k]

A particular case is  get_colour_name. At the beginning of the function we define a dictionary containing all the correspondences between a colour name and RGB values. 
If one colour is not present in the dictionary a message is displayed in the terminal reporting the name of the missing colour. Besides the computation of the table is stopped and the coefficient table is restored at the initial state. Therefore, in this case, the coefficient table is not updated with any distance computed by get_colour_name. 

<a name="desc1.2"></a>
#### 1.2 Table matcher

tableMatcher.py file perform the following task:
1. For each pair of perceptive systems a method suitable to compute the distance between the recognized feature in common is called. Therefore it is created a table for each pair of  perceptive system that share at least two features  (‘id’ and ‘feature_in_common’).
2. Once multiple matrices of coefficients are created, the single coefficients are extracted from the matrices and they are packed into a correlationTable() message

The first task is performed thanks to the presence of if...elif...else statements that are necessary to verify that the detected “feature_name” is really consistent with one of the  available names of the
features.  Therefore the corresponding implemented distance function is called. 

If the feature is not recognized a message is displayed on the terminal suggesting to implement a method for the corresponding not recognized feature. At this point if the “id” feature is the only feature in common, the table is not computed for the considered pair of perceptive system. 

<a name="desc2"></a>
### 2. Steps to add a feature

In this section we want to describe the steps necessary to add new methods for corresponding not recognized feature. 

<a name="desc2.1"></a>
#### 2.1 Correlation function

1. Define a new function in correlation_function.py file

<a name="desc2.2"></a>
#### 2.2 Table matcher

1. Import the new function in tableMatcher.py
2. Add a new parameter setup 
3. Add an elif statement t to the already available ones. This statement verify that the detected “feature_name” is really consistent with the name of the feature and it allows to call the corresponding implemented distance function

<a name="desc3"></a>
### 3. Parameters setup

The parameters already defined  are the following ones:

alfa_2d: 		Normalizing factor for the feature “pose_2d”

beta_3d: 		Normalizing factor for the feature “pose_3d”

gamma_color_rgb: 	Normalizing factor for the feature “color_rgb”. It is equal to 1/255.

delta_hue: 		Normalizing factor for the hue component of the feature “color_hsv”. 			It is equal to 1/360.

delta_sat:		Normalizing factor for the saturation component of the feature 				“color_hsv”. It is equal to 1/100.

delta_value:		Normalizing factor for the value component of the feature 					“color_hsv”. It is equal to 1/100.

w: 			Normalizing factor for the coefficient table. It allows to normalize 
			values in the interval [0,1]. It is equal to 10

rate: 			publishing rate. It is equal to 20

Parameters initialization is performed in the launch file matcher.launch. 

<a name="desc4"></a>
### 4. Test phase

In this section we explain the workflow of ROS nodes. We start to visualize the input of our component, through the analysis of talkerMatcher.py . After the input we analyze the processing of the code, that compute the tables. At the end the values of tables are sent as output of tableMatcher.py. All of these phase are described in the following steps.

<a name="desc4.1"></a>
#### 4.1 Input

In order to test our component we have created a script talkerMatcher.py that simulates the presence of three perceptive modules. The first one perceives three objects while the second and third one perceive only two objects. The first and the third perceptive modules share “color_name” feature, the first and the second perceptive systems share “pose_2d” and “pose_3d” feature.
TalkerMatcher.py starts the node “dummyTalker” and it publishes selectorMatcher message into the “/featureScheduler/pubIntersection” topic. 
From the terminal we can observe the structure of the published message running the command 
rostopic echo /featureScheduler/pubIntersection. 

<p align="left"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/1.png">
</p>
<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/2.png">
</p>

<p align="right"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/3.png">
</p>

<a name="desc4.2"></a>
#### 4.2 Processing




<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/4.png">
</p>






The computed  tables are two. The first table displays the normalized correlation coefficient between the first and the third perceptive system while the second table between the first and the second perceptive system. The sizes of the matrices are 3x2 because the first perceptive system detects three object while the second and the third detect only two object. 
In order to understand the values of the coefficient it is useful to discuss values of the second matrix. The features in common between the first and the second perceptive systems are “pose_2d” and “pose_3d”. The values in position [l][j] of the second table correspond to the comparison between the l-th object of the first perceptive system and the j-th object of the second perceptive system. The value in “0.97235449” in position [3][1] is greater than the value “0.29113921” in position [1][1]. It means that it is more probable that the first object detected by the second perceptive module corresponds to the third object detected by the first perceptive module.

In fact:
the first object perceived by the second perceptive system has feature values:

pose_2d: [7.1;8.2]
pose_3d: [7.1;8.2;9.3]

the first object perceived by the first perceptive system has feature values:

pose_2d: [1;2]
pose_3d: [1;2;3]

the third object perceived by the first perceptive system has feature values:

pose_2d: [7;8]
pose_3d: [7;8;9]

It is clearly visible that the first object perceived by the second perceptive system is more correlated with the third object perceived by the first perceptive system with respect to the first object perceived by the first perceptive system.








<a name="desc4.3"></a>
#### 4.3 Output

Once multiple matrices of coefficients are created, the single coefficients are extracted from the matrices and they are packed into a correlationTable() message. A correlationTable() message is characterized by the following structure:


<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/5.png">
</p>

<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/6.png">
</p>


In the end the correlationTable() message is published to the topic /correlationTables.
From the terminal we can observe the structure of the published message running the command 
rostopic echo /correlationTables. 


<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/7.png">
</p>

As we can observe a correlationTable() message is made of 12 correlation. Since the two matrices have sizes 3x2 each one has 6 elements. In total there are 12 elements. The fields  first_percepted_object and second_percepted_object are filled with strings having the following structure:  “Id_perceptive_module”+ “Id_object”. For example the first corr() message visible in the above image contain the correlation value between the object with Id_object = 0 detected by the perceptive module having Id_perceptive_module = 1 and the object with Id_object = 7 detected by the perceptive module having Id_perceptive_module = 3. The value of correlation is 0.8595139.

<a name="desc4.4"></a>
#### 4.4 Exception management

We suppose to introduce a feature name that it will be not recognized by the tableMatcher.py processing. Assuming to modify the feature name color_name into the feature name color. When the if… elif… else statement will be executed a method to compute the distance for  the feature name will not be found. Therefore a message containing the unrecognized feature name will be displayed on the terminal. 
If it is the unique feature in common a correlation table will not be computed. In other cases the distances related to this feature will not considered. 

<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/9.png">
</p>

It is possible to observe that only one matrix is computed. In fact color_name represents the only feature in common between the first and the third perceptive objects. 
Now we assume to modify only pose_2d feature in the fake name pose. Below we visualize the result:

<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/10.png">
</p>

In this case a message is displayed reporting the unrecognized feature name pose. Unlike the case of color here the computation of tables goes on because there is the remaining feature in common pose_3d between the first and the second perceptive system. 

Finally we consider a color that it is not available in the dictionary. Therefore it is not possible to find a correspondence with an RGB value. If one colour is not present in the dictionary a message is displayed in the terminal reporting the name of the missing colour. Besides the computation of the table is stopped and the coefficient table is restored at the initial state. Therefore, in this case, the coefficient table is not updated with any distance computed by get_colour_name. 

We assume to consider the color terra di siena that it is not present in the dictionary. We report the result. 


<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/immagini/11.png">
</p>


We can observe the message displayed. Besides since color_name is the unique feature in common between the first and the third perceptive system and the original coefficient matrix is in this case composed of only zeros the correlation table is not computed.  

<a name="desc5"></a>
### 5. Commands to launch the package

The terminal commands in order to run the code:

cd catkin_workspace

source devel/setup.bash

catkin_make

cd src/sofar_multimodal/scripts/

chmod +x correlation_functions.py
chmod +x tableMatcher.py

cd ./../../..

roslaunch sofar_multimodal matcher.launch


### Prerequisites
On the machine must be installed RoS with rospy and python.

### Installation
```
cd catkin_ws/src/
catkin_create_pkg sofar_multimodal std_msgs rospy roscpp
catkin_make
 . ~/catkin_ws/devel/setup.bash
cd sofar_multimodal
git clone https://github.com/EmaroLab/mmodal_perception_fusion.git .
catkin_make
```
### How to run the project

```
cd sofar_multimodal
roscore &
source devel/setupe.bash
rosrun sofar_multimodal talkerPitt &
rosrun sofar_multimodal talkerTensor &
rosrun sofar_multimodal adapterPitt &
rosrun sofar_multimodal adapterTensor &
rosrun sofar_multimodal featureScheduler.py &
rosrun sofar_multimodal tableMatcher.py &
rosrun sofar_multimodal reasonerMain.py &
rosrun sofar_multimodal featuresMatcher.py &
```
To monitor the output:
```
rostpic echo /featureMatcher/dataPub
```

## Results

<p align="center"> 
<img src="https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/VirtualBox_EmaroBox_1_29_11_2019_16_32_49.png?raw=true">
</p>

### Video
[![Watch the video](https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/imgs/VirtualBox_EmaroBox_1_29_11_2019_16_32_49.png?raw=true)](https://www.youtube.com/embed/JwPSA9yZG2A)

## Recommendations
To add a perception module, you will need to add an Adapter module between perception module and feature selector module.
Then, add a subscriber and its callback in featureScheduler.py by following the [commented example](https://github.com/EmaroLab/mmodal_perception_fusion/blob/a56460ff915c84362d3897307453054a5fdfaa02/scripts/featureScheduler.py#L104-L105) into the script.

In [docs](https://github.com/EmaroLab/mmodal_perception_fusion/blob/master/docs/) directory there is the doxygen documentation in html or latex format.


## Authors
* Filippo Lapide

* Vittoriofranco Vagge

* Maicol Polvere
* Daniele Torrigino
* Francesco Giovinazzo

* Nicolò Baldassarre
* Andrea Rusconi
* Matteo Panzera

* Francesco Bruno
* Ariel Gjaci
