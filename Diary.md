26/09/2023
Supervisor Meeting

Inital supervisor meeting to discuss project and background information on structure of project.

Objectives set:
	-Need 3 datasets: Small, medium, and large, number of features (only small and large initially add medium later)
	-Implement 3 different machine learning algorithms: Ridge Regression, Decision trees (might change), Neural Networks (extendable)
	
By the first term must complete:
	-Explanation of algorithms and visualisation
	-Implementation of 2 algorithms
	-Run algorithms on 2 datasets
	-Analyse and compare their performance
	

06/10/2023
FYP Plan Submitted

Finished my initial plan for my FYP and submitted it for grading. It was my first time really writing an abstract for a report, so it was very unfamiliar for me but in the end I managed to write something that I'm happy with. I wasn't sure how in-depth I should have gone into the maths behind the algorithm, but I managed to learn a lot early about how it functions which will make my life easier in the future as I already have a better understanding. 

Also created a timeline explaining my goals for each week so that I'm on track to completing all of my final deliverables. I think that I might have given myself more time than necessary for the tasks I set but in general it seems accurate and possible to follow.

The risks and mitigations sections was straightforward to write and I'll keep thinking about any other possible risks that might exist so that I can add them to the list and work on preventing them.


19/10/2023
Supervisor Meeting

Second Supervisor Meeting to asks specific questions on project after having a better understanding from writing the plan.

Questions:
	-When should I start writing up the actual report?
		Answer: As soon as possible, basically best to do it alongside implementation so that most of the work is done when it gets close to the deadline.
	
	-How much explaining of concepts should be done, i.e., how much information should I assume the reader knows?
		Answer: Can start from very basics (supervised vs unsupervised) but keep it extremely brief at the start of the report. Then go in depth on the theoretical explanations of algorithms and preprocessing. Also explain visualisation techniques but briefly as this is not the focus of the report.
		
	-What structure should I follow for write up?
		Answer: Intro, Theoretical Background (Include here algorithm/preprocessing explanations), Practical (Datasets and visualisation).
		
Extras:
	-Take a look at polynomial regression as a possible technique to use.
	-UCI to find other datasets.
	

04/12/2023
Report Progress (Abstract, Introduction, Simple & Multiple Linear Regression)

Finished draft of abstract and introduction chapters of report. Included the importance of the project as a whole and what aims I strive to achieve by the end of it. Also defined what dataset I will be using and what algorithms I will implement to base the models on, as well as what performance measures I will use to analyse and compare their performances.

Have also started the theoretical background chapter of the report by laying out a plan of the sections that it will include and filling them in one by one. I have completed the draft for the general introduction into what is regression and linear regression in the context of machine learning, and I am currently working on mathematical definitions of the algorithms and of the cost functions used to optimize the parameters.


04/12/2023
Report Progress (Simple & Multiple Linear Regression, Ridge Regression)

Completed both the Simple Linear Regression and Multiple Linear Regression subsections in the theory section. I have added theoretical explanations as well as formal mathematical definition with equations. Will start work on new chapter of Ridge Regression.


05/12/2023
Report Progress (Ridge Regression)

I've done a large amount of research on both multicollinearity and Ridge Regression, have also started the Ridge Regression chapter of the report. For the Ridge Regression chapter, I have decided to break it down into two parts: the first being the issue of multicollinearity and then how Ridge Regression proposes to fix them. I have finished writing the part on multicollinearity and how this affects linear regression models and have gotten started on writing how Ridge Regression solves the issues created.


05/12/2023
Report Progress (Ridge Regression)

Finished all the draft sections related to Ridge Regression in the report. It is still possible that I come back later and talk about how to mathematically minimize the cost functions if I have the time. Now I will go back to writing the code for the actual program to work, I need to finish implementing the Ridge Regression class and also import the datasets.


27/12/2023 (past entry)
Code Progress 

Completed implementation of Ridge Regression into python program using the matrix form of the algorithm when fitting the model to get optimal coefficients. Created corresponding TDD test for the Ridge Regression class and did basic refactoring after implementation was finished. In the future I plan on returning to  this class and creating further tests in order to implement proper input sanitisation and exception handling. Next, I plan on writing a proof-of-concept program that runs on a smaller dataset to test its effectiveness and see whether it works or not.


27/12/2023 (past entry)
Proof of Concept Progress

I have created a simple proof of concept (POC) program that demonstrates Ridge Regression working on a simple dataset. I have written the code that fits the training set and predicts the test set. I also added a matplotlib graph in my code that plots the different features against each other and shows the difference between the real test set samples and the predicted test set values. New sections have also been added to the report describing the general importance of a POC program, how it was implemented using code, the visualisation of the data and results, and a reflection on the outcome.


05/01/2024
Interim Report Git Release-1

Have completed first version of my FYP. It includes my Interim Report along with an implementation of the Ridge Regression algorithm form scratch. The code also runs on a smaller dataset as a proof of concept to show algorithm in action and to serve as point where I can branch off in order to expand the project. An official release has been made on git where the candidate release branch has been created, worked on, and tagged with a version number. Also a CHANGELOD.md file was created and updated.


10/02/2024
Report Progress (Ridge Estimator)

Have added subsection to final report explaining how I derived the Ridge Regression estimator that I have used in my program to find the optimal model coefficients. In this section I show the mathematical proof of how I got to my final result, and I also explain what the result means and its significance. I also had to go back into some other sections to introduce the concept of matrix notation, as this was used extensively when finding the estimator.


02/03/2024
Programming Progress (GUI and score)

I have made many advancements this last week in regard to the programming part of my project. I have constructed a GUI for my project where a user can interact with both the proof-of-concept program and the ridge regression model trained on the Boston housing problem dataset. These programs can take input from the user of what alpha values they want to use and using it when predicting their values on the test set. The results information is displayed as a graph where the predicted data is plotted against the true and predicted values.


09/03/2024
Supervisor Meeting & Report Progress

I had a very useful meeting with my supervisor recently where I got a clear idea of what my next steps should be in order to complete my report. Until now I have managed to program and explain theoretical most of the tools that I will need to gather my data. Before I was not sure what steps I should take next, but now I created a plan for what my next chapters, Method and Experiment, should include. I have defined the sections that I will need to discuss within each chapter and now I will get started on filling out this structure with words and explanations.