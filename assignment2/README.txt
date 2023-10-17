This is the second assignment of the 6105 course. The main content is to analyze data through machine learning algorithms. It mainly includes two analyses: one is the prediction of the success rate of customers purchasing deposit products through telemarketing, and the other is the credit classification of bank customers. The prediction results of marketing are dichotomous, yes or no; the customer's credit is classified as good, standard, or poor. There are a lot of missing data in customer credit classification, which were filled in before analysis. Before modeling and analysis, I standardized the data. The algorithm models used mainly include decision trees, neural networks, Boost and logistic regression.

File Tree:
Qiaotong_Huang_002728446
│  
└─assignment2
    │  Credit_Score_Classification.ipynb
    │  Bank_Marketing_Prediction_Classification.ipynb
    │  README.txt
    │  Qiaotong_Huang_002728446_analysis.pdf
    │  
    ├─Bank_Marketing_Dataset
    │      bank_full.csv
    │      bank.csv
    │      introduce.txt
    │      
    └─Credit_Score_Dataset
            test.csv
            train.csv
            introduce.txt


Compile Environment: 

	Mac System

	VSCode 

	Python Version: 3.10.5 64bit

	Python Library Needed:

		numpy  		1.23.2
		pandas		1.4.4
		scikit learn     1.1.2
		matplotlib	3.5.3
		seaborn		0.12.1