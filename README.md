## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In your Cloud IDE, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with *Regenerate API Key*.

# Heritage Housing Issues

**Data Analysis and Predictive Modelling Study**

**Developed by: Emmanuel Quaynor**

![I am responsive image](#)

**Live Site:** [Live webpage](#)

**Link to Repository:** [Repository](https://github.com/equaynor/heritage-housing-issues)


## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Content](#2-dataset-content)
3. [Business Requirements](#3-business-requirements)
    - [CRISP-DM Workflow](#crisp-dm-workflow)
    - [User Stories](#user-stories)
4. [Hypotheses and Validation](#4-hypotheses-and-validation) 
5. [Rationale to map the business requirements to the Data Visualizations and ML tasks](#5-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
6. [ML Business Case](#6-ml-business-case)
7. [Dashboard Design](#7-dashboard-design)
    - [Page 1: Project Summary](#page-1-project-summary)
    - [Page 2: Sale Price Correlation Analysis](#page-2-sale-price-correlation-analysis)
    - [Page 3: Sale Price Prediction](#page-3-sale-price-prediction)
    - [Page 4: Hypothesis and Validation](#page-4-hypothesis-and-validation)
    - [Page 5: Machine Learning Model](#page-5-machine-learning-model)
8. [Unfixed Bugs](#8-unfixed-bugs)
9. [PEP8 Compliance Testing](#9-pep8-compliance-testing)
10. [Deployment](#10-deployment)
11. [Technologies](#11-technologies)
    - [Development and Deployment](#development-and-deployment)
    - [Main Data Analysis and Machine Learning](#main-data-analysis-and-machine-learning)
12. [Credits](#12-credits)
    - [Sources of code](#sources-of-code)
    - [Media](#media)
13. [Acknowledgements](#13-acknowledgements)

## **1. Introduction**

Welcome to the Heritage Housing Issues project, a data science initiative aimed at predicting the sale price of houses in Ames, Iowa. The goal of this project is to develop a machine learning model that can accurately estimate the sale price of houses based on various attributes, including features such as size, quality, and condition.

This Machine Learning Project was developed as the fifth portfolio project during the Code Insititute's Diploma in Full Stack Development. It covers the Predictive Analytics specialization.

## **2. Dataset Content**

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

### **Project Terms and Jargon**

* **Sale price** of a house refers to the current market price of a house with certain attributes.

* **Inherited house** is a house that the client inherited from grandparents.

* **Summed price** is the total of the sale prices of the four inherited houses.

## **3. Business Requirements**

Our client, who has inherited four houses in Ames, Iowa, has requested our assistance in maximizing the sales price for these properties. Despite her excellent understanding of property prices in her own state and residential area, she is concerned that relying on her current knowledge may lead to inaccurate appraisals, as the factors that make a house desirable and valuable in her area may differ from those in Ames, Iowa.

Our client has provided us with a public dataset containing house prices for Ames, Iowa. We have agreed to support her in addressing the following business requirements:

* **BR1:** The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualizations of the correlated variables against the sale price to show that.

* **BR2:** The client is interested in predicting the house sales price from her 4 inherited houses, and any other house in Ames, Iowa.

To address these business requirements, we will break down the project into manageable epics and user stories. Each user story will be further divided into tasks, which will be implemented using an agile process. This approach will enable us to deliver a high-quality solution that meets our client's needs and expectations.

### CRISP-DM Workflow

#### **Epic 1: Business Understanding (CRISP-DM Phase 1)**

---

- **Define the Problem**: Clearly articulate the problem statement and identify the client's (Lydia's) goals and objectives.
- **Create User Stories**: Break down the project requirements into smaller, manageable tasks.
- **Create hypotheses and validations**: Define hypotheses and validate them.

#### **Epic 2: Data Understanding (CRISP-DM Phase 2)**

---

- **Collect and Load the Dataset**: Load the dataset and explore its structure, content, and summary statistics.
- **Explore the Data**: Use visualization and summary statistics to understand the distribution of variables and their relationships.
- **Identify Missing Values and Handle Them**: Detect and handle missing values in the dataset.
- **Document the Data**: Document the data, including its sources, quality, and limitations.

#### **Epic 3: Data Preparation (CRISP-DM Phase 3)**

---

- **Clean the Data**: Clean the dataset by handling outliers, encoding categorical variables, and scaling/normalizing numerical variables.
- **Feature Engineering**: Create new features that might be relevant for the model (e.g., feature interactions, transformations).
- **Split the Data**: Split the dataset into training, validation, and testing sets.
- **Document the Data Preparation**: Document the data preparation steps, including data cleaning, feature engineering, and data splitting.

#### **Epic 4: Modeling (CRISP-DM Phase 4)**

---

- **Choose a Model**: Select a suitable machine learning algorithm for the problem (e.g., linear regression, decision trees, random forest).
- **Train the Model**: Train the model on the training data.
- **Evaluate the Model**: Evaluate the model's performance on the validation data using metrics such as mean squared error, R-squared, etc.
- **Hyperparameter Tuning**: Perform hyperparameter tuning to optimize the model's performance.

#### **Epic 5: Evaluation (CRISP-DM Phase 5)**

---

- **Evaluate the Model's Performance**: Evaluate the model's performance on the testing data.
- **Compare Models**: Compare the performance of different models and select the best one.
- **Refine the Model**: Refine the model by incorporating additional features, handling outliers, or using different algorithms.

#### **Epic 6: Deployment (CRISP-DM Phase 6)**

---

- **Create a Dashboard**: Create a dashboard to visualize the model's predictions and insights.
- **Deploy the Model**: Deploy the model in a production-ready environment (e.g., Flask, Django, Streamlit).
- **Monitor and Update the Model**: Monitor the model's performance and update it as necessary to maintain its accuracy.

These steps can be matched up nicely to 6 Epics in the Agile development process. As we move along the pipeline of the development process we may flow back and forth between stages/epics as we learn new insight and have to revisit previous step in order to refine the development. While ultimately moving towards the final delivery of a product that satisfies the users/clients requirements.

### User Stories

**US1:** As a client, I want to discover which attributes of a house are most correlated with its sale price, so that I can understand the key drivers of sale price. (Business Requirement Covered: BR1)

**US2:** As a client, I want to have a reliable prediction of the sale price of my inherited houses, so that I can sell them at the maximum total price possible. (Business Requirement Covered: BR2)

**US3:** As a technical user, I want to understand the machine learning model used to predict the sale price, so that I can trust the accuracy of the predictions. (Business Requirement Covered: BR2)

**US4:** As a client, I want to visualize the relationships between sale price and other features, so that I can gain insights into the importance of different attributes. (Business Requirement Covered: BR1)

**US5:** As a client, I want to have a user-friendly dashboard to input house data and receive predicted sale prices, so that I can easily make informed decisions about pricing and selling. (Business Requirement Covered: BR2)

## **4. Hypotheses and Validation**

We propose the following hypotheses to explain the relationship between house attributes and sale price:

1.  **Size Hypothesis:** Larger properties tend to have higher sale prices. We will investigate correlations between attributes related to house size (e.g. square footage, number of bedrooms) and sale price to validate this hypothesis.
2. **Quality Hypothesis:** Higher quality ratings are associated with higher sale prices. We will examine correlations between variables related to house quality (e.g. kitchen quality, overall quality) and sale price to test this hypothesis.
3. **Time Hypothesis:** The age of the property and recent renovations can significantly impact its value. We will study the relationship between the year the house was built, recent remodels, and sale price to validate this hypothesis.

## **5. Rationale to map the business requirements to the Data Visualisations and ML tasks**

- **Business Requirement 1 (BR1):** Data Visualization and Correlation Study
    
    - We will analyze the distribution of sale prices in the dataset to gain insights into its characteristics.
    - We will investigate the relationships between various attributes and sale prices using Pearson and Spearman correlation analysis.
    - We will visualize the key variables to understand their impact on sale prices.
    - The [Correlation Study notebook](#) addresses this business requirement.

- **Business Requirement 2 (BR2):** Regression Analysis
    
    - Since the target variable is continuous, we will employ regression analysis to predict sale prices. If the model performance is poor, we may consider alternative approaches.
    - We aim to identify the most influential attributes driving sale prices, enabling our customer to optimize pricing strategies. We may use Principal Component Analysis (PCA) to identify these variables.
    - The [Modeling and Evaluation notebook](#) addresses this business requirement.


## **6. ML Business Case**

### **Predict Sale Price**
#### **Regression Model**

We want an ML model to predict the sale price of houses in Ames, Iowa. The target variable is a continuous number, and we will use a regression model to achieve this goal.

**Ideal Outcome**

Our ideal outcome is to provide our client with a reliable tool to predict the sale price of her inherited houses, as well as any other house with similar attributes. This will enable her to make informed decisions about pricing and selling her properties.

**Model Success Metrics**

The model success metrics are:

- At least 0.75 for R2 score, on both train and test sets
- The model is considered a failure if its predictions are off by more than 25% of the time

**Output**

The output is defined as a continuous value representing the sale price in USD.

**Heuristics**

Our Belgian client, unfamiliar with local property prices, feared her limited knowledge might lead to inaccurate appraisals. She sought our help to maximize the sale price. We employ Machine Learning models and regression algorithms, rather than unreliable heuristics, to ensure accurate property valuation.

**Training Data**

The training data to fit the model comes from a public dataset from Ames, Iowa, which contains approximately 1.5 thousand property sales records and 22 features. We will preprocess the data by dropping variables with more than 75% missing values and selecting the remaining variables as features.

  * Train data: drop variables 'EnclosedPorch' and 'WooddeckSF' because each has more than 75% missing values. 

  * Target variable: SalePrice 

  * Features: all remaining variables.

**Pipeline Steps**

Our pipeline will consist of the following steps:

- Data preprocessing
- Feature selection
- Model training and testing
- Model deployment and maintenance

**Application**

Our model will be useful for our client, who wants to predict the sale price of her inherited houses, as well as for other users who want to estimate the sale price of their own properties. The model can be accessed online, and users can input data for their homes to get a predicted sale price.


<details>
<summary> Pipeline steps</summary>
<img src="docs/screenshots/pipeline steps.png">
</details>

## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

