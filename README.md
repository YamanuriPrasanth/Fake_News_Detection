# ABSTRACT

### FAKE NEWS DETECTION

>The advent of the World Wide Web and the rapid adoption of social media platforms (such as Facebook and Twitter) paved the way for information dissemination that has never been witnessed in the human  history  before  . With the current usage of social media  platforms  ,  consumers are creating and sharing  more information  than ever before , some  of which are misleading with no relevance to  reality. Automated classification of a text article as misinformation or disinformation is a challenging task.<br><br> 
>Even an expert in a particular domain has to explore multiple ascepts before giving ascepts be- fore giving a verdict on the truthfulness of an article. In this work, we purpose to use machine learning en semble approach for automated classification of news article. Our study explores different textual propert ies that can be used to distinguish fake contents fom real. <br><br>
>By using those properties,we traina combination of different machine learning algorithms using Various ensemble methods and evaluate their performance on 4 real world data sets. Experimental evalu- ation confirms the superior performance of our proposed ensemble learner approach in comparison to individual learners. 

                                                                                                     

### 1-Introduction To Project

>The advent of the World Wide Web and the rapid adoption of social media platforms (such as Facebook and Twitter) paved the way for information dissemination that has never been witnessed in the human history before. Besides other use cases, news outlets benefitted from the widespread use of social media platforms by providing updated news in near real time to its subscribers. The news media evolved from newspapers, tabloids, and magazines to a digital form such as online news platforms, blogs, social media feeds, and other digital media formats . It became easier for consumers to acquire the latest news at their fingertips. Facebook referrals account for 70% of traffic to news websites . These social media platforms in their current state are extremely powerful and useful for their ability to allow users to discuss and share ideas and debate over issues such as democracy, education, and health. However, such platforms are also used with a negative perspective by certain entities commonly for monetary gain and in other cases for creating biased opinions, manipulating mindsets, and spreading satire or absurdity. The phenomenon is commonly known as fake news.<br><br> 
>There has been a rapid increase in the spread of fake news in the last decade, most prominently observed in the 2016 US elections . Such proliferation of sharing articles online that do not conform to facts has led to many problems not just limited to politics but covering various other domains such as sports, health, and also science . One such area affected by fake news is the financial markets , where a rumor can have disastrous consequences and may bring the market to a halt.<br><br>  
>Our ability to take a decision relies mostly on the type of information we consume; our world view  is shaped on the basis of information we digest. There is increasing evidence that consumers have reacted absurdly to news that later proved to be fake . One recent case is the spread of novel corona virus, where fake reports spread over the Internet about the origin, nature, and behavior of the virus . The situation worsened as more people read about the fake contents online. Identifying such news online is a daunting task.<br><br>  
>Fortunately, there are a number of computational techniques that can be used to mark certain articles as fake on  the  basis  of  their  textual  content.  Majority  of  these  techniques  use  fact  checking  websites  such  as “PolitiFact” and “Snopes.” There are a number of repositories maintained by researchers that contain lists of websites that are identified as ambiguous and  fake . However, the problem with these resources is that human expertise is required to identify articles/websites as fake. More importantly, the fact checking websites contain articles from particular domains such as politics and are not generalized to identify fake news articles from multiple domains such as entertainment, sports, and technology.<br><br>  
>The World Wide Web contains data in diverse formats such as documents, videos, and audios. News published online in an unstructured format (such as news, articles, videos, and audios) is relatively difficult to detect and classify as this strictly requires human expertise. However, computational techniques such as natural language processing (NLP) can be used to detect anomalies that separate a text article that is deceptive in nature from articles that are based on facts . Other techniques involve the analysis of propagation of fake news in contrast with real news . More specifically, the approach analyzes how a fake news article propagates differently on a network relative to a true article. The response that an article gets can be differentiated at a theoretical level to classify the article as real or fake.<br><br>  



### 2-Existing System

>Using Bag of words,stop words techniques in Natural language processing, and using Machine learning naïve bayes algorithm to get most fake words and most real words .Fake words are having higher negative coefficient it means any sentence or text contain that particular word may have higher chances of being faked... 

### 3-Proposed System

>**News Authenticator** <br>
New authenticator  follows some  steps to check whether the news is true or false. It will compare news which  is  given by our side with different websites and various news sources if that news is found on any news website then it shows the given news is true, else it shows there has been no such news in last few days. This can help us from fake news.  These  days‟  fake  news  spread very fast because of social  media  and  the internet.  So,  news  authenticator  helps us to detect either the given news is fake or real. <br><br>
>**News Suggestion / Recommendation System** <br>
News suggestion suggests recent news and suggests the news related to the news which the user has given for authentication . If the news is fake, then this news suggestion gives the related news on that topic. The news suggestion suggests the news based on keywords which you give in your news based on keywords which you give in your news based on keywords which you give in your news which you wish to authenticate 

### 4-Objective

>The main objective is to detect the fake news, which is a classic text classification problem with a straight forward proposition. It is needed to build a model that can differentiate between “Real” news and “Fake” news. This leads to consequences in social networking sites like content. Secondly, the  trolls  are  real humans who “aim to disrupt online communities” in hopes of provoking social  media users  into an emotional  response.  Other  one  is,  Cyborg. Cyborg users are the combination of “automated activities  with human input.”Humans build accounts and use programs to perform activities in social media. For false information detection, there are two categories: Linguistic Cue and Network Analysis approaches. The methods generally used to do such  type  of  works  are Naïve  Bayes  Classifier  and  Support  Vector Machines (SVM).<br><br> 


### 5-Introduction to Natural Language Processing

So the first question that comes in our mind is **What is NLP** ? Why is it so important and so much famous these days. 

To understand it’s importance, let’s look at some promising examples :- 

So have you ever wondered when we are using famous messaging apps like Whats App, Messenger, Hike etc, they suggest meaningful words before letting you complete the sentence. Another example would be like the SPAM or junk folder in your email, Chat Bots, Google Translation and so much more ! 

So yeah, these were some cool examples of NLP or Natural Language Processing. 

So the term Natural Language Processing can be defined as field concerned with the ability of a computer to understand, analyze, manipulate and potentially generate human language (or close to human language). 

It can be any language English, Hindi, French, Spanish etc. 

**Real Life Examples** :- 

- Auto – Complete 
- Auto – Correct 
- Spam Detection 
- Translation of one Language to Another 
- Conversational Chat Bots 

**Areas of NLP :-** 

- **Sentiment Analysis** :- 

*Sentiment Analysis* is a natural language processing technique used to determine whether data is positive, negative or neutral. Sentiment analysis is often performed on textual data to help businesses monitor brand and product sentiment in customer feedback, and understand customer needs. 

For example :- Movie reviews sentiment analysis, tweets analysis etc. 

- **Topic Modeling** :- 

A *topic model* is one that automatically discovers topics occurring in a collection of documents. A trained model may then be used to discern which of these topics occur in new documents. The model can also pick out which portions of a document cover which topics. 

- **Text Classification** :- 

Text clarification is the process of categorizing the text into a group of words. By using NLP, text classification can automatically analyze text and then assign a set of predefined tags or categories based on its context. 

Fake News Detection ![](Aspose.Words.e79f2849-eb9b-43d2-b2fa-5c70c308d5dd.006.png)

### 6-NLP ToolKit – NLTK

Now that we have some clue about what’s going on and what is Natural Language Processing, we will continue the NLP WITH PYTHON 

NLTK i.e. Natural Language Processing Tool Kit is a suite of open-source tools created to make NLP processes in Python easier to build. 

In the above lesson we have seen that how NLP has revolutionized many areas of language such as sentiment analysis, part-of-speech tagging, text classification, language translation, topic modeling, language generation and many many more. So there are many in-built functions and libraries that are included inside this NLTK library. 

!pip install nltk 

import nltk 

nltk.download() 

- this will allow you to download all the necessary tools present in the library 

This library let us do all the necessary preprocessing on our text data without any pain :D, some of the components of this library are :- **stemming**, **lematizing***,* **tokenizing***, stop-words removal* and so many more… 



Fake News Detection ![](Aspose.Words.e79f2849-eb9b-43d2-b2fa-5c70c308d5dd.007.png)

**NLP PipeLine** 

Whenever we work on *raw text* data, python does not understand words, it just sees a stream of characters and for it, all the characters are same having no meaning. Any Machine Learning algorithm/programming language only understands **Numbers**/**Vectors** and not words so in order to make it understand, we need to perform the above NLP pipeline as shown in the image above. 







### 7-Tokenization

>Tokenization is simply splitting the text/corpus into words or sentences. from nltk.tokenize import word\_tokenize <br>
>sample\_text = 'Hi my name is Yash' tokenized\_text = word\_tokenize(sample\_text)<br> 
>print(tokenized\_text) <br>
>output : [‘Hi’, ‘my’, ‘name’, ‘is’, ‘Yash’] <br>



### 8-Text Cleaning

>Text cleaning basically refers to the functions applied to the raw text **in** order to remove unnecessary words, punctuation, extra white spaces, **and** giving the text more meaning **in** order to be processed by our ML algorithm. <br><br>
>There are various Text Pre-processing/Cleaning techniques like : <br>
>- Punctuation Removal 
>- Stop Words Removal 
>- Extra white space Removal 
>- Emoji Removal 
>- Emoticons Removal 
>- HTML tags Removal 
>- URLs Removal 
>- Conversion to Lower case 
>- Numbers Removal 
?- Expanding Contractions **and** so many more !<br><br> 
>**Stemming** <br>
>**Stemming** is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma.<br><br> 
>**Lemmatizing** <br>
>**Lemmatization** takes into consideration the morphological analysis of the words. To do so, it is necessary to have detailed dictionaries which the algorithm can look through to link the form back to its lemma.<br><br> 
>Basic difference between stemming and lemmatizing* is that in **stemming**, the removal of suffix takes place without any meaning. On the other hand, **lemmatizing** takes morphological and lexical meaning into consideration and then returns a much more meaning full ‘lemma’. <br><br><br>








### 9-Vectorization 

>As we discussed earlier that in order to make our machine learning algorithm make sense of text data, we need to convert the characters/words/sentences into numbers/vectors.<br><br> 
>So what exactly is **Vectorization** ?<br><br> 
**Vectorization** is the process of conversion of words into numbers/vectors.<br><br> 
>sample\_text = 'My name is Yash' vectorized\_text = [123, 32, 15, 107]<br><br> 
>**Types of Vectorization methods :-**<br> 
>- Count Vectorization 
>- N-Gram Vectorization 
>- TF-IDF Vectorization 
>- Word2Vec 
>- BERT (It is basically Word2Vec with Context) (state-of-the-art, Advance topic)<br><br> 
>In this Project we use Count Vectorization and N-Gram Vectorization.Let us discuss about this two Techniques<br><br> 
>**Count Vectorization Technique :-**<br> 
>It is the simplest of all vectorization techniques.<br><br> 
>- It creates a **Document Term Matrix**, now a doc-term matrix is a matrix whose **rows** are every single element our list, for example, we have **4** elements/docs <br> ‘sample\_text’, **columns** are all the unique words from our whole document/whole ‘sample\_text’. Each cell of our Doc-Term matrix represents the **frequency** of that word in current cell.<br> 
>- Now let’s code what we discussed above ! (It’s really simple!). <br><br>
>**N-gram Vectorization Technique :-** <br>
>- It also creates a Document Term matrix.<br> 
>- Columns represent all columns of adjacent words of length ‘n’. <br>
>- Rows represent each document in our sample\_text.<br>
>- Cell represent Count.<br>
>- When n = 1, it is called Uni-gram, which is basically Count Vectorizer. Example = “my”, “name”, “is”, “yash”. When n = 2, it is called Bi-gram, Example = “my name”, “is yash”.<br>
>- When n = 3, it is called Tri-gram, Example = “my name is”, “yash”. 



### 10-Machine Learning

>After we have successfully converted our raw text data into vectors, we are now ready to feed this vectorized data and it’s corresponding label like Fake or Real in to our machine learning algorithm.<br><br> 
>There are many Machine Learning algorithms out there like Decision Trees, Random Forest Classifier, KNN, Logistic Regression etc, but the algorithms which works better on textual data are **Naive Bayes Algorithm** which works on the concept of Conditional Probability<br> 
>After training with any of the mentioned Machine Learning/Deep Learning algorithms, We can now say that we have successfully trained our Fake News Classifier model which can detect whether a message/email is Fake or not!<br> 
>In This project I will be using n-gram Vectorization technique because it gave me around me around <br><br> 
>Now the interesting part – **Model Building** <br> 
>I will be using **Naive Bayes Algorithm** as it considered to be good when dealing with text data.<br>



### 11-Libraries

>These are the libraries used in this Project<br><br>
>https://numpy.org/doc/stable/ <br>
>https://pandas.pydata.org/docs/<br>
>https://matplotlib.org/stable/users/index.html<br>
>https://seaborn.pydata.org/<br>
>https://docs.scipy.org/doc/scipy/<br>
>https://scikit-learn.org/stable/<br>
>https://www.nltk.org/<br>


### 12-Naive Bayes Algorithm 

>It  is  a classification technique Based  on  Bayes’  Theorem  with  an  assumption  of  independence  among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.<br><br> 
>For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, all of these properties independently contribute to the probability that this fruit is an apple and that is why it is known as ‘Naive’.<br>  
>Naive Bayes model is easy to build and particularly useful for very large data sets. Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.<br>  
>Bayes theorem provides a way of calculating posterior probability P(c|x) from P(c), P(x) and P(x|c). Look at the equation below:<br>
>Above,<br> 
>- *P*(*c|x*) is the posterior probability of *class* (c, *target*) given *predictor* (x, *attributes*). 
>- *P*(*c*) is the prior probability of *class*. 
>- *P*(*x|c*) is the likelihood which is the probability of *predictor* given *class*. 
>- *P*(*x*) is the prior probability of *predictor*. 

### 13-Passive Aggressive Algorithm

>The Passive-Aggressive algorithms are a family of Machine learning algorithms that are not very well known by beginners and even intermediate Machine Learning enthusiasts. However, they can be very useful and efficient for certain applications.<br><br> 
>**Note:** This is a high-level overview of the algorithm explaining how it works and when to use it. It does not go deep into the mathematics of how it works.<br><br> 
>Passive-Aggressive algorithms are generally used for large-scale learning. It is one of the few ‘**online- learning algorithms**‘. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. We can simply say that an online-learning algorithm will get a training example, update the classifier, and then throw away the example.<br><br> 
>A very good example of this would be to detect fake news on a social media website like Twitter, where new data is being added every second. To dynamically read data from Twitter continuously, the data would be huge, and using an online-learning algorithm would be ideal.<br><br> 
>Passive-Aggressive algorithms are somewhat similar to a Perceptron model, in the sense that they do not require a learning rate. However, they do include a regularization parameter.<br><br> 
>**How Passive-Aggressive Algorithms Work:** Passive-Aggressive algorithms are called so because :<br><br> 
>**Passive:** If the prediction is correct, keep the model and do not make any changes. i.e., the data in the example is not enough to cause any changes in the model.<br> 
>**Aggressive:** If the prediction is incorrect, make changes to the model. i.e., some change to the model may correct it.<br>


### 14-Data Flow Diagram

![](https://static-01.hindawi.com/articles/complexity/volume-2020/8885861/figures/8885861.fig.001.svgz)

>**Dataset collection**<br>
>In This Project we collect data set From Kaggle website <br><br>
>**Data Cleaning**<br> 
>**Data cleaning** is the process of preparing raw **data** for analysis by removing bad **data**, organizing the raw **data**, and filling in the null values. Ultimately,**cleaning data** prepares the **data** for the process of **Machine learning** when the most valuable information can be pulled from the **data** set. <br><br>
>**Exploration**<br> 
>Data exploration definition: <br>
>Data exploration refers to the initial step in data analysis in which data we use data visualization and statistical techniques to describe dataset characterizations, such as size, quantity, and accuracy, in order to better understand the nature of the data. <br><br>
>Data exploration techniques include both manual analysis and automated data exploration software solutions that visually explore and identify relationships between different data variables, the structure of the dataset, the presence of outliers, and the distribution of data values in order to reveal patterns and points of interest, enabling data analysts to gain greater insight into the raw data.<br><br> 
>After Exploration We extract important features this is called feature selection and split the data set in to training data and testing data <br><br>
>**Model Training**<br> 
>A machine learning training model is a process in which a machine learning (ML) algorithm is fed with sufficient training data to learn from. <br><br>
>**Model Evaluation**<br> 
>**Model evaluation** aims to estimate the generalization accuracy of a **model** on future (unseen/out-of-sample) data.<br><br>
>**Model Tuning**<br> 
>**Model tuning helps to increase the accuracy** of a machine learning **model**. 
Explanation: **Tuning** can be defined as the process of improvising the performance of the **model** without creating any hype or creating over fitting of a variance. 
After Model training our Fake news model detects correctly fake news and real news. <br><br><br>





### 15-Outputs 



### 16-Future Enhancement 
>In this Project We use Machine learning algorithms . In future Enhancement We use deep learning algorithms,Advanced NLP Techniques for better accuracy. 

### Conclusion

>Fake Words are having higher negative coefficient it means any sentence or text contain that particular word may have higher chances of being faked….<br><br> 
>The task of classifying news manually requires in-depth knowledge of the domain and expertise to identify anomalies in the text. In this research, we discussed the problem of classifying fake news articles using machine learning models and ensemble techniques. The data we used in our work is collected from the World Wide Web and contains news articles from various domains to cover most of the news rather than specifically classifying political news. The primary aim of the research is to identify patterns in text that differentiate fake articles from true news. We extracted different textual features from the articles using an LIWC tool and used the feature set as an input to the models. The learning models were trained and parameter-tuned to obtain optimal accuracy. Some models have achieved comparatively higher accuracy than others. We used multiple performance metrics to compare the results for each algorithm. The ensemble learners have shown an overall better score on all performance metrics as compared to the individual learners.<br><br> 
>Fake news detection has many open issues that require attention of researchers. For instance, in order to reduce the spread of fake news, identifying key elements involved in the spread of news is an important step. Graph theory and machine learning techniques can be employed to identify the key sources involved in spread of fake news. Likewise, real time fake news identification in videos can be another possible future direction. 

<br><br><br>
![](https://socialify.git.ci/YamanuriPrasanth/Fake_News_Detection/image?description=1&forks=1&issues=1&language=1&logo=https%3A%2F%2Fencrypted-tbn0.gstatic.com%2Fimages%3Fq%3Dtbn%3AANd9GcTjyHA0z64Zo0Ovz0X6MSIoyjqPYkUVHYFgoQ%26usqp%3DCAU&name=1&owner=1&pulls=1&stargazers=1&theme=Light)
