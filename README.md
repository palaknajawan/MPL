Module 2
 Information Retrieval Models
Formal Characteristics of IR Models
 ● Help to define their behavior, performance, and applicability to various retrieval tasks. 
● These characteristics can be used to compare and evaluate different IR models.
 1. Representation
 ● Document Representation: How documents are represented within the model (e.g., as vectors, 
probabilistic distributions, or networks).
 ● Query Representation: How user queries are represented and processed within the model.
 ● Feature Space: The set of features or dimensions used to represent documents and queries 
(e.g., terms, phrases, concepts).
 2. Matching Function
 ● Relevance Calculation: The method used to calculate the relevance of a document to a query 
(e.g., cosine similarity, probability estimation).
 ● Scoring Mechanism: The algorithm used to assign scores to documents based on their 
relevance to the query.
 ● Normalization: Techniques used to normalize scores to make them comparable across different 
documents and queries.
3. Ranking
 ● Ranking Algorithm: The process used to order documents based on their relevance scores.
 ● Rank Aggregation: Methods for combining multiple relevance scores or rankings into a final 
ranking (e.g., in meta-search engines like Dogpile,Trivago etc.).
   4. Retrieval Model Basis
 ● Theoretical Foundation: The underlying theory or mathematical basis of the model (e.g., 
probability theory, vector space theory, neural networks).
 ● Assumptions: The assumptions made by the model about the data and the retrieval process 
(e.g., term independence in the Vector Space Model).
 5. Efficiency
 ● Computational Complexity: The computational resources required to process queries and 
retrieve documents (e.g., time complexity, space complexity).
 ● Indexing Requirements: The type and structure of indexes needed to support efficient 
retrieval.
6. Effectiveness
 ● Precision and Recall: Metrics used to evaluate the accuracy of the retrieval model in 
returning relevant documents.
 ● F-measure: A combined metric of precision and recall to assess the overall effectiveness of 
the model.
 ● Mean Average Precision (MAP): A measure of the precision of the top-ranked documents 
across multiple queries.
 ● Normalized Discounted Cumulative Gain (NDCG): A measure of ranking quality, 
considering the position of relevant documents in the result list.
 7. Adaptability
 ● Relevance Feedback: The ability of the model to incorporate user feedback to improve 
retrieval performance.
 ● Query Expansion: Techniques to enhance the original query with additional terms or 
concepts.
 ● Personalization: The ability to tailor retrieval results based on user preferences and 
behavior.
8. Robustness
 ● Noise Tolerance: The model's ability to handle noisy or irrelevant data.
 ● Scalability: The model's performance when handling increasing amounts of data or larger query 
loads.
 ● Generalization: The model's capability to perform well across different types of data and retrieval 
tasks.
 9. Interpretability
 ● Transparency: The ease with which users can understand how relevance scores are assigned 
and how results are generated.
 ● Explanation: The model's ability to provide explanations for why certain documents are retrieved 
or ranked higher.
 10. Flexibility
 ● Model Extensions: The ability to extend or modify the model to incorporate new data types, 
features, or retrieval tasks.
 ● Integration: The model's compatibility with other retrieval models or systems.
Formal Characterization of IR Models
Taxonomy of Information Retrieval Models
Taxonomy of Information Retrieval Models
 1. Classical Models
 ● Boolean Model: Represents documents and queries as sets of index words. Hence model 
is set theoretic.  
● Vector Space Model: Represents documents and queries as vectors in a multidimensional 
space. Hence model is Algebraic. 
● Probabilistic Model: The framework for modeling documents and query representation is 
based on probability theory. Hence model is probabilistic.
 2. Structured Models
 ● Extended Boolean Model: Enhances the Boolean model by incorporating partial matching 
and term weighting.
 ● Fuzzy Set Model: Uses fuzzy logic to handle the vagueness and ambiguity in information 
retrieval, allowing partial membership.
Taxonomy of Information Retrieval Models
 1. Classical Models
 ● Boolean Model: Represents documents and queries as sets of index words. Hence model 
is set theoretic.  Uses Boolean logic (AND, OR, NOT) to match documents with queries. 
Documents are either relevant or not, with no ranking.
 ● Vector Space Model: Represents documents and queries as vectors in a multidimensional 
space. Relevance is determined by the cosine similarity between document and query 
vectors.
 ● Probabilistic Model: Estimates the probability that a document is relevant to a query based 
on prior probability and document features.
 2. Structured Models
 ● Extended Boolean Model: Enhances the Boolean model by incorporating partial matching 
and term weighting.
 ● Fuzzy Set Model: Uses fuzzy logic to handle the vagueness and ambiguity in information 
retrieval, allowing partial membership.
Taxonomy of Information Retrieval Models
 3. Language Models
 ● Query Likelihood Model: Assumes a document is relevant if it has a high probability of 
generating the query.
 ● Document Likelihood Model: Assumes queries are generated from relevant documents 
and ranks documents based on the likelihood of generating the query.
 4. Bayesian Network Models
 ● Simple Bayesian Network: Uses Bayesian inference to model the relationships between 
documents, terms, and relevance.
 ● Inference Network Model: Constructs a network of nodes representing queries, terms, and 
documents, using probabilistic inference to determine relevance.
Taxonomy of Information Retrieval Models
 5. Latent Semantic Models
 ● Latent Semantic Analysis (LSA): Reduces dimensionality by identifying latent concepts in 
documents and queries through Singular Value Decomposition (SVD).
 ● Latent Dirichlet Allocation (LDA): Assumes documents are mixtures of topics, and each 
topic is a distribution over words, using probabilistic modeling to uncover these topics.
 6. Neural Network Models
 ● Deep Learning Models: Utilize neural networks, such as Convolutional Neural Networks 
(CNNs) and Recurrent Neural Networks (RNNs), to capture complex patterns and 
semantics in text.
 ● Transformer-Based Models: Use architectures like BERT and GPT to understand context 
and semantics through attention mechanisms.
Taxonomy of Information Retrieval Models
 7. Hybrid Models
 ● Combining Models: Integrate multiple retrieval models to leverage their strengths and 
mitigate weaknesses. For example, combining the vector space model with probabilistic 
elements.
 ● Meta-Search Engines: Aggregate results from multiple search engines, applying different 
IR models to improve overall relevance.
 8. Feedback and Adaptation Models
 ● Relevance Feedback: Modifies the original query based on user feedback, improving 
retrieval performance iteratively.
 ● Pseudo-Relevance Feedback: Automatically selects top-ranked documents from initial 
results to refine the query without explicit user feedback.
Taxonomy of Information Retrieval Models
 9. Time-Based Models
 ● Temporal IR: Incorporates the temporal aspects of documents and queries, focusing on the 
freshness and timeliness of information.
 10. Graph-Based Models
 ● PageRank: Uses link analysis to determine the importance of web pages, ranking them 
based on their connections.
 ● HITS Algorithm: Identifies hubs and authorities in a network of documents to rank them 
based on their link structure.
 Summary
 Each IR model has its own strengths and weaknesses, making them suitable for different types of 
information retrieval tasks and contexts. The choice of model depends on the specific requirements of the 
retrieval system, the nature of the documents and queries, and the desired balance between precision and recall.
Classic IR Models
 The three most well-known classic IR models are: 
1.
 2.
 3.
 Boolean Model, 
Vector Space Model, and 
Probabilistic Model. 
1. Boolean Model (Set Theoretic Model)
 Description:
 ● The Boolean Model is one of the simplest and earliest IR models. 
● It uses Boolean logic to represent documents and queries as sets of terms.
 ● It relies on Boolean logic to match documents with user queries using a binary approach where 
documents are either relevant or not relevant based on the presence or absence of terms.
 ● Queries are expressed using Boolean operators: AND, OR, and NOT.
 Key Concepts
 1. Boolean Logic:
 ○ AND: Retrieves documents that contain all the specified terms.
 ○ OR: Retrieves documents that contain at least one of the specified terms.
 ○ NOT: Excludes documents that contain the specified term.
 ○ Parentheses: Used to group terms and operators to form complex queries.
 2. Document Representation:
 ○ Documents are represented as sets of terms (keywords).
 ○ Each term in the document is treated as a binary feature (either present or absent).
 3. Query Representation:
 ○ Queries are formulated using Boolean expressions involving terms and Boolean operators (AND, 
OR, NOT).

Our Query: HUFFMAN CODE AND TREE BUT NOT DANGLING SUFFIX
                            10100           
AND     10100       
AND      10111
 Result: 10100 i.e  Video 1 and 3 ( Non Binary Huffman Codes And Adaptive Huffman Tree)

A More Realistic Scenario and Problem
 Problem1:
Problem 2: Term Document Matrix Is Sparse
Inverted Index- Basic Idea
Example: Shakespeare’s Play 
● Each document is identified by 
docID, a serial no for document
 We can apply Binary search for 
increasing searching speed)
How to maintain this postings list in the memory?
Building the inverted index : Steps
 Convert each token to lowercase , find base word for each token
Example: Inverted Index
 Figure  : Building an index by sorting and grouping. 
● The dictionary stores the terms, and has a pointer to the postings list for each term. 
● It commonly also stores other summary information such as, here, the document frequency 
of each term. 
● We use this  information for improving query time efficiency and, later, for weighting in ranked 
retrieval models.
 ● Each postings list stores the list of documents in which a term occurs, and may store other 
information such as the term frequency (the frequency of each term in each document) or the 
position(s) of the term in each document
Processing Boolean Queries
Example: Shakespeare’s Play 
● Each document is identified by 
docID, a serial no for document
 We can apply Binary search for 
increasing searching speed)
 2 31
 1 2 4 11 45 31 101 54 173 174
  3

Boolean Model Advantages and Disadvantages
 Advantages
 1. Simplicity:
 ○ Easy to understand and implement.
 ○ Queries are straightforward to construct using Boolean operators.
 2. Precise Matching:
 ○ Exact matching of query terms ensures that retrieved documents contain the specified terms.
 3. Efficiency:
 ○ Efficient retrieval using inverted indexes.
 Disadvantages
 1. Binary Relevance:
 ○ No ranking or relevance scoring of documents; results are binary (relevant or not).
 ○ Does not consider partial matches or degrees of relevance.
 2. Inflexibility:
 ○ Rigid query formulation may not capture the nuances of user information needs.
 ○ Complex queries can become cumbersome to construct and interpret.
 3. Lack of Term Weighting:
 ○ All terms are treated equally, without considering term frequency or importance.
 4. Handling Synonyms and Variants:
 ○ Cannot effectively handle synonyms, variants, or semantic relationships between terms.
Ranked Retrieval Models

Scoring as a basis of Ranked Retrieval
 ● We need to provide the searcher with the documents that are most likely to be 
useful to him.
 ● How can we rank the retrieved documents based on its relevance to user 
query ?
 ■ Assign the score [0-1] to each document.


● Here  D2 is more relevant to Q.
 ● But in real scenario, it is not true.(D1 is more relevant to Q and D2 has nothing to do with Q).
 This is drawback of JC. Sometimes we get answers that are not relevant to Q
Advantages:
 ● Simple and easy to compute.
 ● Works well for sets with distinct, non-repeating elements.
 ● Useful for applications where presence or absence of terms is more important than their frequency.
 Disadvantages:
 ● Does not take into account term frequency; only presence or absence matters.
 ● May not perform well with documents where the term frequency is important.
 ● Can be less effective for large and sparse sets where the overlap may be minimal.
 Use Cases
 Jaccard's coefficient is often used in:
 ● Document clustering: To measure the similarity between documents.
 ● Plagiarism detection: To find overlapping content.
 ● Recommendation systems: To compare user preferences or item features.
 ● Bioinformatics: To compare genetic sequences or protein structures.
 While Jaccard's coefficient is a useful similarity measure, it is often used in combination with other methods for more 
comprehensive ranking and retrieval tasks.
 Jaccard’s Coefficient : Advantages and Disadvantages
TF-IDF (Term Frequency-Inverse Document Frequency) 
● TF-IDF is a statistical measure used to evaluate the importance of a term in a document relative to a 
collection of documents (corpus). 
● It is widely used in information retrieval and text mining to rank and retrieve documents based on their 
relevance to a given query.
 ● Term Frequency (TF)
 ○ Measures how frequently a term t appears in a document (importance of a term within the doc)
 ○ A higher term frequency indicates a more significant term within that document.
 ○ Common formula: 
Where f(t,d) is the raw count of term t in document d, and the denominator is the total number of 
terms in document d.
 ●Inverse Document Frequency (IDF)
 ●Measures the importance of a term within the entire corpus.
 ●A term that appears in fewer documents is considered more significant.
 ●Common formula)
TF-IDF (Term Frequency-Inverse Document Frequency) 
●TF-IDF Score
 ❏ Combines TF and IDF to compute a weight for each term in each document.
 ❏ The weight reflects the importance of the term within the document and across the corpus.
 ❏ Formula: 
❏ Example: Consider a small corpus with three documents:
 ● Document 1: "apple banana apple"
 ● Document 2: "banana orange"
 ● Document 3: "apple orange orange banana"
 Calculate the TF-IDF for the term "apple" in Document 1.
TF-IDF : Advantages and Disadvantages
Example 2: 
● Document 1: "apple banana apple"
 ● Document 2: "banana orange"
 ● Document 3: "apple orange orange banana"
 ● User query: "apple banana"
 Step 1: For each term in each document, compute the term frequency (TF)
 Document 1:
 ● TF(apple, Document 1) = 2 / 3 = 0.67
 ● TF(banana, Document 1) = 1 / 3 = 0.33
 Document 2:
 ● TF(banana, Document 2) = 1 / 2 = 0.50
 ● TF(orange, Document 2) = 1 / 2 = 0.50
 Document 3:
 ● TF(apple, Document 3) = 1 / 4 = 0.25
 ● TF(orange, Document 3) = 2 / 4 = 0.50
 ● TF(banana, Document 3) = 1 / 4 = 0.25
 Step 2: Calculate Inverse Document Frequency (IDF)
 ● Compute the IDF for each term in the corpus.
 ● IDF(apple) = log(3 / 2) ≈ 0.176
 ● IDF(banana) = log(3 / 3) = 0
 ● IDF(orange) = log(3 / 2) ≈ 0.176
 Step 3: Calculate TF-IDF
 ● Multiply the TF by the IDF for each term in each document.
 Document 1:
 ● TF-IDF(apple, Document 1) = 0.67 * 0.176 ≈ 0.118
 ● TF-IDF(banana, Document 1) = 0.33 * 0 = 0
 Document 2:
 ● TF-IDF(banana, Document 2) = 0.50 * 0 = 0
 ● TF-IDF(orange, Document 2) = 0.50 * 0.176 ≈ 0.088
 Document 3:
 ● TF-IDF(apple, Document 3) = 0.25 * 0.176 ≈ 0.044
 ● TF-IDF(orange, Document 3) = 0.50 * 0.176 ≈ 0.088
 ● TF-IDF(banana, Document 3) = 0.25 * 0 = 0
 Step4: Query Representation
 ● Represent the query "apple banana" using the 
TF-IDF scores.
 Query:
 ● TF(apple, Query) = 1 / 2 = 0.50
 ● TF(banana, Query) = 1 / 2 = 0.50
 ● TF-IDF(apple, Query) = 0.50 * 0.176 ≈ 0.088
 ● TF-IDF(banana, Query) = 0.50 * 0 = 0
 Note:  When calculating TF-IDF for the terms in a 
query, we're essentially treating the query as a 
document and comparing it to the entire corpus. 
However, the IDF is calculated based on the 
corpus, not the query itself.
Vector Space Model
 ● The vector model recognizes that the use of binary weights is too limiting and proposes a 
framework in which partial matching is possible.
 ● This is accomplished by assigning non-binary weights to index terms in queries and in documents.
 ● These term weights are ultimately used to compute the degree of similarity between each 
document stored in the system and the user query. 
● By sorting the retrieved documents in decreasing order of this degree of similarity, the vector model 
takes into consideration documents which match the query terms only partially. 
● It allows for more flexible and nuanced retrieval compared to the Boolean model, as it considers 
the similarity between the query vector and document vectors rather than exact matches of terms.
 ● The Vector Space Model (VSM) is a classic and widely used Information Retrieval (IR) model that 
represents documents and queries as vectors in a multi-dimensional space. 
Vector Space Model
 ● Key Concepts
 


Example 
Documents:
 1.
 2.
 3.
 Document 1: "Information retrieval is the process of obtaining relevant information."
 Document 2: "The Vector Space Model is widely used in information retrieval."
 Document 3: "Search engines employ various models for efficient retrieval."
 Query: 
      "information retrieval model"
3.     Calculate Term Frequency (TF):Calculate the term frequency for each term in each document and also for each query term.
 Document 1 TF:
 ● information: 2
 ● retrieval: 1
 ● is: 1
 ● the: 1
 ● process: 1
 ● of: 1
 ● obtaining: 1
 ● relevant: 1
 Document 2 TF:
 ● the: 1
 ● vector: 1
 ● space: 1
 ● model: 1
 ● is: 1
 ● widely: 1
 ● used: 1
 ● in: 1
 ● information: 1
 ● retrieval: 1
 Document 3 TF:
 ● search: 1
 ● engines: 1
 ● employ: 1
 ● various: 1
 ● models: 1
 ● for: 1
 ● efficient: 1
 ● retrieval: 1
 Query TF:
 ● information: 1
 ● retrieval: 1
 ● model: 1
 Documents:
 1. Document 1: "Information retrieval is the process of obtaining relevant information."
 2. Document 2: "The Vector Space Model is widely used in information retrieval."
 3. Document 3: "Search engines employ various models for efficient retrieval."
 Query: 
      "information retrieval model"
4. Calculate Inverse Document Frequency (IDF):
 ● Calculate IDF for each term in the vocabulary
 ● Total number of documents,N=3.
 Documents:
 1. Document 1: "Information retrieval is the process of obtaining 
relevant information."
 2. Document 2: "The Vector Space Model is widely used in 
information retrieval."
 3. Document 3: "Search engines employ various models for efficient 
retrieval."
 Query: 
      "information retrieval model"
5. Calculate TF-IDF:
 ● Multiply TF by IDF for each term in each document and query.
 Document 3 TF-IDF:
 ● retrieval: 1 x 0 = 0
 ● search : 1 x 0.477 =0.477
 ● engines : 1 x 0.477 =0.477
 ● employ : 1 x 0.477 =0.477
 ● various : 1 x 0.477 =0.477
 ● models : 1 x 0.477 =0.477
 ● for : 1 x 0.477 =0.477
 ● efficient : 1 x 0.477 =0.477
 Query Vector: TF-IDF
 ● information =1 × 0.176 = 0.176
 ● retrieval = 1 × 0 = 0
 ● model = 1 × 0.477 = 0.477
6.
 0,0,0,0,0,0,0,0,0
7. Calculate Cosine Similarity between query vector and each of document vectors
Step 8: Ranking Documents
 Rank documents based on their cosine similarity scores. Higher similarity scores indicate higher relevance to the 
query "information retrieval model".
 Document 1 and Query = 0.117
 Document 2 and Query = 0.471
 Document 3 and Query = 0.415
 ● Document 2 has the highest cosine similarity with the query, indicating it is the most relevant 
document to the query "information retrieval model."
Vector Space Model : Advantages and Disadvantages
 Advantages
 1.
 2.
 3.
 Flexibility:
 ○ Handles partial matches and degrees of relevance through cosine similarity scores.
 ○ Can rank documents based on relevance rather than just binary (relevant or not).
 Term Weighting:
 ○ Considers the importance of terms in both documents and queries, using TF-IDF to distinguish 
between frequent and important terms.
 Efficiency:
 ○ Efficient retrieval using precomputed document vectors and fast cosine similarity calculations.
 Disadvantages
 1.
 2.
 3.
 Curse of Dimensionality:
 ○ As the number of unique terms (dimensions) increases, the vector space becomes sparse and 
computation-intensive.
 Handling Synonyms and Polysemy:
 ○ Relies on exact term matching unless expanded with techniques like query expansion or semantic 
analysis.
 Scalability:
 ○ Scaling to large document collections and queries can pose challenges due to the computational 
requirements of vector operations.
Example : Vector Space Model
Probabilistic IR Models
  a: doc is relevant and a bar : a document is not relevant
Log of Odds Ratio
 X is probability. logit(x)= prob of observed / prob of not 
observed
Initially user gives the query say (k1,k2)
 IR system retrieves a set of documents say D1 to D5.
 User will click on the document if he fills that it may be relevant. 
He will not click the doc if he fills that it may not be relevant.
 However it does not mean that if he don't click the doc, it is completely irrelevant document. Or if 
he clicks it is relevant.
 Here we need to calculate prior prob and likelihood



Expression for Ranking Computations in Probabilistic IR Models:
 Wi,q=weight factor of a term w.r.t query
 Wi,j = weight factor of a term w.r.t document dj
 These weight factors are binary in nature.
Query Q=(K1,K2)
 Find  Sim( D1,Q) 
Query Q=(K1,K2)
 Find  Sim( D2,Q) ,Sim( D3,Q),Sim( D4,Q),Sim( D5,Q)   
Find  Sim( D4,Q) ,Sim( D5,Q) 
Ranking: D2,D5,D4,D1,D3
Probabilistic IR Models : Advantages and Disadvantages
 Advantages
 1.
 2.
 3.
 Probabilistic Ranking:
 ○ Provides a probabilistic estimate of document relevance rather than binary (relevant or not) or similarity scores.
 ○ Can handle uncertainty and varying degrees of relevance more effectively.
 Feedback Mechanisms:
 ○ Supports relevance feedback mechanisms where user interactions (e.g., clicks on retrieved documents) update the 
model's estimates.
 Flexibility:
 ○ Can be adapted to different types of documents and queries by adjusting the probabilistic parameters and models.
 Disadvantages
 1.
 2.
 3.
 Complexity:
 ○ Implementing and tuning probabilistic models can be more complex than Boolean or Vector Space models.
 ○ Requires accurate estimation of probabilities and parameters, which may be challenging.
 Assumptions:
 ○ Relies on assumptions such as conditional independence of terms given relevance, which may not always hold in 
practice.
 Performance:
 ○ May require more computational resources compared to simpler models, especially for large-scale document 
collections.
Alternative Set Theoretic Model
 ● Alternative Set Theoretic Information Retrieval (IR) models provide different approaches to retrieving 
information based on set theory principles, often complementing or expanding upon the traditional 
Boolean model.
 ● A few notable alternatives includes:
 1.
 2.
 Extended Binary Model (EBM)
 Fuzzy Set Model
Extended Boolean Model
 ● In Boolean retrieval as there is no provision for term weighting, no ranking of the answer set is generated. 
As a result, the size of the output might be too large or too small.
 ● Because of these problems, modern information retrieval systems are no longer based on the Boolean 
model. In fact, most of the new systems adopt at their core some form of vector retrieval. 
○ The reasons are that the vector space model is simple, fast, and yields better retrieval performance. 
● One alternative approach is to extend the Boolean model with the functionality of partial matching and 
term weighting. i.e EBM is based on the idea of extending the Boolean model with features of the vector 
model.
 ● This strategy allows us to combine Boolean query formulations with characteristics of the vector model. 
Extended Boolean Model: Formal Characterization



Keyword space Representation for AND and OR operators for two-term queries
For AND query,point (1,1) is the most desirable spot. This suggests taking the complement of the distance form (1,1) as a measure of 
similarity w.r.t AND query.Also such distances can be normalized.
 For OR  query,point (0,0)  is the spot to be avoided. This suggests taking the distance form (0,0) as a measure of similarity w.r.t OR query. 
Also such distances can be normalized.
In EBM, even if some term is absent in doc, we get some similarity with AND query. So ranking is there. In simple BM, it 
will give 0 value i.e irrelevant document
 In case of OR query , in SBM,there is no calibration even if one of the term is absent or both are present(in both, it gives 
1). But in EBM is there. It gives some value other than 1
 For AND query , we are taking the distance from (1,1) and for OR, from(0,0)
Generalized formula for Extended Boolean Operators
Problem with EBM?
 Here    Xi 
= w1,j  
 = Weight of K1 in Dj
Example: EBM
 ● Doc1: "Information retrieval is the process of obtaining information from a large repository."
 ● Doc2: "Information retrieval systems are used in digital libraries."
 ● Doc3: "Boolean logic is a cornerstone of traditional information retrieval."
 ● Doc4: "Extended Boolean models improve upon the classic Boolean retrieval approach."
 Query
 Suppose the query is: "(Information AND retrieval) OR systems"

Example Scenario: Imagine you are searching for information about artificial intelligence (AI) and its applications in 
healthcare. You are interested in finding documents that discuss AI in medical diagnostics and treatment.
 Traditional Boolean Model:  
● Query: AI AND healthcare AND (diagnostics OR treatment)
 In this model:
 ● You must find documents that contain all the specified terms exactly as they appear in your query ("AI", "healthcare", 
"diagnostics", "treatment").
 ● If a document doesn't have all these terms in the exact way you've specified, it will not be retrieved, even if it 
discusses related topics.
 Extended Boolean Model (EBM): In this Model, your query could be expanded and weighted to better capture relevant 
documents:
 ● Query: AI OR artificial intelligence AND healthcare OR medical AND (diagnostics OR treatment)
 In this model:
 ● The terms "AI" and "artificial intelligence" are treated as synonymous, so documents containing either term will be 
retrieved.
 ● Similarly, "healthcare" and "medical" are considered related, broadening the scope of relevant documents.
 ● The terms "diagnostics" and "treatment" are used with an OR operator, allowing documents that discuss either or 
both topics to be retrieved.
Additionally:
 ● The EBM might assign weights to terms based on their relevance or importance. For example, if a 
document mentions "AI in medical diagnostics" multiple times, it might receive a higher relevance score 
than a document that briefly mentions "AI in healthcare".
 ● Partial matches or synonyms (like "AI" and "artificial intelligence") can also be accounted for, improving the 
retrieval of documents that use different but related terms.
Advantages of EBM:
 1.
 2.
 3.
 4.
 5.
 Improved Relevance: EBM allows for term weighting, meaning more relevant documents can be ranked 
higher based on the importance or frequency of query terms within them. This enhances the precision of 
search results by prioritizing documents that are more closely related to the query.
 Partial Matching: Unlike the strict Boolean model, which requires exact matches between query terms and 
document content, EBM supports partial matching. This means documents containing variations or related 
terms to those in the query can still be retrieved, improving recall (the ability to retrieve relevant 
documents).
 Synonym Handling: EBM can accommodate synonyms and related terms, broadening the scope of 
retrieval. For example, it can recognize "AI" and "artificial intelligence" as equivalent terms, ensuring that 
documents using either term are retrieved.
 Flexibility in Query Construction: EBM allows for more complex queries using Boolean operators (AND, 
OR, NOT) and parentheses to combine terms and specify relationships between them. This flexibility 
enables users to construct more precise queries tailored to their information needs.
 Conceptual Retrieval: By incorporating concepts and semantic relationships between terms, EBM can 
capture documents that are conceptually related to the query, even if they do not contain the exact terms 
specified. This enhances the overall effectiveness of information retrieval.
Disadvantages of EBM:
 1.
 2.
 3.
 4.
 5.
 Complexity: Implementing and understanding EBM can be more complex compared to the traditional 
Boolean model. Users may need more training or expertise to construct effective queries and interpret 
search results correctly.
 Computational Overhead: EBM involves more complex computations, such as term weighting and 
handling synonyms, which can increase computational overhead compared to the simpler Boolean model. 
This may affect the speed of retrieval in large-scale systems.
 Risk of Over-retrieval: Because EBM allows for partial matches and synonym handling, there is a risk of 
retrieving too many irrelevant documents (over-retrieval). This can happen if the model is not properly tuned 
or if query terms are too broad or ambiguous.
 Dependency on Quality of Indexing: EBM relies heavily on the quality of the indexing process, which 
involves assigning weights to terms and capturing semantic relationships. If indexing is not done well, 
retrieval effectiveness may be compromised.
 User Training Required: Due to its complexity and flexibility, users may need training to fully utilize EBM's 
capabilities and construct effective queries. This could be a barrier in environments where users are not 
familiar with advanced search techniques.
Alternative Set Theoretic Model: Fuzzy Set Model
Fuzzy set Theory
 ● The fuzzy set theory is an extension of the classical set theory designed to handle the concept of 
partial membership making it applicable in various fields where binary classification is insufficient.. 
● In traditional set theory,membership is binary. 
● The fuzzy set theory deals with the representation of the classes whose boundaries are not well 
defined.
 ● The key idea is to associate a membership function with the elements of the class.
 ● This function takes the values in the interval [0,1].
 ● Thus membership in fuzzy set is a intrinsically gradual instead of abrupt (as in conventional 
boolean logic)classic
 ● Fuzzy set theory allows elements to have degrees of membership in a set, which is useful for 
dealing with uncertain or imprecise information.


Example: Fuzzy Algebra
Fuzzy Information Retrieval Model
 ● A thesaurus in the context of fuzzy sets refers to a structured representation of terms that 
captures the relationships and degrees of similarity between them, allowing for more flexible and 
nuanced information retrieval. 
● The fuzzy set approach to a thesaurus enables the representation of synonyms, antonyms, and 
related(contextually) terms  with varying degrees of membership, which can be particularly useful for 
enhancing search queries and improving retrieval accuracy.
 ● The basic idea is to expand index terms in the query with related terms(obtained from the 
thesaurus)  such that additional relevant documents(i.e besides the ones that would be normally 
retrieved) can be retrieved by the user query.

Fuzzy Information Retrieval Model
Example

The correlation matrix c can be used to define fuzzy set associated to each index term Ki. In this fuzzy set , the document 
Dj has a degree of membership 
Degree of membership of 
document Dj in the fuzzy 
set associated to index 
term Ki
 Multiplication 
of (1-ci,l
 )






Fuzzy Set Model Advantages and Disadvantages:
Applications of Fuzzy Set Model
 ● Information Retrieval: Fuzzy set models can handle imprecise queries and rank documents based 
on the degree of relevance.
 ● Control Systems: Fuzzy logic controllers are used in systems where precise control is difficult, such 
as automatic transmission systems and climate control.
 ● Decision Making: Fuzzy set theory is used in multi-criteria decision making where criteria and 
alternatives are not sharply defined.
 ● Pattern Recognition: Fuzzy sets can model and classify data that are not crisply defined.
Query: step, man, mankind china







Suppose that a user remembers that the document he is interested in contains the page  
Here , user would like to express his query using a more 
richer expression such as —- which conveys the details in his 
visual recollection





● To allow searching for index terms and for text regions, a single inverted file is built in 
which each structural component stands as an entry in the index. 
● Associated with each entry, there is a list of text regions as a list of occurrences. 
● Moreover, such a list could be easily merged with the traditional inverted file for the 
words in the text. 
● Since the text regions are non-overlapping, the types of queries which can be asked 
are simple: 
(a) select a region which contains a given word (and does not contain other regions); 
(b) select a region A which does not contain any other region B (where B belongs to a 
list distinct from the list for A); 
(c) select a region not contained within any other region, etc.
Key Concepts of Non-Overlapping Lists Structured Retrieval Model
 1.
 1.
   3.       
Document Segmentation:
 ○ Non-Overlapping Segments: Documents are divided into distinct, non-overlapping segments or 
lists. Each segment might represent a different logical part of the document, such as chapters, 
sections, or specific fields (e.g., title, abstract, body).
 ○ Independence of Segments: Since the segments do not overlap, each one is treated as an 
independent entity during retrieval. This allows for more precise matching of queries to specific 
parts of the document.
 Indexing Process:
 ○ Separate Indexes for Each Segment: Each segment or list is indexed separately. For instance, in 
an academic paper, there would be separate indexes for the title, abstract, introduction, and 
conclusion.
 ○ Field-Specific Terms: The terms within each segment are only associated with that specific 
segment in the index, ensuring that searches within a segment do not inadvertently retrieve content 
from another segment.
 Query Representation:
 ●Segment-Specific Queries: Queries can be directed at specific segments. For example, a user might 
want to search only within the "Abstract" or "Conclusion" sections of documents.
 ●Fielded Queries: Users can specify which part of the document they are interested in by using fielded 
queries (e.g., abstract: "information retrieval"). This directs the retrieval process to 
consider only the relevant segment.
4.    Retrieval Process:
 ● Independent Scoring: Each segment of the document is scored independently based on its relevance to 
the query. The overall document score can then be a combination of these individual segment scores, often 
using a weighted sum.
 ● Non-Overlapping Contribution: Since the segments do not overlap, the relevance of one segment does 
not directly influence the relevance score of another, ensuring that each part of the document contributes 
distinctly to the final score.
 5.    
Aggregation of Results:
 ● Combining Segment Scores: The final retrieval score for a document is an aggregation of the scores from 
the relevant segments. Depending on the application, different segments might be weighted differently 
based on their importance.
 ● Result Presentation: Retrieved documents might be presented with highlights or excerpts from the specific 
segments that matched the query, making it clear which part of the document is relevant.
Example Application: academic paper retrieval.
 Scenario
 You are designing a search engine for an academic database that contains research papers. Each paper is divided into 
non-overlapping sections: Title, Abstract, Introduction, Methodology, Results, Conclusion, and References. Users of 
the system often have specific information needs that pertain to particular sections of these papers.
 Document Structure and Indexing
 Each academic paper is divided into the following non-overlapping segments:
 ● Title
 ● Abstract
 ● Introduction
 ● Methodology
 ● Results
 ● Conclusion
 ● References
 For each paper, these segments are independently indexed. This means:
 ● The Title field is indexed separately, allowing for searches that specifically target the paper's title.
 ● The Abstract is indexed as a separate segment, so users can search within the abstracts across multiple papers.
 ● The Introduction, Methodology, Results, Conclusion, and References are each independently indexed.
Example Document
 Consider the following simplified academic paper:
 ● Title: "Advances in Information Retrieval Techniques"
 ● Abstract: "This paper reviews recent advances in information retrieval, focusing on machine learning algorithms."
 ● Introduction: "Information retrieval has become a critical field with the growth of the internet. Traditional methods have 
limitations that modern techniques seek to overcome."
 ● Methodology: "We applied a support vector machine (SVM) classifier to a dataset of 10,000 documents to evaluate retrieval 
accuracy."
 ● Results: "Our SVM-based approach achieved a 15% improvement in retrieval precision over baseline methods."
 ● Conclusion: "The study demonstrates that machine learning algorithms significantly enhance information retrieval 
performance."
 ● References: A list of cited papers and books.
 Indexing Process
 Each segment is treated as an independent entity during indexing:
 ● Title Index: "Advances in Information Retrieval Techniques"
 ● Abstract Index: "This paper reviews recent advances in information retrieval, focusing on machine learning algorithms."
 ● Introduction Index: "Information retrieval has become a critical field..."
 ● Methodology Index: "We applied a support vector machine (SVM)..."
 ● Results Index: "Our SVM-based approach achieved a 15% improvement..."
 ● Conclusion Index: "The study demonstrates that machine learning algorithms..."
 ● References Index: (Each reference is indexed separately.)  
Query Example 1: Simple Query Targeting a Specific Segment
 A researcher is interested in finding papers that have applied Support Vector Machines (SVMs) in 
the Methodology section. 
The researcher uses the following query: methodology: "Support Vector Machines" 
Retrieval Process:
 Result:
 The paper mentioned on previous slide (No 118) would be retrieved because its Methodology segment 
includes "We applied a support vector machine (SVM)..."
 1. Search Within the Methodology Segment: The retrieval model searches only within the 
Methodology segments of all indexed papers.
 2. Independent Scoring: Each document's Methodology section is scored based on its relevance to 
the query "Support Vector Machines."
 3. Result Ranking: Documents are ranked based on the score derived from their Methodology 
segment. Only those papers where the Methodology explicitly mentions "Support Vector Machines" 
will be retrieved.
Summary
 A structured retrieval model based on non-overlapping lists provides a highly structured and precise 
approach to information retrieval, particularly suited to domains where documents have well-defined, 
independent sections. By treating each segment as an independent entity, this model ensures that 
retrieval is focused and relevant, providing clear and interpretable results to users.
Summary
 A structured retrieval model based on non-overlapping lists provides a highly structured and precise 
approach to information retrieval, particularly suited to domains where documents have well-defined, 
independent sections. By treating each segment as an independent entity, this model ensures that 
retrieval is focused and relevant, providing clear and interpretable results to users.
Query Example 2: Complex Query with Multiple Segments
 A researcher is interested in papers that discuss machine learning algorithms in the Abstract and also provide 
significant findings in the Results section. 
The query:   abstract: "machine learning algorithms" AND results: "significant findings"
 
Retrieval Process:
 1. Search Within the Abstract Segment: The retrieval model first searches within the Abstract segments 
of all indexed papers for the phrase "machine learning algorithms."
 2. Search Within the Results Segment: It then searches within the Results segments for the phrase 
"significant findings."
 3. Independent Scoring and Aggregation:
 ○ Each document's Abstract and Results segments are scored independently.
 ○ The final relevance score for each document is an aggregation (e.g., weighted sum) of the 
individual scores from the Abstract and Results segments.
 Result:
 The paper on previous slide (No 118) would be retrieved because:
 ● The Abstract mentions "machine learning algorithms."
 ● The Results indicate significant findings, stating, "Our SVM-based approach achieved a 15% 
improvement in retrieval precision."
Advantages and Challenges: 
Advantages of Non-Overlapping Lists Model
 ● Precision: By ensuring that each segment is independent, this model allows for highly precise 
retrieval, targeting exactly the part of the document relevant to the query.
 ● Clarity: Since the retrieval is based on non-overlapping segments, users can easily understand 
which part of the document is relevant, making it easier to interpret results.
 ● Scalability: This approach scales well in structured environments, such as legal or academic 
databases, where documents naturally lend themselves to segmentation.
 Challenges
 ● Complexity of Query Formulation: Users need to be familiar with the document structure 
to effectively formulate queries targeting specific segments.
 ● Segment Weighting: Determining appropriate weights for different segments can be 
challenging and might require domain-specific knowledge or user feedback.
Structured Retrieval Models based on Proximal Nodes
 ● Structured retrieval models based on proximal nodes involve a sophisticated approach to information 
retrieval (IR) that enhances traditional models by considering the proximity of relevant nodes (or terms) 
within a document or database.
 ● This method focuses not only on the occurrence of individual terms but also on their co-occurrence and 
relative positions, thus improving the relevance and accuracy of search results.
 Key Concepts:
 1. Proximal Nodes: These are points or locations within a document or dataset that are close to each 
other and contain related information. The proximity is typically measured in terms of distance within 
the text or graph, such as the number of words, sentences, or links between terms.
 2. Structured Retrieval Models: These models extend the traditional IR models (like vector space 
models or probabilistic models) by incorporating structured data such as graphs, trees, or sequences 
that represent the relationship between different elements. This structure allows the model to 
consider the connections and proximity between nodes.
Key Concepts:
 3.
 3.
 3.
 3.
 Proximity-Based Scoring: The core idea behind these models is that the closer the relevant 
nodes (terms) are to each other, the more likely the document is to be relevant to the query. For 
example, if two search terms appear close together in a document, that document may be ranked 
higher because it's more likely to address the query's intent.
 Graph-Based Retrieval: One common approach is to represent documents and queries as 
graphs where nodes represent terms or concepts, and edges represent relationships or 
proximity. The retrieval process then involves searching for subgraphs or patterns within these 
graphs that match the query.
 Query Expansion: Structured retrieval models often involve query expansion techniques where 
the original query is enhanced by adding related terms or concepts that are connected through 
the graph structure. This can improve the retrieval of relevant documents that might not contain 
the exact query terms but are semantically related.
 Applications: These models are particularly useful in domains where relationships between 
concepts are crucial, such as biomedical literature retrieval, legal document analysis, or any 
context where the structure of information plays a significant role.
Example: Legal Document Retrieval Using Structured Retrieval Models Based 
on Proximal Nodes
 Scenario: A legal researcher is investigating how "trade secrets" are protected under "non-compete 
agreements" in the context of "intellectual property" law.
 Data Representation:
 1.
 Document Collection:
 ●   The corpus consists of legal documents, such as court cases, legal analyses, and statutes, where terms 
related to "trade secrets," "non-compete agreements," and "intellectual property" might appear.
 2.
 Graph Construction:
 ●   Each legal document is transformed into a graph. Here’s how this works:
 ○ Nodes: Key legal concepts or terms are identified and treated as nodes. In this case, terms like "trade 
secrets," "non-compete agreements," and "intellectual property" become nodes.
 ○ Edges: Relationships between these nodes are represented by edges. The weight of an edge reflects 
the proximity of these terms within the document:
 ■ Sentence-level proximity: Terms appearing within the same sentence have the highest 
proximity score.
 ■ Paragraph-level proximity: Terms appearing in the same paragraph but different sentences 
have a slightly lower proximity score.
 ■ Section-level proximity: Terms appearing in the same section or chapter have the lowest 
proximity score.
3.   Example Document Graph:
 ● Suppose we have a legal document with the following relevant excerpt:
 ○ Paragraph 1: "Intellectual property rights often intersect with trade secrets. A non-compete 
agreement can protect these secrets by restricting former employees from joining competitors."
 ○ Paragraph 2: "In cases involving intellectual property, courts have upheld the use of non-compete 
agreements to prevent the misuse of trade secrets."
 ● This document would be represented as a graph:
 ○ Nodes: "Intellectual property," "trade secrets," "non-compete agreements"
 ○ Edges:
 ■ High-weight edge between "intellectual property" and "trade secrets" within Paragraph 1 
(sentence-level proximity).
 ■ High-weight edge between "trade secrets" and "non-compete agreements" within Paragraph 1 
(sentence-level proximity).
 ■ Medium-weight edge between "intellectual property" and "non-compete agreements" 
(paragraph-level proximity).
 ■ Another set of edges for the relationships in Paragraph 2, reinforcing the connections.
Query Processing:
 The legal researcher’s query is: "How are trade secrets protected under non-compete agreements 
within the framework of intellectual property law?"
 1. Query Representation:
 ● The query is represented as a set of nodes: "trade secrets," "non-compete agreements," and 
"intellectual property."
 ● Since the query specifically asks about "protection," we might include a node for "protection" 
as well, though it would be connected to the main nodes with a lower weight unless it appears 
in the same context.
 2. Subgraph Matching:
 ● The retrieval model searches for subgraphs within the document graphs that match the structure 
of the query graph.
 ● It looks for clusters of nodes where "trade secrets," "non-compete agreements," and "intellectual 
property" are closely connected, with higher weights assigned to those connections.
3. Scoring and Ranking:
 ● Proximity Scoring:
 ○ Documents where these concepts appear in close proximity (e.g., within the same paragraph or 
sentence) are scored higher.
 ○ If "protection" appears in conjunction with the key terms in a relevant context, this increases the 
score.
 ● Contextual Relevance:
 ○ If a document specifically discusses how "non-compete agreements" protect "trade secrets" as a 
form of "intellectual property," it would receive a higher score due to the direct relevance.
 Detailed Example Output:
 Top-Ranked Document:
 ● Document Excerpt: "In a landmark case, the court ruled that non-compete agreements are essential for 
protecting trade secrets, especially when they are considered part of a company’s intellectual property portfolio. 
The court highlighted that the proximity of an employee's knowledge of trade secrets to their subsequent 
employment poses a significant risk to the intellectual property of the original employer."
 ● Proximity Analysis:
 ○ Nodes: "trade secrets," "non-compete agreements," "intellectual property"
 ○ Edges: Strong edges connecting all three terms within the same sentence.
 ○ Score: High, due to the strong, direct connections.
Moderately Ranked Document:
 ● Document Excerpt: "The enforcement of non-compete agreements can vary, but they are often used to protect 
trade secrets. However, the definition of what constitutes intellectual property in these cases can be complex."
 ● Proximity Analysis:
 ○ Nodes: "trade secrets," "non-compete agreements," "intellectual property"
 ○ Edges: Moderate edges with terms appearing in the same paragraph but not the same sentence.
 ○ Score: Moderate, due to relevant but less tightly connected content.
 Low-Ranked Document:
 ● Document Excerpt: "Trade secrets and intellectual property law are vital for businesses. Non-compete 
agreements are another legal tool used in employment contracts."
 ● Proximity Analysis:
 ○ Nodes: "trade secrets," "intellectual property," "non-compete agreements"
 ○ Edges: Weak or non-existent connections between the nodes (terms mentioned separately without clear 
relationships).
 ○ Score: Low, due to the lack of proximity and direct connection between the terms.
 Conclusion:
 In this example, the structured retrieval model effectively identifies and ranks documents based on the proximity and 
context of key legal concepts. It goes beyond mere keyword matching by considering how closely related the terms are 
within the text, thus providing more relevant and contextually accurate search results for the legal researcher.
Advantages and Challenges:
 Advantages:
 ● Enhanced Relevance: By considering the proximity of related terms, these models can improve 
the accuracy of search results, especially for complex queries.
 ● Context-Aware: The use of structured data allows the model to understand and utilize the 
context in which terms appear, leading to more meaningful retrieval.
 ● Flexibility: These models can be adapted to various types of structured data, such as text, 
graphs, or even semi-structured data like XML.
 Challenges:
 ● Complexity: Implementing and computing proximity-based models can be computationally 
expensive, especially for large datasets.
 ● Data Structure Dependency: The effectiveness of these models heavily depends on the quality 
and structure of the underlying data, which may not always be well-defined or consistent.



Reading Link : 
Models for browsing

Extensions of Vector Space Models:Alternative Algebraic Models
 ● While the basic vector space model represents text or data as vectors in a high-dimensional space, 
several extensions and modifications have been developed to address the limitations of the original 
model and to better capture the complexities of the data. 
● Here are some notable extensions:
 ○ Generalized VSM
 ○ Latent Semantic Indexing(LSI)
 ○ Neural Networks Model
 Reading Link:  
Alternative Algebraic Models
