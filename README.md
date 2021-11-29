# Survival analysis of mRNA expression dataset

The analysis is divided into 3 parts. Each part is divided into function which runs from the main function.
Each part functions are noted by a comment (#Part x)

External libraries that were used (Need to be installed): matplotlib, lifelines, scipy, statsmodels, sklearn, panadas, numpy

#### Running the script using: "python main.py" (After installing necessary libraries)


Below you will find the output files for each part and all the answers to the exercise questions.

## Task 1

### Part 1 - Data parsing

Data was merged using pandas.

Output: Task1Complete-mRNA.csv (parsed data file)

### Explore the relationship between genomic and clinical/demographical characteristics.
All outputs were saved to Output/Task1 folder


The mRNA data was normalized by CPM (to avoid sequencing bias) and lowly expressed genes were removed (to avoid possible noise at low expression).

The criteria used to filter lowly expressed genes: At least 1 cpm in at least 50% of the samples.

##### a. Identify the 1000 most variable genes across the cohort in terms of expression.

Output file: (a)-Top-1000-Variance-Genes.csv

##### b. For each gene found in a. test the association between its expression and patient survival.

Output file: (b)-Significant-Genes-Cox-Results.csv

##### c. Perform a similar analysis to the one performed in (a), but this time the analysis should control for the age at diagnosis, gender and race of the patients.

Output file: (c)-Significant-Genes-Excluding-Demographics-Significance-Cox-Results.csv

##### d. Are there any genes found in (b) but not in (c)? If so, keep their names in a designated vector. What could be the reason for such a case?

Output file: (d)-Significant-Genes-With-Demographics-Cox-Results.csv

###### Answer:
The reason there are genes found in b but not in c is that 
these genes expression is related to the patient group, their demographics such as patient age and race.
Using the demographic as control allows us to find the genes that affect survival that are not related to the different patients group.


## Task 2
### Identify the respective theoretical distribution of the dataset
All outputs were saved to Output/Task2 folder


I've divided the distribution into 2 classes: Continuous distribution and discrete distributions and checked which of the distributions fit the data the best.

For the continuous distributions i've used the Kolmogorov-Smirnov test for goodness of fit.

For the discrete distributions i've calculated the likelihood to this distributions. 

Output file: Mystery-distributions-results.csv

## Task 3
### NSCLC patients 2 groups survival analysis
All outputs were saved to Output/Task3 folder

##### a. Identify and give a visual representation of the two subtypes. 

I separated the 2 groups using PCA in 2 dimensions and labeled them using k-means (The result is plotted in the output file)

Output file: PCA-NSCLC.png

##### b. Is there a survival difference between the two subtypes? Prove your answer by statistical means.

There is a significance difference in survival between the groups with fdr - 3.7591714904928348e-06

Output file: Significance-Target-Groups.csv 