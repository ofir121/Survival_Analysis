import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection
from typing import List, Tuple
from pandas import Series, DataFrame
import scipy.stats as st
import pandas as pd
import os
import numpy as np
from lifelines import CoxPHFitter


############## Task 1 ####################


def read_clinical_data(input_dir: str):
    """
    Read the clinical data on the patients
    :param input_dir: Input directory for files
    :return: A dataframe with clinical data
    """
    input_file = os.path.join(*[input_dir, 'clinical_data_task1.csv'])
    return pd.read_csv(input_file, index_col=0)


def read_rna_matrix(input_dir: str, output_dir: str):
    """
    Read all the mRNA files and merge them to a matrix
    :param input_dir: Input directory, inside should be dir mRNA files
    :param output_dir: Output directory for merged dataframe
    :return: A merged dataframe with all the mRNA expression, columns are ordered by clinical case id in clinical_data_task1 file
    """
    print("Reading mRNA files")
    output_file = os.path.join(*[output_dir, "Complete-mRNA.csv"])
    dfs = []
    mrna_files_dir = os.path.join(*[input_dir, 'mRNA_files'])
    file_names = [f for f in os.listdir(mrna_files_dir) if os.path.isfile(os.path.join(mrna_files_dir, f))]
    for f in file_names:
        file_path = os.path.join(*[mrna_files_dir, f])
        df_sample = pd.read_csv(file_path, index_col=0)
        sample_name = f.replace(".csv", "")
        df_sample.columns = [sample_name]  # Set column name as the sample name
        dfs.append(df_sample)
    df_result = pd.concat(dfs, axis=1)

    # Order patients columns by the clinical data
    clinical_cases = list(read_clinical_data(input_dir).index)
    df_result = df_result.reindex(columns=clinical_cases)

    df_result.to_csv(output_file)
    return df_result


def get_na_columns(df: DataFrame) -> List[str]:
    """
    Get a list of all the columns with NA in the dataframe
    :param df: Dataframe
    :return: List of column names that contain NA values
    """
    return df.columns[df.isna().any()].tolist()


def filter_lowly_expressed_genes(cpm_counts_df: DataFrame, X: float, Y: float, to_print=True) -> DataFrame:
    """
    Filter the count data for lowly-expressed genes.
    Only keep genes with a CPM >= X in at least Y% samples
    :param cpm_counts_df:
    :param X: The minimum number of CPM for at least Y% samples
    :param Y: The percentage of number of samples that need to pass threshold in each group
    :param Z: The minimum number of groups that need to pass the filter for each gene
    :param to_print: Print step name
    :return: A filtered dataframe without lowly-expressed genes
    """
    to_keep_genes = []
    count_passed_filter_genes_dict = defaultdict(lambda: 0)
    if to_print:
        print("Filtering lowly expressed genes")
    num_of_samples = len(cpm_counts_df)
    pass_filter_count = cpm_counts_df[cpm_counts_df > X].count()
    min_num_of_samples = num_of_samples * (Y / 100)
    for g, passed_sample_count in pass_filter_count.items():
        if passed_sample_count >= min_num_of_samples:
            count_passed_filter_genes_dict[g] += 1
    for k, v in count_passed_filter_genes_dict.items():
        if v >= 1:
            to_keep_genes.append(k)
    filtered_df = cpm_counts_df[to_keep_genes]
    return filtered_df


def clean_data(df: DataFrame):
    """
    Clean the dataframe from NA values by mean of feature
    :param df: Dataframe
    :return: A cleaned from NA dataframes
    """
    df_cleaned = df.copy()
    # Fill NA with mean of feature
    for col in get_na_columns(df_cleaned):
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned.groupby(level='target')[col].transform('mean'))
    return df_cleaned


def get_top_k_variance_genes(df: DataFrame,  output_dir: str, k=1000) -> List:
    """
    Get the top k genes with the highest variances in the dataframe
    :param df: Dataframe
    :param k: The number of top elements to return
    :param output_dir: Output directory
    :return: A list of top k genes with the highest variances
    """
    output_file = os.path.join(*[output_dir, f"(a)-Top-{k}-Variance-Genes.csv"])
    df_variances = df.transpose().var()
    high_variance_genes = list(df_variances.nlargest(k).index)
    df_result = pd.DataFrame(pd.Series(high_variance_genes, name="Gene"))
    df_result.to_csv(output_file)
    return high_variance_genes


def run_cox_analysis(df_test_variables: DataFrame, df_clinical: DataFrame, p_value_threshold: float = 0.05):
    """
    Run cox analysis on all columns in dataframe except {'status','time'}
    :param df_test_variables: Dataframe with all the test variables for cox analysis
    :param df_clinical Dataframe with all the clinical features of the patients, includes status and time
    :param p_value_threshold: p-value threshold for significance
    :return: Significant variables of the analysis
    """
    results = defaultdict(list)
    df = pd.concat([df_test_variables, df_clinical], axis=1)

    for c in df_test_variables.columns:
        df_gene = df[[c] + list(df_clinical.columns)]
        try:
            result = CoxPHFitter().fit(df_gene, duration_col='time', event_col='status')
        except:
            print(f"Gene: {c} skipped")
            continue
        results['pvalue'].append(result.summary["p"][0])
        results['coef'].append(result.summary["coef"][0])
        results['Variable'].append(list(df_gene.columns)[0])
    results['fdr'] = fdrcorrection(results['pvalue'])[1]  # Many tests were made and therefore FDR will adjust the pvalue with fdr to reduce false positives
    df_results = pd.DataFrame.from_dict(results)
    df_results_significant = df_results[df_results['fdr'] < p_value_threshold]
    df_results_significant.set_index('Variable', inplace=True)
    df_results_significant['survival_effect'] = df_results_significant['coef'].apply(
        lambda x: 'Negative correlated with survival' if x > 0 else 'Positive correlated with survival')

    print(df_results_significant)
    return df_results_significant


def prepare_clinical_df(df_clinical: DataFrame):
    """
    Prepare the clinical dataframe encoding for the algorithm
    :param df_clinical: Dataframe with all the clinical features
    :return: Dataframe encoded ready for analysis
    """
    columns_to_keep = ['race', 'gender', 'age_at_diagnosis', 'vital_status', 'days_to_last_follow_up']
    df_result = df_clinical[columns_to_keep]

    # Adjust gender to 1 and -1
    df_result["gender"] = df_result["gender"].apply(lambda x: 1 if x == "male" else -1)

    df_result.rename(columns={'vital_status': 'status', 'days_to_last_follow_up': 'time'}, inplace=True)
    df_result['status'].replace({'Alive': False, 'Dead': True}, inplace=True)

    # One hot encode race
    df_result = pd.get_dummies(df_result, columns=["race"], prefix=["race"])
    # Remove unknown race column
    df_result.drop(columns=['race_Unknown'], axis=1, inplace=True)
    return df_result


def test_expression_survival_association(df_genes: DataFrame, df_clinical: DataFrame, output_dir: str):
    """
    Test gene expression association with survival
    Using demographics as control in order to find only genes that don't relate to demographic features
    :param df_genes: Dataframe of gene expression of all the patients
    :param df_clinical: Dataframe of clinical features of the patients
    :param output_dir Output directory for results
    """

    df_genes = df_genes.transpose()
    df_clinical = prepare_clinical_df(df_clinical)

    # Run without demographics
    print("Results for genes without demographics")
    df_results_no_control = run_cox_analysis(df_genes, df_clinical[['status', 'time']])
    output_file = os.path.join(*[output_dir, "(b)-Significant-Genes-Cox-Results.csv"])
    df_results_no_control.to_csv(output_file)

    # Run with demographics
    print("Results for genes with demographics control")
    df_results_with_demographics = run_cox_analysis(df_genes, df_clinical)

    # Get distinct genes that don't relay on demographics
    df_results_with_control = df_results_no_control[
        ~df_results_no_control.index.isin(df_results_with_demographics.index)].dropna(how='all')
    print("Results for genes that don't relay on demographic")
    print(df_results_with_control)
    output_file = os.path.join(*[output_dir, "(c)-Significant-Genes-Excluding-Demographics-Significance-Cox-Results.csv"])
    df_results_with_control.to_csv(output_file)

    # Get distinct genes that do relay on demographic
    print("Genes that were removed because they relay on demographics")
    df_results_that_were_removed_using_control = df_results_no_control[
        df_results_no_control.index.isin(df_results_with_demographics.index)].dropna(how='all')
    output_file = os.path.join(*[output_dir, "(d)-Significant-Genes-With-Demographics-Cox-Results.csv"])
    df_results_that_were_removed_using_control.to_csv(output_file)
    print(df_results_that_were_removed_using_control)

############### Task 2 ###################


def get_best_distribution(data: Series, print_statistics=True):
    """
    Find the best distribution for the data and its parameters
    :param data: A series of data points from the distribution
    :param print_statistics: Print the statistics to stdout
    :return: The best distribution name, p-value that the data does NOT belong to the distribution, the distribution parameters
    """
    if print_statistics:
        print(f"Checking distribution for: {data.name}")
    # Continuous distributions
    cont_dist_names = ["norm", "gamma", "uniform", "expon"]  #
    # Define variables for discrete distribtuions
    mean = data.mean()
    var = data.var()
    p = mean / var
    n = p * data.mean() / (1 - p)
    args = {'p': p, 'n': n, 'mu': mean}
    # Discrete distributions
    disc_dist = {"geom": {'p': p},
                 "binom": {"n": n, "p": p},
                 "nbinom": {"n": n, "p": p},
                 "poisson": {'mu': mean},
                 "bernoulli": {'p': p}}  # , "hypergeom": {}, }
    dist_results = []
    params = {}
    # Fit for continuous distributions
    for dist_name in cont_dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p_value = st.kstest(data, dist_name, args=param)

        dist_results.append((dist_name, p_value))

    # Fit for discrete distributions
    for dist_name, args in disc_dist.items():
        dist = getattr(st, dist_name)

        p_value = data.map(lambda val: dist.pmf(val, **args)).prod()

        params[dist_name] = ()
        dist_results.append((dist_name, p_value))

    # select the best fitted distribution
    print("Distributions p-values")
    print(dist_results)
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    if print_statistics:
        print(f"Best fitting for {data.name}: " + str(best_dist))
        print("Best p value: " + str(best_p))
        print("Parameters for the best fit: " + str(params[best_dist]))

    return {"Feature name": data.name, "Distribution": best_dist, "Reject pvalue": best_p, "Params": params[best_dist]}


def test_mystery_distribution(output_dir: str):
    """
    Tests each column for a set of matching distributions to get the best distribution that fits each column
    :param output_dir Output directory
    :return: Dict of Feature name, Distribution name, Reject assumption pvalue and params for the distributions
    """
    mystery_distribution_file = os.path.join(*['Data', 'mystery_distributions.csv'])
    output_file = os.path.join(*[output_dir, "Mystery-distributions-results.csv"])

    df = pd.read_csv(mystery_distribution_file)
    result = pd.DataFrame(df.apply(get_best_distribution, axis=0))
    pd.DataFrame.from_records(result).to_csv(output_file, index=False)
    print(result)


############### Test 3 ###############

def load_nsclc_patients(input_dir: str) -> Tuple[DataFrame, DataFrame]:
    """
    Load the NSCLC patients mRNA expression and clinical data
    :param input_dir: Input directory
    :return: 2 Dataframes, genes and clinical information
    """
    input_file = os.path.join(*[input_dir, "mRNA_task3.csv"])
    input_clinical_data_file = os.path.join(*[input_dir, 'clinical_data_task3.csv'])
    df_genes = pd.read_csv(input_file, index_col=0)
    df_clinical = pd.read_csv(input_clinical_data_file, index_col=0)
    return df_genes, df_clinical


def calculate_pca(df: DataFrame, normalize, n_dim: int) -> DataFrame:
    """
    Calculate the principal components of the PCA
    :param df: Dataframe
    :param normalize: Normalize the data
    :param n_dim: Number of dimensions for PCA
    :return: Dataframe with principal components
    """
    # Separating out the features
    x = df.loc[:, df.columns].values
    # Standardizing the features
    if normalize:
        x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n_dim)
    principal_components = pca.fit_transform(x)
    df_principal = pd.DataFrame(data=principal_components,
                                columns=[f'principal component {i}' for i in range(1, n_dim + 1)])

    return df_principal


def predict_labels_using_unsupervised_clustering(x: DataFrame, n_clusters=2) -> Series:
    """
    Predicts the labels of the features (x) using k-means to get predicted groups
    :param x: Features
    :param n_clusters: Number of clusters to predict
    :return: A series of labels
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=100).fit(x)
    labels = pd.Series(kmeans.labels_, name='target')

    return labels


def plot_pca(df: DataFrame, output_dir, suffix, normalize=True):
    """
    Plots PCA for the dataframe
    :param df: A dataframe with features, index will contain the label in column 'type'
    :param output_dir: Output directory for plot
    :param suffix: Suffix for file and title
    :param normalize: Do standard scalar normaliztion, should be used if data is not normalized
    """
    print(f"Plotting PCA for {suffix}")
    output_file = os.path.join(*[output_dir, 'PCA-' + suffix + '.png'])

    df_principal = calculate_pca(df, normalize, n_dim=2)
    y = predict_labels_using_unsupervised_clustering(df_principal)  # Predict groups
    final_df = pd.concat([df_principal, y], axis=1)
    final_df.index = df.index  # Set back index of samples

    # Plot PCA
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA ' + suffix, fontsize=15)
    targets = list(final_df['target'].unique())
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'principal component 1']
                   , final_df.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(targets)
    ax.grid()
    plt.savefig(output_file)
    plt.clf()
    print(f"PCA {suffix} was saved to {output_file}")
    return final_df


def normalize_cpm(counts_df: DataFrame, to_print=True) -> DataFrame:
    """
    Normalize read count by CPM method, genes should be in columns
    :param counts_df: A dataframe with read count for each gene
    :param to_print: print step name
    :return: A CPM normalized dataframe
    """
    if to_print:
        print("Normalizing by CPM")
    res = counts_df.div(counts_df.sum(axis=1), axis=0).mul(pow(10, 6))
    return res


def log_cpm(cpm_counts_df: DataFrame) -> DataFrame:
    """
    Log2 of cpm of each gene
    :param cpm_counts_df: Dataframe with CPM counts
    :return: A dataframe that each cell is the log2(value) of the cell
    """
    result_df = pd.DataFrame(np.log2(cpm_counts_df, out=np.zeros_like(cpm_counts_df), where=(cpm_counts_df != 0)),
                             index=cpm_counts_df.index, columns=cpm_counts_df.columns)

    return result_df


def test_target_survival_association(df_test_variables: DataFrame, df_clinical: DataFrame, output_dir: str):
    """
    Test specific target group in df_test_variables to survival using cox model.
    :param df_test_variables: Dataframe with group target column
    :param df_clinical: Clinical features, include event and time.
    :param output_dir: Output directory for results
    :return: Dataframe results for the cox analysis
    """
    df_results_significant = run_cox_analysis(df_test_variables, df_clinical, 1)
    output_file = os.path.join(*[output_dir, "Significance-Target-Groups.csv"])
    df_results_significant.to_csv(output_file)
    return df_results_significant

############### Main ######################


def prepare_output_dir_for_task_i(output_dir_base: str, i: int):
    """
    Creates output directory and returns that path to the new directory
    :param output_dir_base: Parent folder of the directory
    :param i: Number of task
    :return: The new directory created path
    """
    output_dir = os.path.join(*[output_dir_base, f"Task{i}"])
    Path(output_dir).mkdir(exist_ok=True)
    return output_dir


def main():
    input_dir = 'Data'
    output_dir_base = 'Output'
    Path(output_dir_base).mkdir(exist_ok=True)  # Create directory for output
    num_of_top_genes = 1000

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Task 1
        output_dir = prepare_output_dir_for_task_i(output_dir_base, 1)
        print("Running task 1:", end="\n\n")
        df_mrna = read_rna_matrix(input_dir, output_dir)

        df_mrna = normalize_cpm(df_mrna.transpose()).transpose()  # Normalize by CPM calculation
        df_mrna = clean_data(df_mrna)
        df_mrna = filter_lowly_expressed_genes(df_mrna.transpose(), 1, 50).transpose()  # Filter lowly expressed genes - only keep threshold above 1 in at least 50% of the samples
        df_mrna = log_cpm(df_mrna)
        top_variance_genes = get_top_k_variance_genes(df_mrna, output_dir, k=num_of_top_genes)
        df_clinical = read_clinical_data(input_dir)
        df_mrna_top_variance_genes = df_mrna.filter(items=top_variance_genes, axis='index')
        test_expression_survival_association(df_mrna_top_variance_genes, df_clinical, output_dir)

        # Task 2
        output_dir = prepare_output_dir_for_task_i(output_dir_base, 2)
        print("\n\nRunning task 2:", end="\n\n")
        test_mystery_distribution(output_dir)

        # Task 3
        output_dir = prepare_output_dir_for_task_i(output_dir_base, 3)
        print("\n\nRunning task 3:", end="\n\n")
        df_genes, df_clinical = load_nsclc_patients(input_dir)
        df_genes = df_genes.transpose()
        df_genes = log_cpm(normalize_cpm(df_genes))
        pca_df = plot_pca(df_genes, output_dir, "NSCLC")
        test_target_survival_association(pd.DataFrame(pca_df['target']), df_clinical, output_dir)


if __name__ == '__main__':
    main()
