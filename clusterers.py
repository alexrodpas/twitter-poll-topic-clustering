import numpy as np
from abc import abstractmethod

class Clusterer:
    def __init__(self):
        self.clusterer = None
        self.n_clusters = None
        self.cluster_assignments = None
        self.cluster_centers = None

    @abstractmethod
    def fit(self):
        pass

    def cluster_analysis(self, df, cluster_number, n = 10):
        df['cluster'] = self.cluster_assignments
        df_cluster = df[df['cluster'] == cluster_number]

        print(f"Cluster Number {cluster_number} - ")
        print(f"Elements in Cluster: {len(df_cluster)}")
        print(f"Average Words in the Poll: {df_cluster['text'].str.split().apply(len).mean()}")
        
        texts = list(df_cluster.sample(n)['text'])

        print("\nSamples - ")
        for i, text in enumerate(texts): 
            print(f"{i+1}. {text}")

        print()

        return texts
    
    def summarise_clusters(self, df, n = 7):
        labels = list(set(self.cluster_assignments))
        for cluster_number in labels:
            self.cluster_analysis(df, cluster_number, n)

class KMeansClusterer(Clusterer):
    def __init__(self, k = 25):
        super().__init__()

        from sklearn.cluster import KMeans
        self.clusterer = KMeans(n_clusters = k, random_state=0)
        self.n_clusters = k

    def fit(self, embeddings):
        self.clusterer.fit(embeddings)
        self.cluster_assignments = self.clusterer.labels_
        self.cluster_centers = self.clusterer.cluster_centers_

    def save(self, df, path):
        df['cluster'] = self.cluster_assignments
        for i in range(self.n_clusters):
            with open(path + '/output{}.txt'.format(i), 'w') as file:
                df2 = df[df['cluster'] == i]
                for j in df2['text']:
                    file.write('{}\n'.format(j))

class AgglomerativeClusterer(Clusterer):
    def __init__(self, n_clusters = 25):
        super().__init__()

        from sklearn.cluster import AgglomerativeClustering
        self.clusterer = AgglomerativeClustering(n_clusters = 25, compute_distances = True)
        self.n_clusters = n_clusters

    def fit(self, embeddings):
        self.clusterer.fit(embeddings)
        self.cluster_assignments = self.clusterer.labels_

    def plot_dendrogram(self, **kwargs):
        from scipy.cluster.hierarchy import dendrogram
        # Create linkage matrix and then plot the dendrogram
        model = self.clusterer
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, orientation = 'left', **kwargs)

class HDBSCANClusterer(Clusterer):
    def __init__(self, min_cluster_size = 500, min_samples = 10):
        super().__init__()

        import hdbscan
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples)

    def fit(self, embeddings):
        self.clusterer.fit(embeddings)
        self.cluster_assignments = self.clusterer.labels_

    def plot_dendrogram(self):
        self.clusterer.condensed_tree_.plot(select_clusters=True, label_clusters = True)
