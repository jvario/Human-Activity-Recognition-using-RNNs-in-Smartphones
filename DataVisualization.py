import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import rgb2hex
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

#https://github.com/aargun/Human-Activity-Recognition/blob/master/human_activity_recognition_smartphone-checkpoint.ipynb
#APORIA: xrisi dataset aytou gia train h mono gia visualising

if __name__ == "__main__":


    data_ = pd.read_csv('E:/Documents/Thesis/har1/har-smartphone/Data/Human_activity_recognition.csv', encoding='utf-8')
    print(data_.head())
    data_['activity'].value_counts().plot(kind='bar')
    plt.show()
    plt.savefig('Results/Visualisation/ActivityBar.png')



    plt.figure(figsize=(15,10))



    # Create datasets
    trans_data = data_.copy()
    activity_data = trans_data.pop('activity')
    print(f'Feature Data Shape: {trans_data.shape}')
    print(f'Activity Label Data Shape: {activity_data.shape}')

    # Scale Data
    scl = StandardScaler()
    trans_data = scl.fit_transform(trans_data)



    activity_count = activity_data.value_counts()

    ### Plot Activities
    plt.figure(figsize=(15,10))

    # Get colors
    n = activity_data.unique().shape[0]
    colormap = get_cmap('viridis')
    colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]
    rng = np.random.RandomState(0)

    ## Manifold Learing
    # We choose TSNE(t-distributed stochastic neibhour embedding) as the main purpose of t-SNE is visualization of high-dimensional Data. Hence, it works best when the Data will be embedded on two or three dimensions.
    tsne = TSNE(random_state=0)
    tsne_transformed = tsne.fit_transform(trans_data)
    print(f'Feature Data Shape(After Transformation): {tsne_transformed.shape}')
    plt.figure(figsize=(15,10))
    for i, activity in enumerate(activity_count.index):
        # Select all rows with current activity
        mask = (activity_data==activity).values
        plt.scatter(x=tsne_transformed[mask][:,0], y=tsne_transformed[mask][:,1], c=colors[i], alpha=0.5, label=activity)
    plt.title('TSNE: Activity Visualisation')
    plt.legend()
    plt.savefig('Results/Visualisation/TSNE-Activity Visualisation.png')
    plt.show()
