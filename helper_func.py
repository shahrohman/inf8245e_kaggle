
from matplotlib.patches import Ellipse
import numpy as np
from matplotlib.colors import to_rgba

def calcModeleGaussien(data):
    
    moyenne = np.mean(data,axis=0)
    matr_cov = np.cov(data,rowvar=False)
    val_propres, vect_propres = np.linalg.eig(matr_cov)

    return moyenne, matr_cov, val_propres, vect_propres


def viewEllipse(data, ax, scale=1, facecolor='none', edgecolor='red', **kwargs):
    
    moy, cov, lambdas, vectors = calcModeleGaussien(data)
    
    order =np.argsort(lambdas)
    lambdas=np.sqrt(lambdas[order])
    vectors=vectors[:,order]
    selected_v=vectors[:,1]
    angle =np.arctan2(selected_v[1], selected_v[0]) *180/np.pi

    ellipse = Ellipse(moy, width=2*scale*lambdas[1], height=2*scale*lambdas[0],
                      angle= angle, facecolor=facecolor,
                      edgecolor=to_rgba(edgecolor, 1), linewidth=3, **kwargs)
    return ax.add_patch(ellipse)