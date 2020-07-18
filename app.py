import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanv
from matplotlib.figure import Figure
from flask import Flask,render_template,request,Response
import requests
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from io import BytesIO
import base64

'''data = pd.read_csv('Iris.csv')
plt.scatter(data.SepalLengthCm,data.SepalWidthCm,cmap='winter',s=100)
plt.show()'''
X,Y,N = [[]],[],50
def generate_data():
    global X,Y,N
    N = np.random.randint(50,150)
    X,Y = make_blobs(n_samples=N,centers=2,random_state=0,cluster_std=0.60)
    plt.cla()
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='winter',s=100,edgecolors='black')
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.title("DATA POINTS")
    gendat = BytesIO()
    plt.savefig(gendat,format='png')
    # figfile.seek(0)     # Rewind to the beginning of the file
    gendat_img = base64.b64encode(gendat.getbuffer()).decode('ascii')
    return gendat_img
    
# Model
'''svm = SVC(kernel='linear',C=1)
svm.fit(X,Y)'''
def plot_svm_boundary(ax=None,plot_support=True,C_inp=1):
    global X,Y
    model = SVC(kernel='linear',C=C_inp)
    model.fit(X,Y)
    if ax is None:
        ax = plt.gca()
    # Set limits for x and y data. Here, it will return [X.min(),X.max()] and [Y.min(),Y.max()]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Creating a mesh grid to evaluate model
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    YY,XX = np.meshgrid(y,x)
    test_data = np.vstack([XX.ravel(),YY.ravel()]).T
    z = model.decision_function(test_data).reshape(XX.shape)

    # Plotting
    ax.contour(XX,YY,z,levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'],colors=['black','red','black'])
    # Plotting the support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=500,linewidth=1,facecolors='None')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def final_plot(C_i):
    global X,Y
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='winter',s=100,edgecolors='black')
    plot_svm_boundary(C_inp=C_i)
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.title("VISUALIZING THE SVM DECISION BOUNDARY")
    # plt.show()
    figfile = BytesIO()
    plt.savefig(figfile,format='png')
    # figfile.seek(0)     # Rewind to the beginning of the file
    figdata_img = base64.b64encode(figfile.getbuffer()).decode('ascii')
    return figdata_img

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_data',methods=['POST'])
def generate():
    gen_img = generate_data()
    return render_template('generate.html',gen_img=gen_img)

@app.route('/show',methods=['POST'])
def show():
    c = request.form.get('c')
    if c == "1E-4":
        c = 0.0001
    elif c == "1E-2":
        c = 0.01
    elif c == "1E-1":
        c = 0.1
    else:
        c = int(c)
    img = final_plot(c)
    return render_template('show.html',img = img)


if __name__ == "__main__":
    app.run(debug=True)
