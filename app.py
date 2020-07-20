import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigCanv
from matplotlib.figure import Figure
from flask import Flask,render_template,request,Response
import requests
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs,make_circles
from sklearn.svm import SVC
from io import BytesIO
import base64
import pickle as pkl

'''data = pd.read_csv('Iris.csv')
plt.scatter(data.SepalLengthCm,data.SepalWidthCm,cmap='winter',s=100)
plt.show()'''
X,Y,N = [[]],[],50
def generate_data(choice='linear'):
    global X,Y,N
    if choice == 'linear':
        N = np.random.randint(50,150)
        std = (np.random.randint(6,10))*0.1
        X,Y = make_blobs(n_samples=N,centers=2,random_state=0,cluster_std=0.60)
    elif choice == 'circle':
        N = np.random.randint(100,200)
        factor = (np.random.randint(1,6))*0.1
        X,Y = make_circles(n_samples=N,factor=factor,noise=0.1)
    plt.delaxes()
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
def model_ser(C_in=1,kernel_in='linear',degree_in=0):
    global X,Y
    if degree_in == 0:
        m = SVC(kernel=kernel_in,C=C_in)
    else:
        m = SVC(kernel=kernel_in,C=C_in,degree=degree_in)
    m.fit(X,Y)
    pkl.dump(m,open('model.pkl','wb'))

    return pkl.load(open('model.pkl','rb'))

def plot_svm_boundary(ax=None,plot_support=True,C_inp=1,k_in='linear',deg_in=0):
    if deg_in == 0: 
        model = model_ser(C_in=C_inp,kernel_in=k_in)
    else:
        model = model_ser(C_in=C_inp,kernel_in=k_in,degree_in=deg_in)
    if ax is None:
        ax = plt.gca()
    # Set limits for x and y data. Here, it will return [X.min(),X.max()] and [Y.min(),Y.max()]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Creating a mesh grid to evaluate model
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    YY,XX = np.meshgrid(y,x)
    XX_C,YY_C = np.meshgrid((np.arange(start=X[:,0].min()-1,stop=X[:,0].max()+1,step=0.01)),(np.arange(start=X[:,1].min()-1,stop=X[:,1].max()+1,step=0.01)))
    test_data,test = np.vstack([XX.ravel(),YY.ravel()]).T,np.array([XX_C.ravel(),YY_C.ravel()]).T
    z,Z = model.decision_function(test_data).reshape(XX.shape),(model.predict(test)).reshape(XX_C.shape)

    # Plotting
    ax.contour(XX,YY,z,levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'],colors=['black','red','black'])
    plt.contourf(XX_C,YY_C,Z,alpha=0.21,cmap='winter')
    # Plotting the support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=500,linewidth=1,facecolors='None')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def final_plot(C_i,k_i,d_i=0):
    global X,Y
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=Y,cmap='winter',s=100,edgecolors='black')
    plot_svm_boundary(C_inp=C_i,k_in=k_i,deg_in=d_i)
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.title("VISUALIZING THE SVM DECISION BOUNDARY")
    # plt.show()
    figfile = BytesIO()
    plt.savefig(figfile,format='png')
    # figfile.seek(0)     # Rewind to the beginning of the file
    figdata_img = base64.b64encode(figfile.getbuffer()).decode('ascii')
    return figdata_img

def plot_gauss3D():
    global X,Y
    projection = np.exp(-(X**2).sum(1))
    plt.figure()
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:,0],X[:,1],projection,c=Y,cmap='winter',s=100,edgecolors='black')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Projection")
    ax.set_title("3D VIEW OF DATA POINTS PROJECTED IN GAUSSIAN RBF KERNEL")
    figfile3D = BytesIO()
    plt.savefig(figfile3D,format='png')
    figdata_img3D = base64.b64encode(figfile3D.getbuffer()).decode('ascii')
    return figdata_img3D

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_data',methods=['POST','GET'])
def generate():
    choice = request.form.get('type')
    if choice == 'linear':
        gen_img = generate_data(choice='linear')
        return render_template('generate.html',gen_img=gen_img)

    else:
        gen_img = generate_data(choice='circle')
        return render_template('generate-nonlin.html',gen_img=gen_img)

@app.route('/disp3d',methods=['GET'])
def disp3d():
    img3d = plot_gauss3D()
    return render_template('3D.html',img3d=img3d)

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
    k = request.form.get('kernel')
    if k == 'poly2':
        img = final_plot(c,'poly',2)
    elif k == 'poly3':
        img = final_plot(c,'poly',3)
    elif k == 'poly4':
        img = final_plot(c,'poly',4)
    elif k == 'poly5':
        img = final_plot(c,'poly',5)
    else:
        img = final_plot(c,k)    
    return render_template('show.html',img = img)


if __name__ == "__main__":
    app.run(debug=True)
