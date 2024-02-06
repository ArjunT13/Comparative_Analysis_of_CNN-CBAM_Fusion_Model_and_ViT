import pandas as pd
import pickle
import numpy as np

def demo_process(df, column_name):
    df = df.drop(['primcauseofdeath','causeofdeath'], axis=1)
    k1 = df.shape[0]
    df = df[df[column_name]!= -1]
    k2 = df.shape[0] 
    print("Number of Rows containing NaN values: ", k1-k2)
    return df

def load_EHR_data(demodf, column_name):

    features = list(pd.read_csv('/mnt/storage/arjun_postpci_prognosis/InputFeatures.csv')['Features'])
    demo = []
    image_name = []
    # labels = []
    demodf = demo_process(demodf, column_name)
    demodf2 = demodf.drop(['PRE','POST', column_name],axis=1)
    with open('/mnt/storage/arjun_postpci_prognosis/image_name_info.pkl', 'rb') as f:
        image_info = pickle.load(f)
    image_info = pd.DataFrame(image_info)
    print(image_info)
    for i in range(0,image_info.shape[0]):
        #Check if Image exists
        if str(image_info['Image Name'].iloc[i]).split('.')[1] == 'PNG':
            if demodf.loc[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']].shape[0] > 0 and demodf.loc[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']].shape[0]>0:
                image_name.append( image_info['Image Name'].iloc[i].replace('.PNG',''))
                # labels.append(demodf.loc[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']][column_name].iloc[0])
                temp = demodf[demodf['POST']+'.PNG' == image_info.iloc[i]['Image Name']][features].iloc[0].values
                demo.append(temp)             
        elif str(image_info['Image Name'].iloc[i]).split('.')[1] == 'png':
            if demodf.loc[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']].shape[0] > 0 and demodf.loc[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']].shape[0]>0:
                image_name.append( image_info['Image Name'].iloc[i].replace('.png',''))
                temp = demodf[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']][features].iloc[0].values
                # labels.append(demodf.loc[demodf['POST']+'.png' == image_info.iloc[i]['Image Name']][column_name].iloc[0])
                demo.append(temp)
    demo = np.array(demo).astype('float32')
    # labels = np.array(labels)
    return demo, image_name #, labels