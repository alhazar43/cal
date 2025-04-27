import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mirt import JointMIRT

def riasec_items(filename='data.csv', n_dims=2, n_epochs=1000, lr=0.01):
    df = pd.read_csv(filename, low_memory=False)
    riasec_df = df.iloc[:, :48]
    tipi_df = df.iloc[:, 51:61]
    vcl_df = df.iloc[:, 61:77]
    
    riasec_df.replace(0,1,inplace=True)
    tipi_df.replace(0,1,inplace=True)

    n_students = riasec_df.shape[0]
    n_riasec_items = riasec_df.shape[1]
    n_tipi_items = tipi_df.shape[1]
    n_vcl_items = vcl_df.shape[1]
    n_riasec_categories = 5
    n_tipi_categories = 7
    
    combined_responses = [riasec_df.values-1, tipi_df.values-1, vcl_df.values]
    
    jmirt = JointMIRT(n_students, n_dims=n_dims)
    jmirt.add_model("GPCM", n_riasec_items, n_riasec_categories)
    jmirt.add_model("GPCM", n_tipi_items, n_tipi_categories)
    jmirt.add_model("2PL", n_vcl_items)
    
    jmirt.fit(combined_responses, n_epochs=n_epochs, lr=lr)
    
    riasec_a = jmirt.item_models[0].a.detach().numpy()
    riasec_b = jmirt.item_models[0].b.detach().numpy()
    tipi_a = jmirt.item_models[1].a.detach().numpy()
    tipi_b = jmirt.item_models[1].b.detach().numpy()
    vcl_a = jmirt.item_models[2].a.detach().numpy()
    vcl_b = jmirt.item_models[2].b.detach().numpy()
    
    riasec_columns = [f"a{i+1}" for i in range(n_dims)] + [f"b{j+1}" for j in range(n_riasec_categories-1)]
    tipi_columns = [f"a{i+1}" for i in range(n_dims)] + [f"b{j+1}" for j in range(n_tipi_categories-1)]
    vcl_columns = [f"a{i+1}" for i in range(n_dims)] + ["b"]
    
    riasec_prarms = pd.DataFrame(np.concatenate([riasec_a, riasec_b], axis=1), columns=riasec_columns)
    tipi_prarms = pd.DataFrame(np.concatenate([tipi_a, tipi_b], axis=1), columns=tipi_columns)
    vcl_prarms = pd.DataFrame(np.column_stack([vcl_a, vcl_b]), columns=vcl_columns)

    vcl_prarms.to_csv('vcl_params_{}d.csv'.format(n_dims), index=False)
    tipi_prarms.to_csv('tipi_params_{}d.csv'.format(n_dims), index=False)
    riasec_prarms.to_csv('riasec_params_{}d.csv'.format(n_dims), index=False)
    
    riasec_name = [cat+str(num) for cat in ['R', 'I', 'A', 'S', 'E', 'C'] for num in range(1, 9)]
    combined = pd.DataFrame({
        'item_id': list(range(74)),
        'item_type': ['likert']*58+['binary']*16,
        'a': list(riasec_a)+list(tipi_a)+list(vcl_a),
        'b': list(riasec_b)+list(tipi_b)+list(vcl_b),
        'item_name': riasec_name+['TIPI'+str(i) for i in range(1, 11)]+['VCL'+str(i) for i in range(1, 17)]
    })
    combined.to_csv('combined_params_{}d.csv'.format(n_dims), index=False)
    

    
def big5_items(filename='big5.csv', n_dims=2, n_epochs=1000, lr=0.01):
    df = pd.read_csv(filename, low_memory=False)
    big5_df = df.iloc[:, 7:]
    
    big5_df.replace(0,1,inplace=True)

    n_students = big5_df.shape[0]
    n_big5_items = big5_df.shape[1]
    n_big5_categories = 5

    
    combined_responses = [big5_df.values-1]
    
    jmirt = JointMIRT(n_students, n_dims=n_dims)
    jmirt.add_model("GPCM", n_big5_items, 5)

    
    jmirt.fit(combined_responses, n_epochs=n_epochs, lr=lr)
    
    big5_a = jmirt.item_models[0].a.detach().numpy()
    big5_b = jmirt.item_models[0].b.detach().numpy()

    
    big5_columns = [f"a{i+1}" for i in range(n_dims)] + [f"b{j+1}" for j in range(n_big5_categories-1)]

    big5_prarms = pd.DataFrame(np.concatenate([big5_a, big5_b], axis=1), columns=big5_columns)


    big5_prarms.to_csv('big5_params_{}d.csv'.format(n_dims), index=False)
    
    
if __name__ == '__main__':
    # riasec_items(n_epochs=500, n_dims=2, lr=0.01)
    big5_items(n_epochs=500, n_dims=2, lr=0.01)