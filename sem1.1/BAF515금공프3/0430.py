# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
np.random.seed(123)
MAT = np.random.exponential( scale=1/5, size=(10, 7))
MAT = np.around( MAT, decimals=3)
MAT

# %%
Q1 = np.percentile( MAT, q=25)
Q3 = np.percentile( MAT, q=75)
MAT [ ( MAT < Q1- 1.5*(Q3-Q1) ) | ( MAT > Q3 + 1.5*(Q3-Q1) ) ] 

# %%

# %%
m = np.median( MAT )
mdiff = np.abs ( MAT - m )
np.nonzero ( mdiff == np.min ( mdiff ) )

# %%
MAT[ np.argsort(MAT[:, 1]), : ]

# %%
MAT  [  :  , np.argsort( np.mean( MAT, axis = 0 ) ) [::-1]  ]

# %%
MAT [ ( MAT < Q1- 1.5*(Q3-Q1) ) | ( MAT > Q3 + 1.5*(Q3-Q1) ) ] = np.nan
MAT

# %%
MAT[ np.sum ( np.isnan( MAT ) ,  axis = 1 )> 0 , : ]

# %%
np.random.seed(123)
aMAT = np.random.randint ( 0, 100, (6, 10))
aVec = np.where( np.mean (aMAT , axis=0 ) > 50 , 1, 0 )
aVec

# %%
aMAT[  ( aMAT [:, 4] > 50 ) | ( aMAT [:, 6] < 30 ), : ]

# %%
aMAT [ [3, 1] , [1, 6] ] = 999
aMAT

# %%
aMAT [ ~ np.any( aMAT == 999, axis = 1 ), : ].mean(axis=0)

# %%
aMAT [ ~ np.any( aMAT == 999, axis = 1 ), : ].std(axis=0)
