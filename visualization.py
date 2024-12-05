import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# In[]
# Depth comparison
df1 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_4-width_512/fixed_point_evaluations.csv')

df2 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_5-width_512/fixed_point_evaluations.csv')

df3 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_6-width_512/fixed_point_evaluations.csv')

df4 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')

df5 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_8-width_512/fixed_point_evaluations.csv')

df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
df.rename(columns = {'layer_num' : 'hidden layer'}, inplace=True)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="singular_0", y="max_noise_tolerance", height=4, hue="hidden layer", palette="flare")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.3)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/singular_value_and_noise_depth.jpg", dpi=1000)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="eig_norm_0", y="max_noise_tolerance", height=4, hue="hidden layer", palette="crest")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\lambda_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/eig_norm_and_noise_depth.jpg", dpi=1000)

# g = sns.displot(df, x="eig_norm_0", hue="fixed_point",
#             col="layer_num", kde=True, height=3, col_wrap=3, hue_order=[True, False], stat="density")
# g.set(xlabel= "maximum norm of eigenvalues", ylabel="density")
# g.savefig("figures/eigenvalue_norm_and_layer_num_width_512.jpg", dpi=600)
# g.set(xlim=(0.0, None))
# In[多元线性回归与复分析]
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import numpy as np
import joypy
import matplotlib.pyplot as plt

df = df[(df["attractors"]) & (df["fixed_point"])]
s_total = df.iloc[:, 209]
s = df.iloc[:, 210:410]
s_prop = pd.DataFrame()

for i in range(10):
    s_prop_sum = np.sum(s.iloc[:,:i+1], axis=1)
    prop = s_prop_sum/s_total
    s_prop['singular_sum_{}'.format(i)] = prop
    
joypy.joyplot(s_prop)

y = df["max_noise_tolerance"]
singular_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = s.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   singular_coefs.append(coef)
   
singular_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = s.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   singular_coefs_mlp.append(coef)

e_total = df.iloc[:, 6]
e = df.iloc[:, 8:207]
e_prop = pd.DataFrame()
for i in range(10):
    e_prop_sum = np.sum(e.iloc[:,:i+1], axis=1)
    prop = e_prop_sum/e_total
    e_prop['eigenvalue_norm_sum_{}'.format(i)] = prop

joypy.joyplot(e_prop)

eigenvalue_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs.append(coef)
   
eigenvalue_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs_mlp.append(coef)

coefs = pd.DataFrame()
coefs['singular value'] = singular_coefs
coefs['eigenvalue norm'] = eigenvalue_coefs
coefs['singular value mlp'] = singular_coefs_mlp
coefs['eigenvalue norm mlp'] = eigenvalue_coefs_mlp

ax = plt.figure(figsize=(6,5))
plt.plot(range(0,20), coefs['singular value'], 'g-v')
plt.plot(range(0,20), coefs['singular value mlp'], 'g-h')
plt.plot(range(0,20), coefs['eigenvalue norm'], 'r-*')
plt.plot(range(0,20), coefs['eigenvalue norm mlp'], 'r-x')
plt.xlabel(r'Numbers of encountered $\sigma_{i}(J_{*})$ and $\lambda_{i}(J_{*})$', 
           fontdict={'size':14})
plt.ylabel('Multi-correlation coefficient', fontdict={'size':14})
plt.xticks([0, 5, 10, 15, 20], fontsize=11)
plt.yticks(np.arange(0,1.01, 0.2), fontsize=11)

plt.legend([r'$\sigma_{i}(J_{*})$+Linear',
            r'$\sigma_{i}(J_{*})$+MLP',
            r'$\lambda_{i}(J_{*})$+Linear',
            r'$\lambda_{i}(J_{*})$+MLP'], prop = { "size": 11})
plt.xlim(0, )
plt.ylim(0, 1)
plt.grid()
plt.savefig('figures/multi-coefficient-depths.jpg', dpi=600)

# In[]
# df6 = pd.read_csv(
#     'logs/Minist/figure_400-noise_0.0-layer_6-width_384/fixed_point_evaluations.csv')

df7 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_6-width_512/fixed_point_evaluations.csv')

df8 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_6-width_768/fixed_point_evaluations.csv')

df9 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_6-width_1024/fixed_point_evaluations.csv')

df = pd.concat([df7, df8, df9], ignore_index=True)

g = sns.JointGrid(  
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="singular_0", y="max_noise_tolerance", height=4, hue="hidden_size", palette="crest")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.3)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/singular_value_and_noise_width.jpg", dpi=1000)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="eig_norm_0", y="max_noise_tolerance", height=4, hue="hidden_size", palette="flare")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\lambda_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/eig_norm_and_noise_width.jpg", dpi=1000)



df = pd.concat([df7, df8, df9], ignore_index=True)
df['mean_singular_8'] = df.iloc[:, 210:218].mean(axis=1)
g = sns.JointGrid(  
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="mean_singular_8", y="max_noise_tolerance", height=4, hue="hidden_size", palette="crest")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.3)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/mean_singular_value_and_noise_width.jpg", dpi=1000)

df['mean_eig_norm_8'] = df.iloc[:, 8:16].mean(axis=1)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="mean_eig_norm_8", y="max_noise_tolerance", height=4, hue="hidden_size", palette="flare")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\lambda_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/mean_eig_norm_and_noise_width.jpg", dpi=1000)

# sns.displot(df, x="singular_1", hue="hidden_size", kind="kde", fill=False, palette="crest", height=4)
# sns.displot(df, x="singular_3", hue="hidden_size", kind="kde", fill=False, palette="crest", height=4)

# In[]
df = df[(df["attractors"]) & (df["fixed_point"])]
s_total = df.iloc[:, 209]
s = df.iloc[:, 210:410]


si_list = []
for i in range(6):
    si = pd.DataFrame()
    si["value"] = s.iloc[:,i].tolist()
    si['hidden_neuron'] = df['hidden_size'].tolist()
    si['i'] = [i]*len(si)
    si_list.append(si)
    
s_0 = pd.concat(si_list, axis=0, ignore_index = True)

joypy.joyplot(s_0, by="i", )

# g = sns.FacetGrid(s_0, col="hidden_size")
# g.map_dataframe(sns.histplot, x="value", hue="i", palette="deep", kde=True)

s_prop = pd.DataFrame()
for i in range(8):
    s_prop_sum = np.sum(s.iloc[:,:i+1], axis=1)
    prop = s_prop_sum/s_total
    s_prop['count {}'.format(i+1)] = prop
    
joypy.joyplot(
    s_prop, fade=True, figsize=(5,4),
    linewidth=0.6, xlabels=True)

plt.savefig('figures/singular_value_sum.jpg', dpi=600)

y = df["max_noise_tolerance"]
singular_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = s.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   singular_coefs.append(coef)
   
mean_singular_coefs = []
for i in range(20):
   x = s.iloc[:,:i+1].sum(axis=1)
   coef = np.corrcoef(x, y)[0,1]
   mean_singular_coefs.append(coef)
plt.plot(mean_singular_coefs)
   
singular_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = s.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   singular_coefs_mlp.append(coef)

e_total = df.iloc[:, 6]
e = df.iloc[:, 8:207]
e_prop = pd.DataFrame()
for i in range(20):
    e_prop_sum = np.sum(e.iloc[:,:i+1], axis=1)
    prop = e_prop_sum/e_total
    e_prop['eigenvalue_norm_sum_{}'.format(i)] = prop

joypy.joyplot(e_prop)

eigenvalue_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs.append(coef)
   
mean_eigenvalue_coefs = []
for i in range(20):
   x = e.iloc[:,:i+1].sum(axis=1)
   coef = np.corrcoef(x, y)[0,1]
   mean_eigenvalue_coefs.append(coef)
plt.plot(mean_eigenvalue_coefs)
   
eigenvalue_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs_mlp.append(coef)

coefs = pd.DataFrame()
coefs['singular value'] = singular_coefs
coefs['eigenvalue norm'] = eigenvalue_coefs
coefs['singular value mlp'] = singular_coefs_mlp
coefs['eigenvalue norm mlp'] = eigenvalue_coefs_mlp

ax = plt.figure(figsize=(7.0,5.5))
plt.plot(range(1,21), coefs['singular value'], 'g-v')
plt.plot(range(1,21), coefs['eigenvalue norm'], 'r-*')
plt.xlabel(r'Numbers of counted $\sigma_{i}(J_{*})$ and $\lambda_{i}(J_{*})$', 
           fontdict={'size':16})
plt.ylabel('MCC', fontdict={'size':16})
plt.xticks([0, 5, 10, 15, 20], fontsize=14)
plt.yticks(np.arange(0,1.01, 0.2), fontsize=14)

plt.legend([r'$\sigma_{i}(J_{*})$',
            # r'$\sigma_{i}(J_{*})$+MLP',
            r'$\lambda_{i}(J_{*})$',
            # r'$\lambda_{i}(J_{*})$+MLP'
            ],
           prop = { "size": 14}, loc='lower right')
plt.xlim(0, )
plt.ylim(0.4, 0.9)
plt.grid()
plt.savefig('figures/multi-coefficient-width.jpg', dpi=600)

ax = plt.figure(figsize=(7.0,5.5))
plt.plot(range(1,21), mean_singular_coefs, 'g-v')
plt.plot(range(1,21), mean_eigenvalue_coefs, 'r-*')
# plt.plot(range(0,20), coefs['eigenvalue norm mlp'], 'r-x')
plt.xlabel(r'Numbers of counted $\sigma_{i}(J_{*})$ and $\lambda_{i}(J_{*})$', 
           fontdict={'size':16})
plt.ylabel('PCC', fontdict={'size':16})
plt.xticks([0, 5, 10, 15, 20], fontsize=14)
plt.yticks(np.arange(-1.0, 0.01, 0.2), fontsize=14)

plt.legend([r'$\sigma_{i}(J_{*})$',
            # r'$\sigma_{i}(J_{*})$+MLP',
            r'$\lambda_{i}(J_{*})$',
            # r'$\lambda_{i}(J_{*})$+MLP'
            ],
           prop = { "size": 14}, loc='lower right')
plt.xlim(0, )
plt.ylim(-0.9, -0.4)
plt.grid()
plt.savefig('figures/mean-coefficient-width.jpg', dpi=600)

# In[]
df1 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.1-layer_6-width_768/fixed_point_evaluations.csv')

df2 = pd.read_csv(
    'logs/Minist/figure_600-noise_0.1-layer_6-width_768/fixed_point_evaluations.csv')

df3 = pd.read_csv(
    'logs/Minist/figure_800-noise_0.1-layer_6-width_768/fixed_point_evaluations.csv')

df4 = pd.read_csv(
    'logs/Minist/figure_1000-noise_0.1-layer_6-width_768/fixed_point_evaluations.csv')

df5 = pd.read_csv(
    'logs/Minist/figure_1200-noise_0.1-layer_6-width_768/fixed_point_evaluations.csv')

df6 = pd.read_csv(
    'logs/Minist/figure_1400-noise_0.1-layer_6-width_768/fixed_point_evaluations.csv')

df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)

df = df[(df["attractors"]) & (df["fixed_point"])]

df['mean_singular_8'] = df.iloc[:,210:218].mean(1)
df['mean_eig_norm_8'] = df.iloc[:,8:17].mean(1)

g = sns.JointGrid(
    data=df,
    x="mean_singular_8", y="max_noise_tolerance", height=4, palette="flare")
g.ax_joint.set(ylim=(0.0, 2.0))
g.ax_joint.set(xlim=(0.0, 0.1))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.3)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")


g = sns.JointGrid(
    data=df,
    x="mean_eig_norm_8", y="max_noise_tolerance", height=4, palette="flare")
g.ax_joint.set(ylim=(0.0, 2.0))
g.ax_joint.set(xlim=(0.0, 0.1))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.3)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\lambda_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")

# In[]
df = pd.concat([df1, df2, df3, df4, df5, df6, df8, df9], ignore_index=True)

g = sns.JointGrid(  
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="singular_0", y="max_noise_tolerance", height=4, palette="crest")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=8, alpha=1.0, color="tab:cyan")
# g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.3)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/singular_value_and_noise.jpg", dpi=1000)

df = df[(df["attractors"]) & (df["fixed_point"])]
s_total = df.iloc[:, 209]
s = df.iloc[:, 210:410]
s_prop = pd.DataFrame()

for i in range(10):
    s_prop_sum = np.sum(s.iloc[:,:i+1], axis=1)
    prop = s_prop_sum/s_total
    s_prop['singular_sum_{}'.format(i)] = prop
    
joypy.joyplot(s_prop)

y = df["max_noise_tolerance"]
singular_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = s.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   singular_coefs.append(coef)
   
singular_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = s.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   singular_coefs_mlp.append(coef)

e_total = df.iloc[:, 6]
e = df.iloc[:, 8:207]
e_prop = pd.DataFrame()
for i in range(10):
    e_prop_sum = np.sum(e.iloc[:,:i+1], axis=1)
    prop = e_prop_sum/e_total
    e_prop['eigenvalue_norm_sum_{}'.format(i)] = prop

joypy.joyplot(e_prop)

eigenvalue_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs.append(coef)
   
eigenvalue_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs_mlp.append(coef)

coefs = pd.DataFrame()
coefs['singular value'] = singular_coefs
coefs['eigenvalue norm'] = eigenvalue_coefs
coefs['singular value mlp'] = singular_coefs_mlp
coefs['eigenvalue norm mlp'] = eigenvalue_coefs_mlp

ax = plt.figure(figsize=(6,5))
plt.plot(range(0,20), coefs['singular value'], 'g-v')
plt.plot(range(0,20), coefs['singular value mlp'], 'g-h')
plt.plot(range(0,20), coefs['eigenvalue norm'], 'r-*')
plt.plot(range(0,20), coefs['eigenvalue norm mlp'], 'r-x')
plt.xlabel(r'Numbers of encountered $\sigma_{i}(J_{*})$ and $\lambda_{i}(J_{*})$', 
           fontdict={'size':14})
plt.ylabel('Multi-correlation coefficient', fontdict={'size':14})
plt.xticks([0, 5, 10, 15, 20], fontsize=11)
plt.yticks(np.arange(0,1.01, 0.2), fontsize=11)

plt.legend([r'$\sigma_{i}(J_{*})$+Linear',
            r'$\sigma_{i}(J_{*})$+MLP',
            r'$\lambda_{i}(J_{*})$+Linear',
            r'$\lambda_{i}(J_{*})$+MLP'], prop = { "size": 11}, loc='lower right')
plt.xlim(0, )
plt.ylim(0, 1)
plt.grid()
plt.savefig('figures/multi-coefficient-all.jpg', dpi=600)

# In[]
df1 = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
df1 = df1[(df1["attractors"]) & (df["fixed_point"])]
s1 = df1.iloc[:, 210:410]

df2 = pd.concat([df6, df7, df8, df9], ignore_index=True)
df2 = df2[(df2["attractors"]) & (df["fixed_point"])]
s2 = df2.iloc[:, 210:410]

y1 = df1["max_noise_tolerance"]
y2 = df2["max_noise_tolerance"]
singular_coefs = []

for i in range(20):
   reg = linear_model.LinearRegression()
   x1 = s1.iloc[:,:i+1]
   reg.fit(x1, y1)
   x2 = s2.iloc[:,:i+1]
   y_pred = reg.predict(x2)
   coef = np.corrcoef(y_pred, y2)[0,1]
   singular_coefs.append(coef)
   
singular_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x1 = s1.iloc[:,:i+1]
   reg.fit(x1, y1)
   x2 = s2.iloc[:,:i+1]
   y_pred = reg.predict(x2)
   coef = np.corrcoef(y_pred, y2)[0,1]
   singular_coefs.append(coef)

e_total = df.iloc[:, 6]
e = df.iloc[:, 8:207]
e_prop = pd.DataFrame()
for i in range(10):
    e_prop_sum = np.sum(e.iloc[:,:i+1], axis=1)
    prop = e_prop_sum/e_total
    e_prop['eigenvalue_norm_sum_{}'.format(i)] = prop

joypy.joyplot(e_prop)

eigenvalue_coefs = []
for i in range(20):
   reg = linear_model.LinearRegression()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs.append(coef)
   
eigenvalue_coefs_mlp = []
for i in range(20):
   reg = MLPRegressor()
   x = e.iloc[:,:i+1]
   reg.fit(x, y)
   y_pred = reg.predict(x)
   coef = np.corrcoef(y_pred, y)[0,1]
   eigenvalue_coefs_mlp.append(coef)

coefs = pd.DataFrame()
coefs['singular value'] = singular_coefs
coefs['eigenvalue norm'] = eigenvalue_coefs
coefs['singular value mlp'] = singular_coefs_mlp
coefs['eigenvalue norm mlp'] = eigenvalue_coefs_mlp

ax = plt.figure(figsize=(6,5))
plt.plot(range(0,20), coefs['singular value'], 'g-v')
plt.plot(range(0,20), coefs['singular value mlp'], 'g-h')
plt.plot(range(0,20), coefs['eigenvalue norm'], 'r-*')
plt.plot(range(0,20), coefs['eigenvalue norm mlp'], 'r-x')
plt.xlabel(r'Numbers of encountered $\sigma_{i}(J_{*})$ and $\lambda_{i}(J_{*})$', 
           fontdict={'size':14})
plt.ylabel('Multi-correlation coefficient', fontdict={'size':14})
plt.xticks([0, 5, 10, 15, 20], fontsize=11)
plt.yticks(np.arange(0,1.01, 0.2), fontsize=11)

plt.legend([r'$\sigma_{i}(J_{*})$+Linear',
            r'$\sigma_{i}(J_{*})$+MLP',
            r'$\lambda_{i}(J_{*})$+Linear',
            r'$\lambda_{i}(J_{*})$+MLP'], prop = { "size": 11}, loc='lower right')
plt.xlim(0, )
plt.ylim(0, 1)
plt.grid()
plt.savefig('figures/multi-coefficient-all.jpg', dpi=600)


# In[]
# Sample num comparison
df1 = pd.read_csv(
    'logs/Minist/figure_200-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df1["figure_num"] = [200]*len(df1)
df2 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df2["figure_num"] = [400]*len(df2)
df3 = pd.read_csv(
    'logs/Minist/Jul01_15-30-04-figure_600-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df3["figure_num"] = [600]*len(df3)
df4 = pd.read_csv(
    'logs/Minist/Jul01_15-30-10-figure_800-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df4["figure_num"] = [800]*len(df4)

df = pd.concat([df2], ignore_index=True)

g = sns.JointGrid(data=df[(df["attractors"]) & (df["fixed_point"])],
                  x="eig_norm_0", y="max_noise_tolerance", height=4)

g.ax_joint.set(xlim=(0.0,0.7), ylim=(0.0, 1.4))
# sns.regplot(
#     data=df[df["fixed_point"]], y="max_eig_norm",
#     x="max_noise_tolerance", ci=95, ax=g.ax_joint, order=4,
#     scatter_kws={'s':12, 'alpha':0.6})
g.plot_joint(sns.scatterplot, s=12, alpha=0.8)
g.plot_joint(sns.kdeplot, color="tab:blue", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:blue")
g.ax_joint.set(xlabel= r"$\lambda_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/eigenvalue_norm_and_noise.jpg", dpi=1000)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="singular_0", y="max_noise_tolerance", height=4)
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=12, alpha=0.8, color="tab:cyan")
g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/singule_value_and_noise.jpg", dpi=1000)

df = pd.concat([df1, df2, df3, df4], ignore_index=True)
g = sns.displot(df, x="max_eig_norm", hue="fixed_point",
            col="figure_num", kde=True, height=3, col_wrap=4, hue_order=[True, False], stat="density")
g.set(xlim=(0.0,1.2))
g.set(xlabel= "maximum norm of eigenvalues", ylabel="count")
g.savefig("figures/eigenvalue_norm_and_figure_num.jpg", dpi=600)


# In[]
df = pd.concat([df3, df5], ignore_index=True)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="singular_0", y="max_noise_tolerance", height=4, hue="hidden_size", palette="crest")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=12, alpha=0.8, color="tab:cyan")
g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\sigma_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/singular_value_and_noise_width.jpg", dpi=1000)

g = sns.JointGrid(
    data=df[(df["attractors"]) & (df["fixed_point"])],
    x="eig_norm_0", y="max_noise_tolerance", height=4, hue="hidden_size", palette="flare")
g.ax_joint.set(ylim=(0.0, 1.4))
g.plot_joint(sns.scatterplot, s=12, alpha=0.8, color="tab:cyan")
g.plot_joint(sns.kdeplot, color="tab:cyan", zorder=1, levels=6, fill=True, alpha=0.5)
# g.plot_joint(sns.regplot, ci=95, order=1, scatter_kws={'s':12, 'alpha':0.6}, line_kws={"color":"royalblue"})
g.plot_marginals(sns.histplot, kde=True, color="tab:cyan")
g.ax_joint.set(xlabel= r"$\lambda_{1}(J_{*})$", ylabel=r"$\rho_{\rm max}$")
g.savefig("figures/eig_norm_and_noise_width.jpg", dpi=1000)


g = sns.displot(df, x="eig_norm_0", hue="fixed_point",
            col="hidden_size", kde=True, height=3, col_wrap=3, hue_order=[True, False], stat="density",
            bins=30)

g.set(xlabel= "maximum norm of eigenvalues", ylabel="count")
g.savefig("figures/eigenvalue_norm_and_hidden_size_layer_7.jpg", dpi=600)
g.set(xlim=(0.0, 1.6))


df1 = pd.read_csv(
    'logs/Minist/figure_400-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df1["figure_num"] = [400]*len(df1)

df2 = pd.read_csv(
    'logs/Minist/figure_600-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df2["figure_num"] = [600]*len(df2)

df3 = pd.read_csv(
    'logs/Minist/figure_800-noise_0.0-layer_7-width_512/fixed_point_evaluations.csv')
df3["figure_num"] = [800]*len(df3)

df = pd.concat([df1, df2, df3], ignore_index=True)
# df.rename(columns={'fixed_point' : 'fixed point', 'figure_num' : 'figure number'})
sns.axes_style("darkgrid")
g = sns.displot(df, x="eig_norm_0", hue="fixed_point",
            col="figure_num", kde=True, height=3, col_wrap=3, hue_order=[True, False], stat="density")
g.set(xlabel= "maximum norm of eigenvalues", ylabel="density")
g.savefig("figures/eigenvalue_norm_and_figure_num.jpg", dpi=600)
g.set(xlim=(0.0, None))





