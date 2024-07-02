import matplotlib.pyplot as plt
import numpy as np

def create_radar_chart(data, labels, ax, colors, model_name):
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    data += data[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_rscale('linear')
    ax.set_ylim(40, 100)
    ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_yticklabels(['40', '50', '60', '70', '80', '90', '100'])

    ax.plot(angles, data, color=colors, linewidth=2, linestyle='solid', label=model_name)
    ax.fill(angles, data, color=colors, alpha=0.2)

    ax.legend(loc='upper left', framealpha=1)

labels_anomaly_gpt = ['Overall Accuracy', 'i_AUROC', 'p_AUROC']
data_anomaly_gpt = [
    [87.83, 94.90, 95.63], 
    [81.9, 94.85, 81.2]
]
model_names_anomaly_gpt = ['AnomalyGPT on MVTec', 'AnomalyGPT on Grape Leaves']

labels_other_models = ['Overall Accuracy', 'Healthy Recall', 'Esca Recall']
data_other_models = [
    [97, 99, 94],       
    [71, 90, 50],       
    [96, 97, 93]        
]
model_names_other_models = ['Traditional CNN', 'CLIP', 'KAN']

fig = plt.figure(figsize=(12, 24))

# Top chart
ax_top = plt.subplot2grid((4, 2), (0, 0), colspan=2, polar=True)
create_radar_chart(data_anomaly_gpt[1], labels_anomaly_gpt, ax_top, '#1f77b4', model_names_anomaly_gpt[1])
ax_top.set_title('AnomalyGPT on Grape Leaves')

# Bottom charts
ax1 = plt.subplot2grid((4, 2), (1, 0), polar=True)
create_radar_chart(data_anomaly_gpt[0], labels_anomaly_gpt, ax1, '#7f7f7f', model_names_anomaly_gpt[0])
#ax1.set_title('AnomalyGPT on MVTec')

ax2 = plt.subplot2grid((4, 2), (1, 1), polar=True)
create_radar_chart(data_other_models[0], labels_other_models, ax2, '#2ca02c', model_names_other_models[0])
#ax2.set_title('Traditional CNN')

ax3 = plt.subplot2grid((4, 2), (2, 0), polar=True)
create_radar_chart(data_other_models[1], labels_other_models, ax3, '#d62728', model_names_other_models[1])
#ax3.set_title('CLIP')

ax4 = plt.subplot2grid((4, 2), (2, 1), polar=True)
create_radar_chart(data_other_models[2], labels_other_models, ax4, '#9467bd', model_names_other_models[2])
#ax4.set_title('KAN')

plt.tight_layout()
plt.show()
fig.savefig('/Users/michaelrodden/Desktop/radar_charts.png')
